from __future__ import annotations

from typing import List, Tuple

import numpy as np

from otllm.tree.thought_tree import ThoughtTree


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def compute_drift_from_anchor(
    node_embedding: np.ndarray, anchor_embedding: np.ndarray,
) -> float:
    return cosine_distance(node_embedding, anchor_embedding)


def compute_drift_velocity(
    current_drift: float, parent_drift: float,
) -> float:
    return current_drift - parent_drift


def compute_drift_curve(tree: ThoughtTree) -> List[Tuple[int, float]]:
    curve = []
    for i, node in enumerate(tree.bfs()):
        if node.drift_from_anchor is not None:
            curve.append((i, node.drift_from_anchor))
    return curve


def count_drift_reversals(drift_values: List[float]) -> int:
    if len(drift_values) < 3:
        return 0
    reversals = 0
    for i in range(1, len(drift_values) - 1):
        v_before = drift_values[i] - drift_values[i - 1]
        v_after = drift_values[i + 1] - drift_values[i]
        if v_before * v_after < 0:
            reversals += 1
    return reversals


def classify_drift_regime(drift_values: List[float]) -> str:
    if len(drift_values) < 3:
        return "insufficient_data"

    arr = np.array(drift_values)
    velocities = np.diff(arr)

    max_drift = float(np.max(arr))
    std_drift = float(np.std(arr))

    # Only positive velocity spikes count as catastrophic —
    # negative spikes are corrections (snapping back to topic), which is good
    positive_velocities = velocities[velocities > 0]
    max_divergence_spike = float(np.max(positive_velocities)) if len(positive_velocities) > 0 else 0.0
    if max_divergence_spike > 0.3:
        return "catastrophic"

    positive_frac = float(np.mean(velocities > 0.001))
    if positive_frac > 0.8 and max_drift > 0.3:
        return "divergent"

    reversals = count_drift_reversals(drift_values)
    reversal_rate = reversals / len(drift_values)
    if reversal_rate > 0.3 and std_drift > 0.05:
        return "oscillating"

    return "stable"
