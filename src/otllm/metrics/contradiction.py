from __future__ import annotations

from typing import List

import numpy as np

from otllm.metrics.drift import cosine_distance


def contradiction_score(
    node_embedding: np.ndarray,
    ancestor_embeddings: List[np.ndarray],
) -> float:
    if not ancestor_embeddings:
        return 0.0
    similarities = [
        float(np.dot(node_embedding, anc) / (np.linalg.norm(node_embedding) * np.linalg.norm(anc)))
        for anc in ancestor_embeddings
        if np.linalg.norm(anc) > 0
    ]
    if not similarities:
        return 0.0
    negative_sims = [s for s in similarities if s < 0]
    if not negative_sims:
        return 0.0
    return float(abs(min(negative_sims)))
