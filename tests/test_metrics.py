import numpy as np

from otllm.metrics.drift import (
    classify_drift_regime,
    compute_drift_from_anchor,
    compute_drift_velocity,
    cosine_distance,
    count_drift_reversals,
)
from otllm.metrics.compressibility import gzip_compressibility, semantic_compressibility
from otllm.metrics.contradiction import contradiction_score


def test_cosine_distance_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_distance(v, v) < 1e-6


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_distance(a, b) - 1.0) < 1e-6


def test_drift_velocity():
    assert compute_drift_velocity(0.5, 0.3) > 0
    assert compute_drift_velocity(0.2, 0.4) < 0


def test_drift_regime_stable():
    values = [0.05, 0.06, 0.05, 0.07, 0.06, 0.05, 0.06, 0.07, 0.05, 0.06]
    assert classify_drift_regime(values) == "stable"


def test_drift_regime_divergent():
    values = [0.1, 0.15, 0.22, 0.30, 0.38, 0.45, 0.52, 0.60, 0.68, 0.75]
    assert classify_drift_regime(values) == "divergent"


def test_drift_reversals():
    values = [0.1, 0.3, 0.2, 0.4, 0.1, 0.5]
    assert count_drift_reversals(values) >= 2


def test_gzip_compressibility_repetitive():
    texts = ["hello world this is a longer repeated sentence for testing"] * 50
    ratio = gzip_compressibility(texts)
    unique_texts = [f"completely unique text number {i} with very different and varied content about topic {i * 7}" for i in range(50)]
    ratio_unique = gzip_compressibility(unique_texts)
    assert ratio < ratio_unique


def test_semantic_compressibility():
    vecs = [np.random.randn(10) for _ in range(5)]
    result = semantic_compressibility(vecs, eps=0.5)
    assert "n_clusters" in result
    assert "ratio" in result


def test_contradiction_score_no_ancestors():
    node_emb = np.array([1.0, 0.0])
    assert contradiction_score(node_emb, []) == 0.0


def test_contradiction_score_opposing():
    node_emb = np.array([1.0, 0.0])
    ancestor_emb = [np.array([-1.0, 0.0])]
    score = contradiction_score(node_emb, ancestor_emb)
    assert score > 0.9
