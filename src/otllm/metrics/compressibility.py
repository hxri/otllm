from __future__ import annotations

import gzip
from typing import Dict, List

import numpy as np


def gzip_compressibility(texts: List[str]) -> float:
    if not texts:
        return 1.0
    concatenated = "\n".join(texts).encode("utf-8")
    if len(concatenated) == 0:
        return 1.0
    compressed = gzip.compress(concatenated)
    return len(compressed) / len(concatenated)


def semantic_compressibility(
    embeddings: List[np.ndarray], eps: float = 0.15,
) -> Dict:
    if len(embeddings) < 2:
        return {"n_clusters": len(embeddings), "n_nodes": len(embeddings), "ratio": 1.0}

    from sklearn.cluster import DBSCAN

    matrix = np.array(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(matrix)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    return {
        "n_clusters": n_clusters,
        "n_nodes": len(embeddings),
        "n_noise": n_noise,
        "ratio": (n_clusters + n_noise) / len(embeddings) if len(embeddings) > 0 else 1.0,
        "labels": labels.tolist(),
    }
