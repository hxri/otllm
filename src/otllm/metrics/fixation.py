from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from otllm.tree.thought_tree import ThoughtTree


def fixation_score(
    tree: ThoughtTree,
    similarity_threshold: float = 0.85,
) -> Dict:
    nodes_list = list(tree.bfs())
    if len(nodes_list) < 2:
        return {"score": 0.0, "fixation_pairs": [], "total_pairs": 0}

    embeddings = []
    node_ids = []
    for n in nodes_list:
        if n.embedding is not None:
            embeddings.append(np.array(n.embedding))
            node_ids.append(n.id)

    if len(embeddings) < 2:
        return {"score": 0.0, "fixation_pairs": [], "total_pairs": 0}

    matrix = np.array(embeddings)
    sim_matrix = matrix @ matrix.T

    parent_child: Set[Tuple[str, str]] = set()
    for nid in node_ids:
        node = tree.get_node(nid)
        if node.parent_id and node.parent_id in tree.nodes:
            parent_child.add((node.parent_id, nid))
            parent_child.add((nid, node.parent_id))

    fixation_pairs = []
    total_non_adjacent = 0
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            pair = (node_ids[i], node_ids[j])
            if pair in parent_child or (pair[1], pair[0]) in parent_child:
                continue
            total_non_adjacent += 1
            if sim_matrix[i, j] >= similarity_threshold:
                fixation_pairs.append((node_ids[i], node_ids[j], float(sim_matrix[i, j])))

    score = len(fixation_pairs) / total_non_adjacent if total_non_adjacent > 0 else 0.0
    return {
        "score": score,
        "fixation_pairs": fixation_pairs,
        "total_pairs": total_non_adjacent,
    }
