from __future__ import annotations

import json
from collections import deque
from typing import Dict, Iterator, List, Optional

import numpy as np

from otllm.tree.node import ThoughtNode


class ThoughtTree:
    def __init__(self) -> None:
        self.root: Optional[ThoughtNode] = None
        self.nodes: Dict[str, ThoughtNode] = {}
        self.children: Dict[str, List[str]] = {}
        self.revisit_edges: Dict[str, List[str]] = {}
        self.anchor_embedding: Optional[np.ndarray] = None

    def add_node(self, node: ThoughtNode) -> None:
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.root = node
        else:
            self.children.setdefault(node.parent_id, []).append(node.id)
        self.children.setdefault(node.id, [])

    def get_node(self, node_id: str) -> ThoughtNode:
        return self.nodes[node_id]

    def get_children(self, node_id: str) -> List[ThoughtNode]:
        return [self.nodes[cid] for cid in self.children.get(node_id, [])]

    def get_ancestors(self, node_id: str) -> List[ThoughtNode]:
        ancestors = []
        current = self.nodes[node_id]
        while current.parent_id is not None:
            current = self.nodes[current.parent_id]
            ancestors.append(current)
        return ancestors

    def get_path_to_root(self, node_id: str) -> List[ThoughtNode]:
        path = [self.nodes[node_id]]
        path.extend(self.get_ancestors(node_id))
        path.reverse()
        return path

    def get_leaves(self) -> List[ThoughtNode]:
        return [n for n in self.nodes.values() if not self.children.get(n.id)]

    def get_all_at_depth(self, depth: int) -> List[ThoughtNode]:
        return [n for n in self.nodes.values() if n.depth == depth]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def max_depth_reached(self) -> int:
        if not self.nodes:
            return 0
        return max(n.depth for n in self.nodes.values())

    def add_revisit_edge(self, source_id: str, target_id: str) -> None:
        self.revisit_edges.setdefault(source_id, []).append(target_id)
        self.nodes[source_id].revisit_edges.append(target_id)

    def compressed_view(self, current_node_id: str, include_drift: bool = True) -> str:
        path = self.get_path_to_root(current_node_id)
        lines = []
        for i, node in enumerate(path):
            prefix = "  " * node.depth
            drift_str = ""
            if include_drift and node.drift_from_anchor is not None:
                drift_str = f" [drift: {node.drift_from_anchor:.3f}]"
            summary = node.context_summary[:120] if node.context_summary else node.text[:120]
            lines.append(f"{prefix}[{i}] {summary}{drift_str}")
        return "\n".join(lines)

    def bfs(self) -> Iterator[ThoughtNode]:
        if self.root is None:
            return
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            for child in self.get_children(node.id):
                queue.append(child)

    def dfs(self) -> Iterator[ThoughtNode]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            for child in reversed(self.get_children(node.id)):
                stack.append(child)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.bfs()],
            "revisit_edges": {k: v for k, v in self.revisit_edges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThoughtTree:
        tree = cls()
        for nd in data["nodes"]:
            tree.add_node(ThoughtNode.from_dict(nd))
        for src, targets in data.get("revisit_edges", {}).items():
            for tgt in targets:
                tree.revisit_edges.setdefault(src, []).append(tgt)
        return tree

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
