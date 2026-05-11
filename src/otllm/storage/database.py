from __future__ import annotations

import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from otllm.config import ExperimentConfig
from otllm.tree.node import ThoughtNode
from otllm.tree.thought_tree import ThoughtTree

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'running',
    total_nodes INTEGER,
    max_depth_reached INTEGER,
    final_drift REAL,
    mean_drift REAL,
    gzip_compressibility REAL,
    semantic_compressibility REAL,
    drift_regime TEXT,
    mean_sentiment REAL
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    parent_id TEXT,
    depth INTEGER NOT NULL,
    branch_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    context_summary TEXT NOT NULL,
    embedding BLOB,
    context_embedding BLOB,
    drift_from_anchor REAL,
    drift_from_parent REAL,
    drift_velocity REAL,
    sentiment REAL,
    contradiction_score REAL,
    branching_factor INTEGER DEFAULT 0,
    reanchor_decision TEXT,
    reanchor_target_id TEXT,
    timestamp TEXT NOT NULL,
    generation_time_ms REAL,
    token_count INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS revisit_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS reanchor_events (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    drift_at_decision REAL,
    drift_after REAL,
    durability INTEGER,
    was_drift_blind INTEGER,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
"""


def _embed_to_blob(arr: Optional[List[float]]) -> Optional[bytes]:
    if arr is None:
        return None
    buf = io.BytesIO()
    np.save(buf, np.array(arr, dtype=np.float32))
    return buf.getvalue()


def _blob_to_embed(blob: Optional[bytes]) -> Optional[List[float]]:
    if blob is None:
        return None
    buf = io.BytesIO(blob)
    return np.load(buf).tolist()


class Database:
    def __init__(self, db_path: str = "otllm_data.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def create_experiment(self, config: ExperimentConfig) -> str:
        exp_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO experiments (id, name, config_json, created_at, status) VALUES (?, ?, ?, ?, ?)",
            (exp_id, config.name, json.dumps(config.to_dict()), datetime.now(timezone.utc).isoformat(), "running"),
        )
        self._conn.commit()
        return exp_id

    def save_node(self, experiment_id: str, node: ThoughtNode) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
            (id, experiment_id, parent_id, depth, branch_index,
             text, context_summary, embedding, context_embedding,
             drift_from_anchor, drift_from_parent, drift_velocity,
             sentiment, contradiction_score, branching_factor,
             reanchor_decision, reanchor_target_id,
             timestamp, generation_time_ms, token_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                node.id, experiment_id, node.parent_id, node.depth, node.branch_index,
                node.text, node.context_summary,
                _embed_to_blob(node.embedding), _embed_to_blob(node.context_embedding),
                node.drift_from_anchor, node.drift_from_parent, node.drift_velocity,
                node.sentiment, node.contradiction_score, node.branching_factor,
                node.reanchor_decision, node.reanchor_target_id,
                node.timestamp, node.generation_time_ms, node.token_count,
            ),
        )
        self._conn.commit()

    def save_reanchor_event(
        self, experiment_id: str, node_id: str, decision: str,
        drift_at_decision: float, drift_after: Optional[float] = None,
        durability: Optional[int] = None, was_drift_blind: bool = False,
    ) -> None:
        self._conn.execute(
            "INSERT INTO reanchor_events (id, experiment_id, node_id, decision, drift_at_decision, drift_after, durability, was_drift_blind) VALUES (?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex[:12], experiment_id, node_id, decision, drift_at_decision, drift_after, durability, int(was_drift_blind)),
        )
        self._conn.commit()

    def save_revisit_edge(self, experiment_id: str, source_id: str, target_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO revisit_edges (source_id, target_id, experiment_id) VALUES (?,?,?)",
            (source_id, target_id, experiment_id),
        )
        self._conn.commit()

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        updates = {"status": status}
        if status in ("completed", "failed"):
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()
        sets = ", ".join(f"{k} = ?" for k in updates)
        self._conn.execute(
            f"UPDATE experiments SET {sets} WHERE id = ?",
            (*updates.values(), experiment_id),
        )
        self._conn.commit()

    def save_aggregate_metrics(self, experiment_id: str, metrics: dict) -> None:
        cols = ["total_nodes", "max_depth_reached", "final_drift", "mean_drift",
                "gzip_compressibility", "semantic_compressibility", "drift_regime", "mean_sentiment"]
        sets = ", ".join(f"{c} = ?" for c in cols if c in metrics)
        vals = [metrics[c] for c in cols if c in metrics]
        if sets:
            self._conn.execute(f"UPDATE experiments SET {sets} WHERE id = ?", (*vals, experiment_id))
            self._conn.commit()

    def load_experiment(self, experiment_id: str) -> Tuple[ExperimentConfig, ThoughtTree, dict]:
        row = self._conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        config = ExperimentConfig.from_dict(json.loads(row["config_json"]))
        tree = ThoughtTree()

        node_rows = self._conn.execute(
            "SELECT * FROM nodes WHERE experiment_id = ? ORDER BY depth, branch_index", (experiment_id,)
        ).fetchall()
        for nr in node_rows:
            node = ThoughtNode(
                id=nr["id"], parent_id=nr["parent_id"], depth=nr["depth"],
                branch_index=nr["branch_index"], text=nr["text"],
                context_summary=nr["context_summary"],
                embedding=_blob_to_embed(nr["embedding"]),
                context_embedding=_blob_to_embed(nr["context_embedding"]),
                drift_from_anchor=nr["drift_from_anchor"],
                drift_from_parent=nr["drift_from_parent"],
                drift_velocity=nr["drift_velocity"],
                sentiment=nr["sentiment"],
                contradiction_score=nr["contradiction_score"],
                branching_factor=nr["branching_factor"],
                reanchor_decision=nr["reanchor_decision"],
                reanchor_target_id=nr["reanchor_target_id"],
                timestamp=nr["timestamp"],
                generation_time_ms=nr["generation_time_ms"] or 0.0,
                token_count=nr["token_count"] or 0,
            )
            tree.add_node(node)

        edges = self._conn.execute(
            "SELECT * FROM revisit_edges WHERE experiment_id = ?", (experiment_id,)
        ).fetchall()
        for e in edges:
            tree.revisit_edges.setdefault(e["source_id"], []).append(e["target_id"])

        if tree.root and tree.root.context_embedding:
            tree.anchor_embedding = np.array(tree.root.context_embedding)

        exp_meta = {
            "id": row["id"], "name": row["name"], "status": row["status"],
            "created_at": row["created_at"], "completed_at": row["completed_at"],
            "total_nodes": row["total_nodes"], "max_depth_reached": row["max_depth_reached"],
            "final_drift": row["final_drift"], "mean_drift": row["mean_drift"],
            "gzip_compressibility": row["gzip_compressibility"],
            "semantic_compressibility": row["semantic_compressibility"],
            "drift_regime": row["drift_regime"], "mean_sentiment": row["mean_sentiment"],
        }
        return config, tree, exp_meta

    def list_experiments(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT id, name, status, created_at, total_nodes, max_depth_reached, final_drift, drift_regime, mean_sentiment FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_experiment(self, experiment_id: str) -> None:
        self._conn.execute("DELETE FROM reanchor_events WHERE experiment_id = ?", (experiment_id,))
        self._conn.execute("DELETE FROM revisit_edges WHERE experiment_id = ?", (experiment_id,))
        self._conn.execute("DELETE FROM nodes WHERE experiment_id = ?", (experiment_id,))
        self._conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        self._conn.commit()

    def get_reanchor_events(self, experiment_id: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM reanchor_events WHERE experiment_id = ? ORDER BY rowid", (experiment_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
