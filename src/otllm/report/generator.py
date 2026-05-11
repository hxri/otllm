from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from otllm.metrics.compressibility import gzip_compressibility, semantic_compressibility
from otllm.metrics.drift import classify_drift_regime, compute_drift_curve, count_drift_reversals
from otllm.metrics.fixation import fixation_score
from otllm.storage.database import Database


class HTMLReportGenerator:
    def __init__(self, db: Database, experiment_id: str) -> None:
        self.db = db
        self.experiment_id = experiment_id

    def generate(self, output_path: str) -> None:
        config, tree, meta = self.db.load_experiment(self.experiment_id)
        reanchor_events = self.db.get_reanchor_events(self.experiment_id)

        tree_data = self._build_tree_data(tree)
        drift_curve = compute_drift_curve(tree)
        sentiment_curve = [(i, n.sentiment) for i, n in enumerate(tree.bfs()) if n.sentiment is not None]

        drift_values = [n.drift_from_anchor for n in tree.bfs() if n.drift_from_anchor is not None]
        texts = [n.text for n in tree.bfs()]
        embeddings = [np.array(n.embedding) for n in tree.bfs() if n.embedding is not None]

        metrics: Dict[str, Any] = {}
        if drift_values:
            metrics["drift_regime"] = classify_drift_regime(drift_values)
            metrics["final_drift"] = drift_values[-1]
            metrics["mean_drift"] = float(np.mean(drift_values))
            metrics["drift_reversals"] = count_drift_reversals(drift_values)
        if texts:
            metrics["gzip_compressibility"] = gzip_compressibility(texts)
        if len(embeddings) >= 2:
            sem = semantic_compressibility(embeddings)
            metrics["semantic_compressibility"] = sem["ratio"]
            metrics["semantic_clusters"] = sem["n_clusters"]

        sentiments = [n.sentiment for n in tree.bfs() if n.sentiment is not None]
        if sentiments:
            metrics["mean_sentiment"] = float(np.mean(sentiments))

        if embeddings:
            fix = fixation_score(tree)
            metrics["fixation_score"] = fix["score"]

        data = {
            "experiment": meta,
            "config": config.to_dict(),
            "tree": tree_data,
            "drift_curve": drift_curve,
            "sentiment_curve": sentiment_curve,
            "metrics": metrics,
            "reanchor_events": reanchor_events,
            "nodes": [
                {
                    "id": n.id,
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "text": n.text,
                    "context_summary": n.context_summary,
                    "drift_from_anchor": n.drift_from_anchor,
                    "drift_velocity": n.drift_velocity,
                    "sentiment": n.sentiment,
                    "contradiction_score": n.contradiction_score,
                    "reanchor_decision": n.reanchor_decision,
                    "generation_time_ms": n.generation_time_ms,
                    "token_count": n.token_count,
                }
                for n in tree.bfs()
            ],
        }

        template_path = Path(__file__).parent / "templates" / "report.html.j2"
        template_str = template_path.read_text()
        html = template_str.replace("{{ DATA_JSON }}", json.dumps(data, default=str))

        Path(output_path).write_text(html)

    def _build_tree_data(self, tree) -> Dict:
        if tree.root is None:
            return {}

        def build(node_id: str) -> Dict:
            node = tree.get_node(node_id)
            children_data = [build(c.id) for c in tree.get_children(node_id)]
            return {
                "id": node.id,
                "name": node.text[:60] + ("..." if len(node.text) > 60 else ""),
                "depth": node.depth,
                "drift": node.drift_from_anchor,
                "sentiment": node.sentiment,
                "reanchor": node.reanchor_decision,
                "children": children_data,
            }

        return build(tree.root.id)
