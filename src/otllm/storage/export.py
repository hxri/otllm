from __future__ import annotations

import csv
import json
from typing import Optional

from otllm.storage.database import Database


def export_to_json(
    experiment_id: str, db: Database, output_path: str,
    include_embeddings: bool = False,
) -> None:
    config, tree, meta = db.load_experiment(experiment_id)
    data = {
        "experiment": meta,
        "config": config.to_dict(),
        "tree": tree.to_dict(),
        "reanchor_events": db.get_reanchor_events(experiment_id),
    }
    if include_embeddings:
        for node_dict in data["tree"]["nodes"]:
            node = tree.nodes.get(node_dict["id"])
            if node:
                node_dict["embedding"] = node.embedding
                node_dict["context_embedding"] = node.context_embedding

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def export_summary_csv(db: Database, output_path: str) -> None:
    experiments = db.list_experiments()
    if not experiments:
        return
    fields = list(experiments[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(experiments)
