from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from otllm.storage.database import Database

console = Console()


class HuggingFacePublisher:
    """Exports all experiment data and artifacts to a Hugging Face dataset repo."""

    def __init__(
        self,
        db: Database,
        repo_id: str,
        db_path: str = "otllm_data.db",
        report_glob: str = "otllm_report_*.html",
        private: bool = False,
    ) -> None:
        self.db = db
        self.repo_id = repo_id
        self.db_path = db_path
        self.report_glob = report_glob
        self.private = private

    def publish(
        self,
        experiment_ids: Optional[List[str]] = None,
        include_db: bool = True,
        include_reports: bool = True,
        commit_message: str = "Update OTllm experiment data",
    ) -> str:
        from huggingface_hub import HfApi

        api = HfApi()

        api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            private=self.private,
            exist_ok=True,
        )

        staging = tempfile.mkdtemp(prefix="otllm_hf_")

        try:
            self._export_data(staging, experiment_ids)

            if include_reports:
                self._collect_reports(staging)

            if include_db and os.path.exists(self.db_path):
                shutil.copy2(self.db_path, os.path.join(staging, "otllm_data.db"))

            self._write_readme(staging, experiment_ids)

            api.upload_folder(
                folder_path=staging,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )

            url = f"https://huggingface.co/datasets/{self.repo_id}"
            console.print(f"[green]Published to {url}[/green]")
            return url

        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _export_data(
        self, staging: str, experiment_ids: Optional[List[str]],
    ) -> None:
        data_dir = os.path.join(staging, "data")
        os.makedirs(data_dir, exist_ok=True)

        all_experiments = self.db.list_experiments()
        if experiment_ids:
            all_experiments = [
                e for e in all_experiments if e["id"] in experiment_ids
            ]

        if not all_experiments:
            console.print("[yellow]No experiments to export.[/yellow]")
            return

        exp_path = os.path.join(data_dir, "experiments.jsonl")
        nodes_path = os.path.join(data_dir, "nodes.jsonl")
        reanchor_path = os.path.join(data_dir, "reanchor_events.jsonl")

        node_count = 0
        event_count = 0

        with (
            open(exp_path, "w") as ef,
            open(nodes_path, "w") as nf,
            open(reanchor_path, "w") as rf,
        ):
            for exp_meta in all_experiments:
                exp_id = exp_meta["id"]

                try:
                    config, tree, meta = self.db.load_experiment(exp_id)
                except Exception as e:
                    console.print(f"[yellow]Skipping {exp_id}: {e}[/yellow]")
                    continue

                exp_record = {
                    **meta,
                    "config": config.to_dict(),
                }
                ef.write(json.dumps(exp_record, default=str) + "\n")

                for node in tree.bfs():
                    node_record = node.to_dict()
                    node_record["experiment_id"] = exp_id
                    nf.write(json.dumps(node_record, default=str) + "\n")
                    node_count += 1

                events = self.db.get_reanchor_events(exp_id)
                for ev in events:
                    rf.write(json.dumps(ev, default=str) + "\n")
                    event_count += 1

        console.print(
            f"  Exported {len(all_experiments)} experiments, "
            f"{node_count} nodes, {event_count} reanchor events"
        )

    def _collect_reports(self, staging: str) -> None:
        reports = glob.glob(self.report_glob)
        if not reports:
            return

        reports_dir = os.path.join(staging, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        for rpath in reports:
            shutil.copy2(rpath, os.path.join(reports_dir, os.path.basename(rpath)))

        console.print(f"  Collected {len(reports)} HTML reports")

    def _write_readme(
        self, staging: str, experiment_ids: Optional[List[str]],
    ) -> None:
        all_experiments = self.db.list_experiments()
        if experiment_ids:
            all_experiments = [
                e for e in all_experiments if e["id"] in experiment_ids
            ]

        total_experiments = len(all_experiments)
        completed = sum(1 for e in all_experiments if e.get("status") == "completed")
        regimes = {}
        for e in all_experiments:
            r = e.get("drift_regime")
            if r:
                regimes[r] = regimes.get(r, 0) + 1

        regime_lines = "\n".join(f"  - **{k}**: {v}" for k, v in sorted(regimes.items()))

        readme = f"""\
---
license: mit
task_categories:
  - text-generation
tags:
  - otllm
  - overthinking
  - rumination
  - llm-stability
  - cognitive-drift
  - thought-tree
pretty_name: OTllm Experiment Results
---

# OTllm: Overthinking LLM Experiment Data

Research dataset from [OTllm](https://github.com/hariprasad/OTllm) — a framework
for inducing and studying overthinking / rumination in LLMs.

## Dataset Summary

- **Total experiments**: {total_experiments}
- **Completed**: {completed}
- **Drift regimes observed**:
{regime_lines}

## Structure

```
data/
  experiments.jsonl    # Experiment configs + aggregate metrics
  nodes.jsonl          # Per-node thought text, context summaries, drift, sentiment
  reanchor_events.jsonl # Metacognitive self-correction decisions
reports/
  *.html               # Interactive HTML reports (D3.js tree + Chart.js)
otllm_data.db          # Raw SQLite database
```

## Schema

### experiments.jsonl
Each line is one experiment with fields: `id`, `name`, `status`, `config` (nested),
`total_nodes`, `max_depth_reached`, `final_drift`, `mean_drift`,
`gzip_compressibility`, `semantic_compressibility`, `drift_regime`, `mean_sentiment`.

### nodes.jsonl
Each line is one thought node: `id`, `experiment_id`, `parent_id`, `depth`,
`branch_index`, `text` (raw thought), `context_summary` (model's evolving understanding),
`drift_from_anchor`, `drift_from_parent`, `drift_velocity`, `sentiment`,
`contradiction_score`, `branching_factor`, `reanchor_decision`.

### reanchor_events.jsonl
Each line is a reanchoring check: `experiment_id`, `node_id`, `decision`
(continue/backtrack/reanchor), `drift_at_decision`, `was_drift_blind`.

## Key Metrics

| Metric | Description |
|---|---|
| **Context drift** | Cosine distance from current context embedding to anchor (root) |
| **Drift regime** | stable / oscillating / divergent / catastrophic |
| **Gzip compressibility** | Text repetition (lower = more repetitive) |
| **Semantic compressibility** | Embedding cluster ratio (lower = more semantic loops) |
| **Sentiment** | VADER compound score (-1 to +1) |
| **Drift blindness** | Model chose "continue" despite high drift (>0.4) |

## Citation

If you use this data, please cite the OTllm project.
"""
        with open(os.path.join(staging, "README.md"), "w") as f:
            f.write(readme)
