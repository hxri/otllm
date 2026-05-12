"""Parallel experiment runner for multi-GPU vLLM cluster.

Architecture:
    GPU 0: vLLM :8000 ─── workers[0] ─── experiments assigned round-robin
    GPU 1: vLLM :8001 ─── workers[1]
    GPU 2: vLLM :8002 ─── workers[2]
    GPU 3: vLLM :8003 ─── workers[3]
    CPU:   sentence-transformers embedder (shared, thread-safe)

Usage:
    # 1. Launch vLLM servers (see launch_vllm_servers.sh)
    # 2. Run experiments
    python experiments/gpu_cluster.py
    python experiments/gpu_cluster.py --gpus 2 --category blind_vs_informed
    python experiments/gpu_cluster.py --quick
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()


def scoped_experiment_name(exp_name: str, base_name: str) -> str:
    return f"{exp_name}::{base_name}"


def load_completed_experiments(db_path: str, exp_name: str) -> set[str]:
    """Return base experiment names already completed for this exp_name scope."""
    if not os.path.exists(db_path):
        return set()

    prefix = f"{exp_name}::%"
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        rows = cur.execute(
            """
            WITH latest AS (
                SELECT e1.name, e1.status
                FROM experiments e1
                JOIN (
                    SELECT name, MAX(created_at) AS max_created
                    FROM experiments
                    WHERE name LIKE ?
                    GROUP BY name
                ) e2
                ON e1.name = e2.name AND e1.created_at = e2.max_created
            )
            SELECT name
            FROM latest
            WHERE status = 'completed'
            """,
            (prefix,),
        ).fetchall()
    finally:
        con.close()

    completed = set()
    scope_prefix = f"{exp_name}::"
    for (full_name,) in rows:
        if full_name.startswith(scope_prefix):
            completed.add(full_name[len(scope_prefix):])
    return completed


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


PROMPTS = {
    "job":        "Should I take this job offer?",
    "friendship": "My friend hasn't texted me back in 3 days.",
    "mistake":    "I think I made a serious mistake at work yesterday.",
    "health":     "I've been having headaches every day this week.",
    "breakup":    "My partner said we need to talk tonight.",
}


@dataclass
class ExperimentDef:
    name: str
    prompt_key: str
    mode: str = "branching"
    max_depth: int = 5
    max_branches: int = 3
    max_nodes: int = 50
    strategy: str = "recursive"
    anxiety: float = 0.5
    reanchor_blind: bool = True
    reanchor_enabled: bool = True
    category: str = ""


def build_all_experiments() -> List[ExperimentDef]:
    experiments = []

    for blind in [True, False]:
        tag = "blind" if blind else "informed"
        for key in PROMPTS:
            experiments.append(ExperimentDef(
                name=f"reanchor-{tag}-{key}", prompt_key=key,
                reanchor_blind=blind, category="blind_vs_informed",
            ))

    for mode in ["linear", "branching", "cyclic"]:
        for key in PROMPTS:
            experiments.append(ExperimentDef(
                name=f"mode-{mode}-{key}", prompt_key=key,
                mode=mode, category="tree_mode",
            ))

    for anxiety in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for key in PROMPTS:
            experiments.append(ExperimentDef(
                name=f"anxiety-{anxiety:.1f}-{key}", prompt_key=key,
                strategy="anxiety_amplifying", anxiety=anxiety,
                category="anxiety_dose",
            ))

    for strategy in ["recursive", "anxiety_amplifying", "multi_persona", "termination_suppression"]:
        for key in PROMPTS:
            experiments.append(ExperimentDef(
                name=f"strategy-{strategy}-{key}", prompt_key=key,
                strategy=strategy, anxiety=0.7,
                category="strategy",
            ))

    for depth in [3, 5, 8, 12]:
        experiments.append(ExperimentDef(
            name=f"depth-{depth}-job", prompt_key="job",
            max_depth=depth, category="depth_scaling",
        ))

    return experiments


def run_single_experiment(
    exp: ExperimentDef,
    vllm_url: str,
    db_path: str,
    model_name: str,
    exp_name: str,
    embedder_device: str,
) -> Dict:
    from otllm.config import ExperimentConfig
    from otllm.engine.runner import ExperimentRunner
    from otllm.models.embeddings import SentenceTransformerEmbedder
    from otllm.models.vllm_llm import VLLMBackend
    from otllm.storage.database import Database

    config = ExperimentConfig(
        name=scoped_experiment_name(exp_name, exp.name),
        prompt=PROMPTS[exp.prompt_key],
        model_name=model_name,
        mode=exp.mode,
        max_depth=exp.max_depth,
        max_branches_per_node=exp.max_branches,
        max_nodes=exp.max_nodes,
        induction_strategy=exp.strategy,
        anxiety_intensity=exp.anxiety,
        reanchor_enabled=exp.reanchor_enabled,
        reanchor_blind=exp.reanchor_blind,
        db_path=db_path,
    )

    llm = VLLMBackend(base_url=vllm_url, model=model_name, enable_thinking=True)
    embedder = SentenceTransformerEmbedder(device=embedder_device)
    db = Database(db_path)

    t0 = time.time()
    try:
        runner = ExperimentRunner(config, llm, embedder, db)
        result = runner.run()
        elapsed = time.time() - t0
        m = result.aggregate_metrics
        return {
            "name": exp.name,
            "category": exp.category,
            "id": result.experiment_id,
            "nodes": m.get("total_nodes", 0),
            "drift_regime": m.get("drift_regime", "?"),
            "final_drift": m.get("final_drift"),
            "mean_drift": m.get("mean_drift"),
            "gzip": m.get("gzip_compressibility"),
            "sentiment": m.get("mean_sentiment"),
            "fixation": m.get("fixation_score"),
            "elapsed": elapsed,
            "gpu": vllm_url,
            "status": "ok",
        }
    except Exception as e:
        return {
            "name": exp.name,
            "category": exp.category,
            "status": "failed",
            "error": str(e),
            "elapsed": time.time() - t0,
            "gpu": vllm_url,
        }
    finally:
        llm.close()
        db.close()


def _worker(args):
    exp, vllm_url, db_path, model_name, exp_name, embedder_device = args
    return run_single_experiment(exp, vllm_url, db_path, model_name, exp_name, embedder_device)


def run_parallel(
    experiments: List[ExperimentDef],
    gpu_urls: List[str],
    db_path: str,
    model_name: str,
    exp_name: str,
    embedder_device: str,
) -> List[Dict]:
    tasks = []
    for i, exp in enumerate(experiments):
        url = gpu_urls[i % len(gpu_urls)]
        tasks.append((exp, url, db_path, model_name, exp_name, embedder_device))

    results = []
    total = len(tasks)
    elapsed_times: List[float] = []

    console.print(f"[bold]Running {total} experiments across {len(gpu_urls)} GPUs[/bold]\n")

    batch_start = time.time()

    with ProcessPoolExecutor(max_workers=len(gpu_urls)) as executor:
        futures = {executor.submit(_worker, t): t[0].name for t in tasks}

        for future in as_completed(futures):
            name = futures[future]
            result = future.result()
            results.append(result)
            elapsed_times.append(result.get("elapsed", 0))

            done = len(results)
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining_experiments = total - done
            remaining_parallel = remaining_experiments / len(gpu_urls)
            eta_seconds = remaining_parallel * avg_time
            eta_str = _format_eta(eta_seconds)

            if result["status"] == "ok":
                console.print(
                    f"[cyan][{done}/{total}][/cyan] {name} -> "
                    f"{result['drift_regime'].upper()} drift={result.get('final_drift', 0):.3f} "
                    f"({result['elapsed']:.0f}s on {result['gpu']}) "
                    f"[dim]ETA: {eta_str}[/dim]"
                )
            else:
                console.print(
                    f"[red][{done}/{total}] {name} FAILED: {result.get('error', '?')}[/red] "
                    f"[dim]ETA: {eta_str}[/dim]"
                )

    total_elapsed = time.time() - batch_start
    console.print(f"\n[bold]Batch completed in {_format_eta(total_elapsed)}[/bold]")

    return results


def print_results(results: List[Dict]) -> None:
    table = Table(title="Parallel Batch Results")
    table.add_column("Experiment", style="cyan", max_width=30)
    table.add_column("Category", style="dim")
    table.add_column("Regime", style="white")
    table.add_column("Final Drift", style="red")
    table.add_column("Gzip", style="green")
    table.add_column("Sentiment", style="blue")
    table.add_column("Time", style="dim")
    table.add_column("GPU", style="dim")

    for r in sorted(results, key=lambda x: (x["category"], x["name"])):
        if r["status"] != "ok":
            table.add_row(r["name"], r["category"], "[red]FAILED[/red]", "-", "-", "-", f"{r['elapsed']:.0f}s", r.get("gpu", "?"))
            continue

        regime = r.get("drift_regime", "?")
        color = {"stable": "green", "oscillating": "yellow", "divergent": "red", "catastrophic": "bold red"}.get(regime, "white")
        table.add_row(
            r["name"],
            r["category"],
            f"[{color}]{regime.upper()}[/{color}]",
            f"{r['final_drift']:.3f}" if r.get("final_drift") is not None else "-",
            f"{r['gzip']:.3f}" if r.get("gzip") is not None else "-",
            f"{r['sentiment']:.3f}" if r.get("sentiment") is not None else "-",
            f"{r['elapsed']:.0f}s",
            r.get("gpu", "?").split(":")[-1],
        )
    console.print(table)

    # Category summaries
    categories: Dict[str, List[Dict]] = {}
    for r in results:
        if r["status"] == "ok":
            categories.setdefault(r["category"], []).append(r)

    console.print("\n[bold]Category Summaries[/bold]")
    for cat, runs in sorted(categories.items()):
        drifts = [r["final_drift"] for r in runs if r.get("final_drift") is not None]
        regimes = [r["drift_regime"] for r in runs if r.get("drift_regime")]
        if drifts:
            import numpy as np
            console.print(f"\n  [cyan]{cat}[/cyan] ({len(runs)} runs)")
            console.print(f"    Drift: mean={np.mean(drifts):.3f}  std={np.std(drifts):.3f}  min={min(drifts):.3f}  max={max(drifts):.3f}")
            for regime in ["stable", "oscillating", "divergent", "catastrophic"]:
                count = regimes.count(regime)
                if count:
                    console.print(f"    {regime}: {count}/{len(regimes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel OTllm experiments on multi-GPU vLLM cluster")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs / vLLM instances")
    parser.add_argument("--base-port", type=int, default=8000, help="First vLLM server port")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Model name in vLLM")
    parser.add_argument("--db", type=str, default="otllm_parallel.db", help="Database path")
    parser.add_argument("--exp-name", type=str, default="default", help="Experiment namespace within shared DB")
    parser.add_argument("--resume", action="store_true", help="Skip already completed experiments for this --exp-name")
    parser.add_argument("--embedder-device", type=str, default="cpu", help="Sentence-transformer device: cpu or cuda")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only: blind_vs_informed, tree_mode, anxiety_dose, strategy, depth_scaling")
    parser.add_argument("--quick", action="store_true", help="Run 4 experiments (1 per GPU) for smoke test")
    args = parser.parse_args()

    gpu_urls = [f"http://localhost:{args.base_port + i}" for i in range(args.gpus)]

    all_experiments = build_all_experiments()

    if args.quick:
        experiments = all_experiments[:args.gpus]
    elif args.category:
        experiments = [e for e in all_experiments if e.category == args.category]
    else:
        experiments = all_experiments

    if not experiments:
        console.print("[red]No experiments matched filter.[/red]")
        sys.exit(1)

    skipped = 0
    if args.resume:
        completed = load_completed_experiments(args.db, args.exp_name)
        if completed:
            before = len(experiments)
            experiments = [e for e in experiments if e.name not in completed]
            skipped = before - len(experiments)

    total_est_minutes = len(experiments) * 8 / args.gpus / 60
    console.print(f"[bold]OTllm Parallel Batch[/bold]")
    console.print(f"  Exp Name: {args.exp_name}")
    console.print(f"  GPUs: {args.gpus} ({', '.join(gpu_urls)})")
    console.print(f"  Embedder device: {args.embedder_device}")
    console.print(f"  Experiments: {len(experiments)}")
    if args.resume:
        console.print(f"  Resume: enabled (skipping {skipped} completed)")
    console.print(f"  Estimated time: ~{total_est_minutes:.0f} min")
    console.print()

    if not experiments:
        console.print("[green]Nothing to run. All experiments for this exp name are already completed.[/green]")
        sys.exit(0)

    results = run_parallel(experiments, gpu_urls, args.db, args.model, args.exp_name, args.embedder_device)
    print_results(results)
