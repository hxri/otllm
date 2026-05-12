"""Batch comparison experiments for OTllm.

Systematically varies key parameters to answer:
1. Is drift blindness real? (blind vs informed reanchoring)
2. Does tree structure matter? (linear vs branching vs cyclic)
3. Does anxiety intensity drive instability? (0.0 vs 0.5 vs 1.0)
4. Is catastrophic drift prompt-specific? (multiple prompts)
5. Does induction strategy matter? (recursive vs anxiety vs multi-persona)

Usage:
    python experiments/batch_comparison.py [--quick]

    --quick: Run a minimal subset (3 experiments) for smoke testing
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List

from rich.console import Console
from rich.table import Table

from otllm.config import ExperimentConfig
from otllm.engine.runner import ExperimentRunner
from otllm.models.embeddings import SentenceTransformerEmbedder
from otllm.models.ollama_llm import OllamaLLM
from otllm.storage.database import Database

console = Console()


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


# ---------------------------------------------------------------------------
# Prompts — chosen to cover different emotional/cognitive domains
# ---------------------------------------------------------------------------
PROMPTS = {
    "job":        "Should I take this job offer?",
    "friendship": "My friend hasn't texted me back in 3 days.",
    "mistake":    "I think I made a serious mistake at work yesterday.",
    "health":     "I've been having headaches every day this week.",
    "breakup":    "My partner said we need to talk tonight.",
}

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
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


def build_experiments() -> List[ExperimentDef]:
    experiments = []

    # ----- Category 1: Blind vs Informed (same prompt, same settings) -----
    for blind in [True, False]:
        tag = "blind" if blind else "informed"
        experiments.append(ExperimentDef(
            name=f"reanchor-{tag}-job",
            prompt_key="job",
            reanchor_blind=blind,
            category="blind_vs_informed",
        ))

    # ----- Category 2: Tree mode comparison -----
    for mode in ["linear", "branching", "cyclic"]:
        experiments.append(ExperimentDef(
            name=f"mode-{mode}-job",
            prompt_key="job",
            mode=mode,
            category="tree_mode",
        ))

    # ----- Category 3: Anxiety dose-response -----
    for anxiety in [0.0, 0.3, 0.6, 1.0]:
        experiments.append(ExperimentDef(
            name=f"anxiety-{anxiety:.1f}-job",
            prompt_key="job",
            strategy="anxiety_amplifying",
            anxiety=anxiety,
            category="anxiety_dose",
        ))

    # ----- Category 4: Cross-prompt stability -----
    for key in PROMPTS:
        experiments.append(ExperimentDef(
            name=f"prompt-{key}",
            prompt_key=key,
            category="cross_prompt",
        ))

    # ----- Category 5: Induction strategy comparison -----
    for strategy in ["recursive", "anxiety_amplifying", "multi_persona", "termination_suppression"]:
        experiments.append(ExperimentDef(
            name=f"strategy-{strategy}-job",
            prompt_key="job",
            strategy=strategy,
            anxiety=0.7,
            category="strategy",
        ))

    return experiments


QUICK_SET = [
    "reanchor-blind-job",
    "reanchor-informed-job",
    "mode-linear-job",
]


def run_batch(experiments: List[ExperimentDef], db_path: str) -> None:
    llm = OllamaLLM(model="qwen3:4b", enable_thinking=True)
    health = llm.check_health()
    if not health.get("connected"):
        console.print("[red]Ollama not running. Start with: ollama serve[/red]")
        sys.exit(1)
    if not health.get("model_available"):
        console.print("[red]qwen3:4b not found. Run: ollama pull qwen3:4b[/red]")
        sys.exit(1)

    embedder = SentenceTransformerEmbedder()
    db = Database(db_path)
    results = []

    total = len(experiments)
    console.print(f"[bold]Running {total} experiments[/bold]\n")

    batch_start = time.time()
    elapsed_times: List[float] = []

    for i, exp in enumerate(experiments):
        prompt = PROMPTS[exp.prompt_key]
        config = ExperimentConfig(
            name=exp.name,
            prompt=prompt,
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

        console.print(f"[cyan][{i+1}/{total}][/cyan] {exp.name} ({exp.category})")
        t0 = time.time()

        try:
            runner = ExperimentRunner(config, llm, embedder, db)
            result = runner.run()
            elapsed = time.time() - t0
            elapsed_times.append(elapsed)
            m = result.aggregate_metrics
            results.append({
                "name": exp.name,
                "category": exp.category,
                "id": result.experiment_id,
                "nodes": m.get("total_nodes", 0),
                "drift_regime": m.get("drift_regime", "?"),
                "final_drift": m.get("final_drift"),
                "mean_drift": m.get("mean_drift"),
                "gzip": m.get("gzip_compressibility"),
                "semantic_ratio": m.get("semantic_compressibility"),
                "sentiment": m.get("mean_sentiment"),
                "fixation": m.get("fixation_score"),
                "elapsed": elapsed,
                "status": "ok",
            })
            done = i + 1
            avg_time = sum(elapsed_times) / len(elapsed_times)
            remaining = (total - done) * avg_time
            eta_str = _format_eta(remaining)
            console.print(
                f"  -> {m.get('drift_regime', '?').upper()} | drift={m.get('final_drift', 0):.3f} | {elapsed:.0f}s"
                f"  [dim]({done}/{total} done, ETA: {eta_str})[/dim]\n"
            )
        except Exception as e:
            elapsed = time.time() - t0
            elapsed_times.append(elapsed)
            console.print(f"  [red]FAILED: {e}[/red]\n")
            results.append({"name": exp.name, "category": exp.category, "status": "failed", "error": str(e)})

    total_elapsed = time.time() - batch_start
    console.print(f"[bold]Batch completed in {_format_eta(total_elapsed)}[/bold]\n")

    # Print summary table
    console.print("\n")
    print_summary(results)
    llm.close()
    db.close()


def print_summary(results: List[dict]) -> None:
    table = Table(title="Batch Comparison Results")
    table.add_column("Experiment", style="cyan", max_width=30)
    table.add_column("Category", style="dim")
    table.add_column("Regime", style="white")
    table.add_column("Final Drift", style="red")
    table.add_column("Mean Drift", style="yellow")
    table.add_column("Gzip", style="green")
    table.add_column("Sentiment", style="blue")
    table.add_column("Fixation", style="magenta")
    table.add_column("Time", style="dim")

    for r in results:
        if r["status"] != "ok":
            table.add_row(r["name"], r["category"], "[red]FAILED[/red]", *["-"]*5, "-")
            continue

        regime = r["drift_regime"] or "?"
        regime_style = {"stable": "[green]", "oscillating": "[yellow]", "divergent": "[red]", "catastrophic": "[bold red]"}.get(regime, "")
        regime_end = "[/green]" if "green" in regime_style else "[/yellow]" if "yellow" in regime_style else "[/red]" if "red" in regime_style else "[/bold red]" if "bold" in regime_style else ""

        table.add_row(
            r["name"],
            r["category"],
            f"{regime_style}{regime.upper()}{regime_end}",
            f"{r['final_drift']:.3f}" if r["final_drift"] is not None else "-",
            f"{r['mean_drift']:.3f}" if r["mean_drift"] is not None else "-",
            f"{r['gzip']:.3f}" if r["gzip"] is not None else "-",
            f"{r['sentiment']:.3f}" if r["sentiment"] is not None else "-",
            f"{r['fixation']:.3f}" if r["fixation"] is not None else "-",
            f"{r['elapsed']:.0f}s",
        )

    console.print(table)

    # Print category comparisons
    categories = {}
    for r in results:
        if r["status"] == "ok":
            categories.setdefault(r["category"], []).append(r)

    for cat, runs in categories.items():
        console.print(f"\n[bold]{cat}[/bold]")
        drifts = [(r["name"], r["final_drift"]) for r in runs if r["final_drift"] is not None]
        if drifts:
            best = min(drifts, key=lambda x: x[1])
            worst = max(drifts, key=lambda x: x[1])
            console.print(f"  Most stable: {best[0]} (drift={best[1]:.3f})")
            console.print(f"  Least stable: {worst[0]} (drift={worst[1]:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTllm batch comparison experiments")
    parser.add_argument("--quick", action="store_true", help="Run minimal subset (3 experiments)")
    parser.add_argument("--db", default="otllm_batch.db", help="Database file path")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only one category: blind_vs_informed, tree_mode, anxiety_dose, cross_prompt, strategy")
    args = parser.parse_args()

    all_experiments = build_experiments()

    if args.quick:
        experiments = [e for e in all_experiments if e.name in QUICK_SET]
    elif args.category:
        experiments = [e for e in all_experiments if e.category == args.category]
    else:
        experiments = all_experiments

    if not experiments:
        console.print("[red]No experiments matched the filter.[/red]")
        sys.exit(1)

    console.print(f"[bold]OTllm Batch Comparison[/bold]")
    console.print(f"Database: {args.db}")
    console.print(f"Experiments: {len(experiments)}")
    console.print()

    run_batch(experiments, args.db)
