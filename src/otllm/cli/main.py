from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="otllm", help="Research framework for studying overthinking and rumination in LLMs")
console = Console()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The scenario/question to induce overthinking about"),
    name: str = typer.Option("untitled", "--name", "-n", help="Experiment name"),
    mode: str = typer.Option("branching", "--mode", "-m", help="Tree mode: linear, branching, cyclic"),
    depth: int = typer.Option(5, "--depth", "-d", help="Maximum tree depth"),
    branches: int = typer.Option(3, "--branches", "-b", help="Max branches per node"),
    max_nodes: int = typer.Option(50, "--max-nodes", help="Stop after this many nodes"),
    strategy: str = typer.Option("recursive", "--strategy", "-s", help="Induction strategy: recursive, anxiety_amplifying, multi_persona, termination_suppression"),
    anxiety: float = typer.Option(0.5, "--anxiety", help="Anxiety intensity 0.0-1.0"),
    reanchor_interval: int = typer.Option(5, "--reanchor-interval", help="Check reanchoring every N nodes"),
    no_reanchor: bool = typer.Option(False, "--no-reanchor", help="Disable reanchoring"),
    reanchor_blind: bool = typer.Option(True, "--reanchor-blind/--reanchor-informed", help="Blind mode hides drift score from model during reanchoring (tests genuine self-awareness)"),
    temp: float = typer.Option(0.7, "--temp", "-t", help="Generation temperature"),
    model: str = typer.Option("qwen3:4b", "--model", help="Model name (Ollama: qwen3:4b, vLLM: Qwen/Qwen3-4B)"),
    backend: str = typer.Option("ollama", "--backend", help="LLM backend: ollama or vllm"),
    backend_url: Optional[str] = typer.Option(None, "--backend-url", help="Backend server URL (default: auto)"),
    thinking: bool = typer.Option(True, "--thinking/--no-thinking", help="Enable Qwen3 thinking mode"),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="Max tokens per generation (includes Qwen3 thinking tokens)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    db: str = typer.Option("otllm_data.db", "--db", help="Database file path"),
) -> None:
    """Run an overthinking experiment."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from otllm.config import ExperimentConfig
    from otllm.engine.runner import ExperimentRunner
    from otllm.models.embeddings import SentenceTransformerEmbedder
    from otllm.storage.database import Database

    config = ExperimentConfig(
        name=name, prompt=prompt, model_name=model,
        temperature=temp, max_tokens=max_tokens, enable_thinking=thinking,
        mode=mode, max_depth=depth, max_branches_per_node=branches, max_nodes=max_nodes,
        induction_strategy=strategy, anxiety_intensity=anxiety,
        reanchor_enabled=not no_reanchor, reanchor_interval=reanchor_interval, reanchor_blind=reanchor_blind,
        seed=seed, db_path=db,
    )

    if backend == "vllm":
        from otllm.models.vllm_llm import VLLMBackend
        url = backend_url or "http://localhost:8000"
        llm = VLLMBackend(base_url=url, model=model, enable_thinking=thinking)
    else:
        from otllm.models.ollama_llm import OllamaLLM
        if backend_url:
            llm = OllamaLLM(model=model, base_url=backend_url, enable_thinking=thinking)
        else:
            llm = OllamaLLM(model=model, enable_thinking=thinking)

    health = llm.check_health()
    if not health.get("connected"):
        console.print(f"[red]Cannot connect to {backend}. Is it running?[/red]")
        raise typer.Exit(1)
    if not health.get("model_available"):
        console.print(f"[red]Model '{model}' not found on {backend}.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]OTllm Experiment: {name}[/bold]")
    console.print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    console.print(f"  Mode: {mode} | Depth: {depth} | Branches: {branches}")
    console.print(f"  Strategy: {strategy} | Anxiety: {anxiety}")
    console.print()

    embedder = SentenceTransformerEmbedder()
    database = Database(db)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=None)

        def on_node(node, total):
            drift_str = f" drift={node.drift_from_anchor:.3f}" if node.drift_from_anchor is not None else ""
            sent_str = f" sent={node.sentiment:.2f}" if node.sentiment is not None else ""
            progress.update(task, description=f"Node {total} (d={node.depth}){drift_str}{sent_str}")

        runner = ExperimentRunner(config, llm, embedder, database, on_node=on_node)
        result = runner.run()

    console.print()
    _print_summary(result.experiment_id, result.aggregate_metrics)
    console.print(f"\n[dim]Experiment ID: {result.experiment_id}[/dim]")
    console.print(f"[dim]Run 'otllm analyze {result.experiment_id}' for detailed metrics[/dim]")
    console.print(f"[dim]Run 'otllm report {result.experiment_id}' to generate HTML report[/dim]")

    llm.close()
    database.close()


@app.command()
def analyze(
    experiment_id: str = typer.Argument(..., help="Experiment ID to analyze"),
    db: str = typer.Option("otllm_data.db", "--db"),
) -> None:
    """Analyze an existing experiment and display metrics."""
    from otllm.metrics.compressibility import gzip_compressibility, semantic_compressibility
    from otllm.metrics.drift import classify_drift_regime, compute_drift_curve, count_drift_reversals
    from otllm.metrics.fixation import fixation_score
    from otllm.storage.database import Database

    import numpy as np

    database = Database(db)
    config, tree, meta = database.load_experiment(experiment_id)

    console.print(f"[bold]Experiment: {meta['name']}[/bold] ({experiment_id})")
    console.print(f"  Status: {meta['status']} | Created: {meta['created_at']}")
    console.print(f"  Prompt: {config.prompt[:80]}")
    console.print()

    table = Table(title="Stability Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Interpretation", style="dim")

    drift_values = [n.drift_from_anchor for n in tree.bfs() if n.drift_from_anchor is not None]

    if drift_values:
        regime = classify_drift_regime(drift_values)
        regime_interp = {
            "stable": "Model stays on-topic despite overthinking",
            "oscillating": "Ruminating but self-correcting",
            "divergent": "Spiraling, losing the thread",
            "catastrophic": "Context collapse",
        }
        table.add_row("Drift Regime", regime.upper(), regime_interp.get(regime, ""))
        table.add_row("Final Drift", f"{drift_values[-1]:.4f}", "0=on-topic, 1=off-topic")
        table.add_row("Mean Drift", f"{np.mean(drift_values):.4f}", "")
        table.add_row("Max Drift", f"{np.max(drift_values):.4f}", "")
        table.add_row("Drift Reversals", str(count_drift_reversals(drift_values)), "Higher=more self-correction")

    texts = [n.text for n in tree.bfs()]
    if texts:
        gz = gzip_compressibility(texts)
        table.add_row("Gzip Compressibility", f"{gz:.4f}", "Lower=more repetitive")

    embeddings = [np.array(n.embedding) for n in tree.bfs() if n.embedding is not None]
    if len(embeddings) >= 2:
        sem = semantic_compressibility(embeddings)
        table.add_row("Semantic Clusters", f"{sem['n_clusters']}/{sem['n_nodes']}", "Fewer clusters=more repetitive")
        table.add_row("Semantic Ratio", f"{sem['ratio']:.4f}", "Lower=more semantic repetition")

    sentiments = [n.sentiment for n in tree.bfs() if n.sentiment is not None]
    if sentiments:
        table.add_row("Mean Sentiment", f"{np.mean(sentiments):.4f}", "Negative=anxious tone")
        table.add_row("Sentiment Std", f"{np.std(sentiments):.4f}", "Higher=more volatile")
        table.add_row("Min Sentiment", f"{np.min(sentiments):.4f}", "Lowest emotional point")

    if embeddings:
        fix = fixation_score(tree)
        table.add_row("Fixation Score", f"{fix['score']:.4f}", "Higher=more topic repetition")

    table.add_row("Total Nodes", str(tree.node_count), "")
    table.add_row("Max Depth", str(tree.max_depth_reached), "")
    table.add_row("Leaf Nodes", str(len(tree.get_leaves())), "")

    console.print(table)

    reanchor_events = database.get_reanchor_events(experiment_id)
    if reanchor_events:
        console.print()
        rt = Table(title="Reanchoring Events")
        rt.add_column("Node", style="cyan")
        rt.add_column("Decision", style="white")
        rt.add_column("Drift at Decision", style="yellow")
        rt.add_column("Drift Blind?", style="red")
        for ev in reanchor_events:
            rt.add_row(
                ev["node_id"][:8],
                ev["decision"],
                f"{ev['drift_at_decision']:.4f}" if ev["drift_at_decision"] else "-",
                "YES" if ev["was_drift_blind"] else "no",
            )
        console.print(rt)

    database.close()


@app.command()
def report(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output HTML path"),
    db: str = typer.Option("otllm_data.db", "--db"),
) -> None:
    """Generate an interactive HTML report for an experiment."""
    from otllm.report.generator import HTMLReportGenerator
    from otllm.storage.database import Database

    database = Database(db)
    if output is None:
        output = f"otllm_report_{experiment_id}.html"

    generator = HTMLReportGenerator(database, experiment_id)
    generator.generate(output)
    console.print(f"[green]Report generated: {output}[/green]")
    database.close()


@app.command("list")
def list_experiments(
    db: str = typer.Option("otllm_data.db", "--db"),
) -> None:
    """List all experiments in the database."""
    from otllm.storage.database import Database

    database = Database(db)
    experiments = database.list_experiments()

    if not experiments:
        console.print("[dim]No experiments found.[/dim]")
        database.close()
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Status", style="green")
    table.add_column("Nodes", style="yellow")
    table.add_column("Depth", style="yellow")
    table.add_column("Drift", style="red")
    table.add_column("Regime", style="magenta")
    table.add_column("Created", style="dim")

    for exp in experiments:
        table.add_row(
            exp["id"][:8],
            exp["name"] or "untitled",
            exp["status"] or "?",
            str(exp["total_nodes"] or "-"),
            str(exp["max_depth_reached"] or "-"),
            f"{exp['final_drift']:.3f}" if exp["final_drift"] else "-",
            exp["drift_regime"] or "-",
            (exp["created_at"] or "")[:19],
        )

    console.print(table)
    database.close()


@app.command()
def export(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    format: str = typer.Option("json", "--format", "-f", help="Export format: json, csv"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    db: str = typer.Option("otllm_data.db", "--db"),
) -> None:
    """Export experiment data."""
    from otllm.storage.database import Database
    from otllm.storage.export import export_summary_csv, export_to_json

    database = Database(db)
    if format == "json":
        out = output or f"otllm_export_{experiment_id}.json"
        export_to_json(experiment_id, database, out)
        console.print(f"[green]Exported to {out}[/green]")
    elif format == "csv":
        out = output or "otllm_experiments.csv"
        export_summary_csv(database, out)
        console.print(f"[green]Exported to {out}[/green]")
    else:
        console.print(f"[red]Unknown format: {format}[/red]")
    database.close()


@app.command()
def push(
    repo_id: str = typer.Argument(..., help="HuggingFace dataset repo (e.g., username/otllm-experiments)"),
    db: str = typer.Option("otllm_data.db", "--db"),
    experiment_ids: Optional[str] = typer.Option(None, "--experiments", "-e", help="Comma-separated experiment IDs (default: all)"),
    report_glob: str = typer.Option("otllm_report_*.html", "--reports", help="Glob pattern for HTML reports"),
    no_db: bool = typer.Option(False, "--no-db", help="Skip uploading the raw SQLite database"),
    no_reports: bool = typer.Option(False, "--no-reports", help="Skip uploading HTML reports"),
    private: bool = typer.Option(False, "--private", help="Create a private dataset repo"),
    message: str = typer.Option("Update OTllm experiment data", "--message", "-m", help="Commit message"),
) -> None:
    """Push experiments, reports, and database to a HuggingFace dataset."""
    from otllm.storage.database import Database
    from otllm.storage.hf_publisher import HuggingFacePublisher

    database = Database(db)
    exp_list = experiment_ids.split(",") if experiment_ids else None

    if exp_list:
        console.print(f"[bold]Pushing {len(exp_list)} experiments to {repo_id}[/bold]")
    else:
        all_exps = database.list_experiments()
        console.print(f"[bold]Pushing all {len(all_exps)} experiments to {repo_id}[/bold]")

    publisher = HuggingFacePublisher(
        db=database,
        repo_id=repo_id,
        db_path=db,
        report_glob=report_glob,
        private=private,
    )

    publisher.publish(
        experiment_ids=exp_list,
        include_db=not no_db,
        include_reports=not no_reports,
        commit_message=message,
    )

    database.close()


@app.command()
def info(
    db: str = typer.Option("otllm_data.db", "--db"),
) -> None:
    """Show system info: Ollama status, model availability, embedding model."""
    from otllm.models.ollama_llm import OllamaLLM

    llm = OllamaLLM()
    health = llm.check_health()

    table = Table(title="System Info")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")

    if health["connected"]:
        table.add_row("Ollama", "[green]Connected[/green]")
        table.add_row("Available Models", ", ".join(health.get("models", [])) or "none")
        table.add_row("qwen3:4b", "[green]Available[/green]" if health.get("model_available") else "[red]Not found — run: ollama pull qwen3:4b[/red]")
    else:
        table.add_row("Ollama", f"[red]Not connected: {health.get('error', 'unknown')}[/red]")

    table.add_row("Embedding Model", "all-MiniLM-L6-v2 (loaded on first use)")
    table.add_row("Database", db)

    console.print(table)
    llm.close()


@app.command("batch-analyze")
def batch_analyze(
    db: str = typer.Option("otllm_data.db", "--db"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category (parsed from experiment name)"),
    export_csv: Optional[str] = typer.Option(None, "--csv", help="Export results to CSV"),
    status_filter: str = typer.Option("completed", "--status", help="Filter by status: completed, all"),
) -> None:
    """Analyze all experiments in batch. Compare across categories, strategies, and prompts."""
    import csv as csv_mod
    import json
    import re

    import numpy as np

    from otllm.storage.database import Database

    database = Database(db)
    all_experiments = database.list_experiments()

    if status_filter != "all":
        all_experiments = [e for e in all_experiments if e.get("status") == status_filter]

    if not all_experiments:
        console.print("[dim]No experiments found.[/dim]")
        database.close()
        return

    # Parse category and variable from experiment names
    # Naming convention: category-variable-prompt, e.g. "reanchor-blind-job", "anxiety-0.4-health"
    enriched = []
    for exp in all_experiments:
        name = exp.get("name", "")
        config_json = None
        try:
            row = database._conn.execute(
                "SELECT config_json FROM experiments WHERE id = ?", (exp["id"],)
            ).fetchone()
            if row:
                config_json = json.loads(row["config_json"])
        except Exception:
            pass

        cat = ""
        variable = ""
        prompt_key = ""

        if config_json:
            # Derive category from config
            strat = config_json.get("induction_strategy", "recursive")
            mode = config_json.get("mode", "branching")
            blind = config_json.get("reanchor_blind", True)

            if name.startswith("reanchor-"):
                cat = "blind_vs_informed"
                variable = "blind" if blind else "informed"
            elif name.startswith("mode-"):
                cat = "tree_mode"
                variable = mode
            elif name.startswith("anxiety-"):
                cat = "anxiety_dose"
                m = re.search(r"anxiety-([\d.]+)", name)
                variable = m.group(1) if m else str(config_json.get("anxiety_intensity", ""))
            elif name.startswith("strategy-"):
                cat = "strategy"
                variable = strat
            elif name.startswith("depth-"):
                cat = "depth_scaling"
                m = re.search(r"depth-(\d+)", name)
                variable = m.group(1) if m else str(config_json.get("max_depth", ""))
            elif name.startswith("prompt-"):
                cat = "cross_prompt"
                variable = name.replace("prompt-", "")

        # Extract prompt key from name (last segment after last dash)
        parts = name.rsplit("-", 1)
        if len(parts) == 2:
            prompt_key = parts[1]

        if not cat:
            cat = "uncategorized"

        enriched.append({**exp, "_category": cat, "_variable": variable, "_prompt": prompt_key})

    if category:
        enriched = [e for e in enriched if e["_category"] == category]

    if not enriched:
        console.print(f"[red]No experiments matched category '{category}'.[/red]")
        database.close()
        return

    # ── Overview table ──
    console.print(f"\n[bold]Batch Analysis: {len(enriched)} experiments[/bold]\n")

    overview = Table(title="All Experiments")
    overview.add_column("Name", style="cyan", max_width=32)
    overview.add_column("Category", style="dim")
    overview.add_column("Variable", style="white")
    overview.add_column("Nodes", style="yellow")
    overview.add_column("Regime", style="white")
    overview.add_column("Final Drift", style="red")
    overview.add_column("Mean Drift", style="yellow")
    overview.add_column("Gzip", style="green")
    overview.add_column("Sentiment", style="blue")

    for e in sorted(enriched, key=lambda x: (x["_category"], x["_variable"], x["_prompt"])):
        regime = e.get("drift_regime") or "?"
        color = {"stable": "green", "oscillating": "yellow", "divergent": "red", "catastrophic": "bold red"}.get(regime, "white")
        overview.add_row(
            e["name"] or "?",
            e["_category"],
            e["_variable"],
            str(e.get("total_nodes") or "-"),
            f"[{color}]{regime.upper()}[/{color}]",
            f"{e['final_drift']:.3f}" if e.get("final_drift") is not None else "-",
            f"{e.get('mean_drift', 0):.3f}" if e.get("mean_drift") is not None else "-",
            f"{e.get('gzip_compressibility', 0):.3f}" if e.get("gzip_compressibility") is not None else "-",
            f"{e.get('mean_sentiment', 0):.3f}" if e.get("mean_sentiment") is not None else "-",
        )
    console.print(overview)

    # ── Per-category summaries ──
    categories: dict[str, list] = {}
    for e in enriched:
        categories.setdefault(e["_category"], []).append(e)

    for cat, runs in sorted(categories.items()):
        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]{cat}[/bold cyan] ({len(runs)} experiments)\n")

        # Group by variable within category
        by_var: dict[str, list] = {}
        for r in runs:
            by_var.setdefault(r["_variable"], []).append(r)

        var_table = Table(title=f"{cat} — By Condition")
        var_table.add_column("Condition", style="cyan")
        var_table.add_column("N", style="dim")
        var_table.add_column("Mean Drift", style="red")
        var_table.add_column("Std Drift", style="yellow")
        var_table.add_column("Min Drift", style="green")
        var_table.add_column("Max Drift", style="red")
        var_table.add_column("Mean Sentiment", style="blue")
        var_table.add_column("Regime Distribution", style="white")

        var_summaries = []
        for var, var_runs in sorted(by_var.items()):
            drifts = [r["final_drift"] for r in var_runs if r.get("final_drift") is not None]
            sentiments = [r["mean_sentiment"] for r in var_runs if r.get("mean_sentiment") is not None]
            regimes = [r.get("drift_regime", "?") for r in var_runs if r.get("drift_regime")]

            regime_counts = {}
            for reg in regimes:
                regime_counts[reg] = regime_counts.get(reg, 0) + 1
            regime_str = " ".join(f"{k[0].upper()}:{v}" for k, v in sorted(regime_counts.items()))

            if drifts:
                mean_d = float(np.mean(drifts))
                std_d = float(np.std(drifts))
                var_table.add_row(
                    var or "default",
                    str(len(var_runs)),
                    f"{mean_d:.3f}",
                    f"{std_d:.3f}",
                    f"{min(drifts):.3f}",
                    f"{max(drifts):.3f}",
                    f"{np.mean(sentiments):.3f}" if sentiments else "-",
                    regime_str,
                )
                var_summaries.append({"var": var, "mean_drift": mean_d, "std_drift": std_d})
            else:
                var_table.add_row(var or "default", str(len(var_runs)), "-", "-", "-", "-", "-", regime_str)

        console.print(var_table)

        # Category-specific insights
        if cat == "blind_vs_informed" and len(by_var) == 2:
            blind_drifts = [r["final_drift"] for r in by_var.get("blind", []) if r.get("final_drift") is not None]
            informed_drifts = [r["final_drift"] for r in by_var.get("informed", []) if r.get("final_drift") is not None]
            if blind_drifts and informed_drifts:
                diff = float(np.mean(blind_drifts)) - float(np.mean(informed_drifts))
                console.print(f"\n  [bold]Drift blindness gap:[/bold] blind mean {np.mean(blind_drifts):.3f} vs informed mean {np.mean(informed_drifts):.3f} (delta={diff:+.3f})")
                if diff > 0.05:
                    console.print("  [yellow]-> Blind mode drifts MORE. Model benefits from seeing its drift score.[/yellow]")
                elif diff < -0.05:
                    console.print("  [green]-> Blind mode drifts LESS. Model self-corrects better without numeric feedback.[/green]")
                else:
                    console.print("  [dim]-> No meaningful difference. Drift score doesn't affect self-correction.[/dim]")

            # Count drift blindness events
            for label, var_key in [("Blind", "blind"), ("Informed", "informed")]:
                blind_events = 0
                total_events = 0
                for r in by_var.get(var_key, []):
                    events = database.get_reanchor_events(r["id"])
                    for ev in events:
                        total_events += 1
                        if ev.get("was_drift_blind"):
                            blind_events += 1
                if total_events > 0:
                    console.print(f"  {label}: {blind_events}/{total_events} reanchor checks were drift-blind ({blind_events/total_events*100:.0f}%)")

        elif cat == "anxiety_dose" and var_summaries:
            console.print(f"\n  [bold]Dose-response curve:[/bold]")
            sorted_vars = sorted(var_summaries, key=lambda x: float(x["var"]) if x["var"] else 0)
            for v in sorted_vars:
                bar_len = int(v["mean_drift"] * 40)
                bar = "#" * bar_len
                console.print(f"    anxiety={v['var']:>3s}  drift={v['mean_drift']:.3f}  [red]{bar}[/red]")

        elif cat == "strategy" and var_summaries:
            sorted_vars = sorted(var_summaries, key=lambda x: x["mean_drift"])
            console.print(f"\n  [bold]Strategy ranking (most to least stable):[/bold]")
            for i, v in enumerate(sorted_vars, 1):
                console.print(f"    {i}. {v['var']} (mean drift={v['mean_drift']:.3f} +/- {v['std_drift']:.3f})")

        elif cat == "tree_mode" and var_summaries:
            sorted_vars = sorted(var_summaries, key=lambda x: x["mean_drift"])
            console.print(f"\n  [bold]Mode ranking (most to least stable):[/bold]")
            for i, v in enumerate(sorted_vars, 1):
                console.print(f"    {i}. {v['var']} (mean drift={v['mean_drift']:.3f} +/- {v['std_drift']:.3f})")

        elif cat == "depth_scaling" and var_summaries:
            console.print(f"\n  [bold]Depth vs drift:[/bold]")
            sorted_vars = sorted(var_summaries, key=lambda x: int(x["var"]) if x["var"].isdigit() else 0)
            for v in sorted_vars:
                bar_len = int(v["mean_drift"] * 40)
                bar = "#" * bar_len
                console.print(f"    depth={v['var']:>2s}  drift={v['mean_drift']:.3f}  [red]{bar}[/red]")

    # ── Global summary ──
    console.print(f"\n{'='*60}")
    console.print("[bold]Global Summary[/bold]\n")

    all_drifts = [e["final_drift"] for e in enriched if e.get("final_drift") is not None]
    all_regimes = [e.get("drift_regime", "?") for e in enriched if e.get("drift_regime")]
    all_sentiments = [e["mean_sentiment"] for e in enriched if e.get("mean_sentiment") is not None]

    if all_drifts:
        console.print(f"  Drift:     mean={np.mean(all_drifts):.3f}  std={np.std(all_drifts):.3f}  min={min(all_drifts):.3f}  max={max(all_drifts):.3f}")
    if all_sentiments:
        console.print(f"  Sentiment: mean={np.mean(all_sentiments):.3f}  std={np.std(all_sentiments):.3f}  min={min(all_sentiments):.3f}  max={max(all_sentiments):.3f}")

    if all_regimes:
        regime_counts: dict[str, int] = {}
        for r in all_regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        console.print(f"\n  Regime distribution:")
        for regime in ["stable", "oscillating", "divergent", "catastrophic"]:
            count = regime_counts.get(regime, 0)
            pct = count / len(all_regimes) * 100
            bar = "#" * int(pct / 2)
            color = {"stable": "green", "oscillating": "yellow", "divergent": "red", "catastrophic": "bold red"}.get(regime, "white")
            console.print(f"    [{color}]{regime:>13s}[/{color}]: {count:>3d}/{len(all_regimes)}  ({pct:4.1f}%)  {bar}")

    # Most/least stable experiments
    if all_drifts and len(enriched) > 1:
        with_drift = [e for e in enriched if e.get("final_drift") is not None]
        sorted_by_drift = sorted(with_drift, key=lambda x: x["final_drift"])

        console.print(f"\n  [green]Most stable:[/green]")
        for e in sorted_by_drift[:3]:
            console.print(f"    {e['name']} (drift={e['final_drift']:.3f}, regime={e.get('drift_regime', '?')})")

        console.print(f"\n  [red]Least stable:[/red]")
        for e in sorted_by_drift[-3:]:
            console.print(f"    {e['name']} (drift={e['final_drift']:.3f}, regime={e.get('drift_regime', '?')})")

    # ── CSV export ──
    if export_csv:
        fields = ["name", "_category", "_variable", "_prompt", "status",
                  "total_nodes", "max_depth_reached", "final_drift", "mean_drift",
                  "gzip_compressibility", "semantic_compressibility",
                  "drift_regime", "mean_sentiment"]
        with open(export_csv, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(enriched, key=lambda x: (x["_category"], x["_variable"])))
        console.print(f"\n[green]Exported to {export_csv}[/green]")

    database.close()


def _print_summary(experiment_id: str, metrics: dict) -> None:
    table = Table(title="Experiment Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    for key, val in metrics.items():
        if isinstance(val, float):
            table.add_row(key, f"{val:.4f}")
        else:
            table.add_row(key, str(val))

    console.print(table)


if __name__ == "__main__":
    app()
