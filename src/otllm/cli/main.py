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
    depth: int = typer.Option(10, "--depth", "-d", help="Maximum tree depth"),
    branches: int = typer.Option(3, "--branches", "-b", help="Max branches per node"),
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
        mode=mode, max_depth=depth, max_branches_per_node=branches,
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
