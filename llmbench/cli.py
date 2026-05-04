"""CLI entry point — llm-bench compare / llm-bench list-models"""

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .metrics import aggregate
from .prompts import load_prompts, CATEGORIES
from .reporter import print_run_start, print_summary
from .runner import run_benchmark
from .storage import save
from .visualizer import save_charts

app = typer.Typer(
    name="llm-bench",
    help="Local LLM inference benchmark: Dense vs MoE, Thinking vs Standard.",
    add_completion=False,
)
console = Console()


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_thinking(value: str) -> list[bool]:
    """'standard' → [False], 'thinking' → [True], 'both' → [False, True]"""
    mapping = {"standard": [False], "thinking": [True], "both": [False, True]}
    if value not in mapping:
        raise typer.BadParameter(f"Must be one of: standard, thinking, both. Got '{value}'")
    return mapping[value]


@app.command()
def compare(
    models: str = typer.Option(
        ...,
        "--models", "-m",
        help="Comma-separated Ollama model names. e.g. qwen3:8b,qwen3:14b,qwen3:30b-a3b",
    ),
    category: str = typer.Option(
        "all",
        "--category", "-c",
        help=f"Prompt categories: all, or comma-separated from {CATEGORIES}",
    ),
    mode: str = typer.Option(
        "both",
        "--mode",
        help="Inference mode: standard | thinking | both",
    ),
    shots: int = typer.Option(
        3,
        "--shots", "-s",
        help="Number of times each prompt is run per config (min 2 for p99).",
        min=1,
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        help="Max simultaneous requests to Ollama. Keep at 1 for accurate benchmarking (default).",
        min=1,
    ),
    output: Path = typer.Option(
        Path("results"),
        "--output", "-o",
        help="Directory to save JSON, CSV, and chart files.",
    ),
    warmup: bool = typer.Option(
        True,
        "--warmup/--no-warmup",
        help="Run one silent warmup request per model before measuring.",
    ),
    no_chart: bool = typer.Option(
        False,
        "--no-chart",
        help="Skip generating the PNG dashboard.",
    ),
) -> None:
    """Benchmark and compare Ollama models across prompt categories."""

    model_list = _parse_list(models)
    thinking_modes = _parse_thinking(mode)
    categories = None if category == "all" else _parse_list(category)

    try:
        prompts = load_prompts(categories)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error loading prompts: {e}[/red]")
        raise typer.Exit(1)

    print_run_start(model_list, thinking_modes, len(prompts), shots)

    # ── Warmup ────────────────────────────────────────────────────────────
    if warmup:
        console.print("[bright_black]  Running warmup...[/bright_black]")
        asyncio.run(run_benchmark(
            models=model_list,
            prompts=[{"category": "qa", "length": "short", "text": "Say hi."}],
            thinking_modes=thinking_modes,
            concurrency=1,
            shots=1,
        ))
        console.print("[bright_black]  Warmup done.\n[/bright_black]")

    # ── Benchmark ─────────────────────────────────────────────────────────
    completed = 0
    total = len(model_list) * len(thinking_modes) * len(prompts) * shots

    def on_progress(model, thinking, prompt_dict, shot):
        nonlocal completed
        completed += 1
        mode_str = "think" if thinking else "std"
        console.print(
            f"  [bright_black][{completed:>3}/{total}][/bright_black] "
            f"[cyan]{model}[/cyan] [{mode_str}] "
            f"{prompt_dict['category']}/{prompt_dict['length']}  shot {shot}",
            highlight=False,
        )

    raw = asyncio.run(run_benchmark(
        models=model_list,
        prompts=prompts,
        thinking_modes=thinking_modes,
        concurrency=concurrency,
        shots=shots,
        progress_cb=on_progress,
    ))

    # ── Fine-grained aggregate: (model, thinking, category, length) → CSV ──
    fine_groups: dict[tuple, list] = defaultdict(list)
    for r in raw:
        fine_groups[(r.model, r.thinking, r.category, r.length)].append(r)
    fine_stats = [s for g in fine_groups.values() if (s := aggregate(g)) is not None]
    fine_stats.sort(key=lambda s: (s.category, s.length, s.model, s.thinking))

    # ── Coarse aggregate: (model, thinking, category) → terminal + chart ──
    coarse_groups: dict[tuple, list] = defaultdict(list)
    for r in raw:
        coarse_groups[(r.model, r.thinking, r.category)].append(r)
    coarse_stats = [s for g in coarse_groups.values() if (s := aggregate(g)) is not None]
    coarse_stats.sort(key=lambda s: (s.category, s.model, s.thinking))

    # ── Output ────────────────────────────────────────────────────────────
    print_summary(coarse_stats)

    json_path, csv_path = save(raw, fine_stats, output)
    console.print(f"  [green]✓[/green] Raw results  → [cyan]{json_path}[/cyan]")
    console.print(f"  [green]✓[/green] Summary CSV  → [cyan]{csv_path}[/cyan]")

    if not no_chart:
        chart_path = save_charts(raw, coarse_stats, output)
        console.print(f"  [green]✓[/green] Dashboard   → [cyan]{chart_path}[/cyan]")

    console.print()


@app.command(name="list-models")
def list_models() -> None:
    """List all models currently available in Ollama."""
    import ollama
    try:
        models = ollama.list()
        console.print("\n[bold cyan]Available Ollama models:[/bold cyan]\n")
        for m in models.models:
            size_gb = m.size / 1e9
            console.print(f"  [cyan]{m.model:<30}[/cyan]  {size_gb:.1f} GB")
        console.print()
    except Exception as e:
        console.print(f"[red]Could not connect to Ollama: {e}[/red]")
        console.print("[bright_black]  Make sure Ollama is running: ollama serve[/bright_black]")
        raise typer.Exit(1)
