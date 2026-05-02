"""Rich terminal table output for benchmark results."""

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

from .metrics import AggregatedStats

console = Console()


def _color_tps(tps: float, max_tps: float) -> Text:
    ratio = tps / max_tps if max_tps > 0 else 0
    if ratio >= 0.85:
        return Text(f"{tps:.1f}", style="bold green")
    elif ratio >= 0.5:
        return Text(f"{tps:.1f}", style="yellow")
    else:
        return Text(f"{tps:.1f}", style="red")


def _color_ttft(ttft: float, min_ttft: float) -> Text:
    ratio = ttft / min_ttft if min_ttft > 0 else 1
    if ratio <= 1.3:
        return Text(f"{ttft:.0f}", style="bold green")
    elif ratio <= 2.0:
        return Text(f"{ttft:.0f}", style="yellow")
    else:
        return Text(f"{ttft:.0f}", style="red")


def print_summary(stats_list: list[AggregatedStats]) -> None:
    """Print a color-coded summary table to the terminal."""

    if not stats_list:
        console.print("[red]No results to display.[/red]")
        return

    max_tps = max(s.tps_mean for s in stats_list)
    min_ttft = min(s.ttft_p50 for s in stats_list if s.ttft_p50 > 0)

    table = Table(
        title="[bold cyan]LLM Inference Benchmark — Summary[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
        row_styles=["", "dim"],
    )

    table.add_column("Model", style="cyan", no_wrap=True, min_width=24)
    table.add_column("Think", justify="center", min_width=6)
    table.add_column("Category", min_width=12)
    table.add_column("TPS\n(mean)", justify="right", min_width=8)
    table.add_column("TTFT p50\n(ms)", justify="right", min_width=10)
    table.add_column("TTFT p99\n(ms)", justify="right", min_width=10)
    table.add_column("Lat p50\n(ms)", justify="right", min_width=10)
    table.add_column("Lat p99\n(ms)", justify="right", min_width=10)
    table.add_column("Tokens\n(avg out)", justify="right", min_width=10)
    table.add_column("Err%", justify="right", min_width=6)

    for s in stats_list:
        think_icon = "✓" if s.thinking else "—"
        err_pct = f"{s.error_rate * 100:.0f}%"
        err_style = "red" if s.error_rate > 0 else "bright_black"

        table.add_row(
            s.model,
            think_icon,
            s.category,
            _color_tps(s.tps_mean, max_tps),
            _color_ttft(s.ttft_p50, min_ttft),
            Text(f"{s.ttft_p99:.0f}", style="bright_black"),
            Text(f"{s.latency_p50:.0f}"),
            Text(f"{s.latency_p99:.0f}", style="bright_black"),
            Text(f"{s.avg_tokens:.0f}"),
            Text(err_pct, style=err_style),
        )

    console.print()
    console.print(table)
    console.print(
        "[bright_black]  🧠 = thinking mode ON  |  "
        "TPS: green ≥85% of best, yellow ≥50%, red below  |  "
        "TTFT: green ≤1.3× best[/bright_black]\n"
    )


def print_run_start(models: list[str], thinking_modes: list[bool], n_prompts: int, shots: int) -> None:
    total = len(models) * len(thinking_modes) * n_prompts * shots
    mode_str = " + ".join(("thinking" if t else "standard") for t in thinking_modes)
    console.rule("[bold cyan]llm-bench[/bold cyan]")
    console.print(f"  Models   : [cyan]{', '.join(models)}[/cyan]")
    console.print(f"  Modes    : [cyan]{mode_str}[/cyan]")
    console.print(f"  Prompts  : [cyan]{n_prompts}[/cyan]  ×  shots [cyan]{shots}[/cyan]  =  [bold]{total}[/bold] total requests")
    console.print()
