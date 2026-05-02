"""Seaborn/Matplotlib visualization — dark modern style."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .metrics import SingleResult, AggregatedStats

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8", "#80DEEA"]
BG = "#0F1117"
GRID = "#1E2130"
TEXT = "#E0E0E0"


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": GRID,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": "#2A2D3E",
        "grid.linewidth": 0.6,
        "legend.facecolor": "#1A1D2E",
        "legend.edgecolor": "#2A2D3E",
        "font.family": "sans-serif",
        "font.size": 11,
    })


def _label(model: str, thinking: bool) -> str:
    return model + ("\n[think]" if thinking else "")


def save_charts(
    raw: list[SingleResult],
    stats: list[AggregatedStats],
    output_dir: Path,
) -> Path:
    """Generate a 2×2 dashboard PNG and return its path."""
    _setup_style()

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        "LLM Inference Benchmark — Dense vs MoE · Thinking vs Standard",
        fontsize=15, fontweight="bold", color=TEXT, y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_tps   = fig.add_subplot(gs[0, 0])
    ax_ttft  = fig.add_subplot(gs[0, 1])
    ax_violin = fig.add_subplot(gs[1, 0])
    ax_p99   = fig.add_subplot(gs[1, 1])

    labels = [_label(s.model, s.thinking) for s in stats]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(stats))]

    # ── 1. TPS mean bar ───────────────────────────────────────────────────
    tps_vals = [s.tps_mean for s in stats]
    tps_err  = [s.tps_std for s in stats]
    bars = ax_tps.bar(labels, tps_vals, yerr=tps_err, color=colors,
                      error_kw=dict(ecolor="#FFFFFF44", capsize=4), zorder=3)
    ax_tps.set_title("Throughput (tokens / sec)", fontweight="bold")
    ax_tps.set_ylabel("TPS")
    ax_tps.yaxis.grid(True, zorder=0)
    ax_tps.set_axisbelow(True)
    for bar, val in zip(bars, tps_vals):
        ax_tps.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tps_vals) * 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    # ── 2. TTFT p50 bar ───────────────────────────────────────────────────
    ttft_vals = [s.ttft_p50 for s in stats]
    bars2 = ax_ttft.bar(labels, ttft_vals, color=colors, zorder=3)
    ax_ttft.set_title("Time to First Token — p50 (ms)", fontweight="bold")
    ax_ttft.set_ylabel("TTFT (ms)")
    ax_ttft.yaxis.grid(True, zorder=0)
    ax_ttft.set_axisbelow(True)
    for bar, val in zip(bars2, ttft_vals):
        ax_ttft.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ttft_vals) * 0.01,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    # ── 3. Latency violin ─────────────────────────────────────────────────
    # Build per-config latency arrays
    config_latencies: dict[str, list[float]] = {}
    for r in raw:
        if r.error:
            continue
        key = _label(r.model, r.thinking)
        config_latencies.setdefault(key, []).append(r.total_ms)

    vdata = [config_latencies.get(l, []) for l in labels]
    # violinplot requires >= 2 points; fall back to scatter for sparse configs
    violin_pos, violin_data, violin_colors = [], [], []
    for i, (d, c) in enumerate(zip(vdata, colors)):
        if len(d) >= 2:
            violin_pos.append(i)
            violin_data.append(d)
            violin_colors.append(c)
        elif len(d) == 1:
            ax_violin.scatter([i], d, color=c, zorder=3, s=40, marker="D")

    if violin_data:
        parts = ax_violin.violinplot(violin_data, positions=violin_pos,
                                     showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], violin_colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        parts["cmedians"].set_colors("#FFFFFF")
        parts["cmedians"].set_linewidth(1.5)
    ax_violin.set_xticks(range(len(labels)))
    ax_violin.set_xticklabels(labels, fontsize=9)
    ax_violin.set_title("Total Latency Distribution (ms)", fontweight="bold")
    ax_violin.set_ylabel("Latency (ms)")
    ax_violin.yaxis.grid(True, zorder=0)

    # ── 4. p99 latency bar (tail latency) ────────────────────────────────
    p99_vals = [s.latency_p99 for s in stats]
    bars4 = ax_p99.bar(labels, p99_vals, color=colors, zorder=3, alpha=0.85)
    ax_p99.set_title("Tail Latency — p99 (ms)", fontweight="bold")
    ax_p99.set_ylabel("p99 Latency (ms)")
    ax_p99.yaxis.grid(True, zorder=0)
    ax_p99.set_axisbelow(True)
    for bar, val in zip(bars4, p99_vals):
        ax_p99.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(p99_vals) * 0.01,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    for ax in [ax_tps, ax_ttft, ax_violin, ax_p99]:
        ax.tick_params(axis="x", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "benchmark_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path
