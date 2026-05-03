"""Seaborn/Matplotlib visualization — dark modern style."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

from .metrics import SingleResult, AggregatedStats

# ── Palette ───────────────────────────────────────────────────────────────
BG           = "#0F1117"
GRID         = "#1E2130"
TEXT         = "#E0E0E0"
MODE_PALETTE = {"standard": "#4FC3F7", "think": "#FFB74D"}


def _setup_style() -> None:
    sns.set_theme(style="dark", rc={
        "figure.facecolor":  BG,
        "axes.facecolor":    GRID,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        "#2A2D3E",
        "grid.linewidth":    0.6,
        "legend.facecolor":  "#1A1D2E",
        "legend.edgecolor":  "#2A2D3E",
        "font.family":       "sans-serif",
        "font.size":         11,
    })


def _short_name(model: str) -> str:
    return model if len(model) <= 20 else model[:18] + "…"


def _build_agg_frame(stats: list[AggregatedStats]) -> pd.DataFrame:
    """Long-form DataFrame from aggregated stats for seaborn grouped plots."""
    rows = [
        {
            "model":      _short_name(s.model),
            "mode":       "think" if s.thinking else "standard",
            "category":   s.category,
            "tps_mean":   s.tps_mean,
            "ttft_p50":   s.ttft_p50,
            "latency_p99": s.latency_p99,
        }
        for s in stats
    ]
    return pd.DataFrame(rows)


def _build_raw_frame(raw: list[SingleResult]) -> pd.DataFrame:
    """Long-form DataFrame from per-request raw results for violin/strip plots."""
    rows = [
        {
            "model":    _short_name(r.model),
            "mode":     "think" if r.thinking else "standard",
            "category": r.category,
            "total_ms": r.total_ms,
            "tps":      r.tps,
        }
        for r in raw
        if not r.error
    ]
    return pd.DataFrame(rows)


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    """Value labels on top of every bar patch."""
    ylim_top = ax.get_ylim()[1]
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                h + ylim_top * 0.01,
                fmt.format(h),
                ha="center", va="bottom", fontsize=8, color=TEXT,
            )


def save_charts(
    raw: list[SingleResult],
    stats: list[AggregatedStats],
    output_dir: Path,
) -> Path:
    """Generate a 2×2 dashboard PNG and return its path."""
    _setup_style()

    agg_df = _build_agg_frame(stats)
    raw_df = _build_raw_frame(raw)

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        "LLM Inference Benchmark — Dense vs MoE · Thinking vs Standard",
        fontsize=15, fontweight="bold", color=TEXT, y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.38)

    ax_tps    = fig.add_subplot(gs[0, 0])
    ax_ttft   = fig.add_subplot(gs[0, 1])
    ax_violin = fig.add_subplot(gs[1, 0])
    ax_heat   = fig.add_subplot(gs[1, 1])

    # ── 1. TPS grouped bar (seaborn) ──────────────────────────────────────
    if not agg_df.empty:
        sns.barplot(
            data=agg_df, x="model", y="tps_mean", hue="mode",
            palette=MODE_PALETTE, ax=ax_tps, capsize=0.08, errwidth=1.2,
        )
        _annotate_bars(ax_tps)
    ax_tps.set_title("Throughput (tokens / sec)", fontweight="bold")
    ax_tps.set_xlabel("")
    ax_tps.set_ylabel("TPS")
    ax_tps.yaxis.grid(True, zorder=0)
    ax_tps.set_axisbelow(True)
    ax_tps.legend(title="Mode", fontsize=9)

    # ── 2. TTFT grouped bar (seaborn) ─────────────────────────────────────
    if not agg_df.empty:
        sns.barplot(
            data=agg_df, x="model", y="ttft_p50", hue="mode",
            palette=MODE_PALETTE, ax=ax_ttft,
        )
        _annotate_bars(ax_ttft, "{:.0f}")
    ax_ttft.set_title("Time to First Token — p50 (ms)", fontweight="bold")
    ax_ttft.set_xlabel("")
    ax_ttft.set_ylabel("TTFT (ms)")
    ax_ttft.yaxis.grid(True, zorder=0)
    ax_ttft.set_axisbelow(True)
    ax_ttft.legend(title="Mode", fontsize=9)

    # ── 3. Latency violin (seaborn) ───────────────────────────────────────
    # Need ≥ 4 points for a meaningful violin; fall back to strip otherwise
    if not raw_df.empty and len(raw_df) >= 4:
        sns.violinplot(
            data=raw_df, x="model", y="total_ms", hue="mode",
            palette=MODE_PALETTE, ax=ax_violin,
            split=True, inner="quart", linewidth=0.8,
        )
    elif not raw_df.empty:
        sns.stripplot(
            data=raw_df, x="model", y="total_ms", hue="mode",
            palette=MODE_PALETTE, ax=ax_violin, jitter=True, size=6,
        )
    ax_violin.set_title("Total Latency Distribution (ms)", fontweight="bold")
    ax_violin.set_xlabel("")
    ax_violin.set_ylabel("Latency (ms)")
    ax_violin.yaxis.grid(True, zorder=0)
    legend = ax_violin.get_legend()
    if legend:
        legend.set_title("Mode")

    # ── 4. Category × Model TPS heatmap (seaborn) ─────────────────────────
    if not agg_df.empty:
        pivot = agg_df.pivot_table(
            values="tps_mean",
            index=["model", "mode"],
            columns="category",
            aggfunc="mean",
        )
        if not pivot.empty:
            sns.heatmap(
                pivot, ax=ax_heat,
                annot=True, fmt=".1f", linewidths=0.4,
                cmap="YlOrRd",
                cbar_kws={"label": "TPS (mean)", "shrink": 0.8},
                annot_kws={"size": 9},
            )
    ax_heat.set_title("TPS by Category (model × mode)", fontweight="bold")
    ax_heat.set_xlabel("Category")
    ax_heat.set_ylabel("")

    for ax in [ax_tps, ax_ttft, ax_violin, ax_heat]:
        ax.tick_params(axis="x", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "benchmark_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path
