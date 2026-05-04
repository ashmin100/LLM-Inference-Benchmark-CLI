"""Seaborn/Matplotlib visualization — dark modern style, model-comparison focused."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

from .metrics import SingleResult, AggregatedStats

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


def _short(model: str) -> str:
    return model if len(model) <= 20 else model[:18] + "…"


def _build_agg_frame(stats: list[AggregatedStats]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model":       _short(s.model),
            "mode":        "think" if s.thinking else "standard",
            "category":    s.category,
            "tps_mean":    s.tps_mean,
            "ttft_p50":    s.ttft_p50,
            "latency_p50": s.latency_p50,
            "latency_p99": s.latency_p99,
        }
        for s in stats
    ])


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    ylim_top = ax.get_ylim()[1]
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                h + ylim_top * 0.012,
                fmt.format(h),
                ha="center", va="bottom", fontsize=8, color=TEXT,
            )


def _style_ax(ax: plt.Axes, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color="#2A2D3E", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _heatmap_panel(
    agg_df: pd.DataFrame,
    mode: str,
    ax: plt.Axes,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    df = agg_df[agg_df["mode"] == mode]
    if df.empty:
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.text(0.5, 0.5, f"{mode} mode\nnot run",
                ha="center", va="center", color=TEXT,
                fontsize=11, transform=ax.transAxes)
        ax.set_title(title, fontweight="bold")
        return

    pivot = df.pivot_table(values="tps_mean", index="model", columns="category", aggfunc="mean")
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".1f", linewidths=0.5,
        cmap="YlOrRd", vmin=vmin, vmax=vmax,
        cbar_kws={"label": "TPS (mean)", "shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Category", fontsize=9)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9, colors=TEXT)
    ax.tick_params(axis="y", labelsize=9, colors=TEXT, rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_charts(
    raw: list[SingleResult],
    stats: list[AggregatedStats],
    output_dir: Path,
) -> Path:
    """5-panel dashboard:
      [0, :]  Overall TPS — headline model comparison (full width)
      [1, 0]  TTFT p50 per model
      [1, 1]  Total latency p50 per model
      [2, 0]  Per-category TPS heatmap — standard mode
      [2, 1]  Per-category TPS heatmap — thinking mode
    """
    _setup_style()

    agg_df = _build_agg_frame(stats)
    if agg_df.empty:
        return output_dir / "benchmark_dashboard.png"

    # Overall: mean across categories per (model, mode) — headline numbers
    overall = (
        agg_df.groupby(["model", "mode"], sort=False)[["tps_mean", "ttft_p50", "latency_p50"]]
        .mean()
        .reset_index()
    )

    has_thinking = "think" in agg_df["mode"].unique()

    # Shared colour scale for both heatmaps so they're directly comparable
    tps_min = agg_df["tps_mean"].min()
    tps_max = agg_df["tps_mean"].max()

    fig = plt.figure(figsize=(18, 16), facecolor=BG)
    fig.suptitle(
        "LLM Inference Benchmark — Model Comparison",
        fontsize=15, fontweight="bold", color=TEXT, y=0.995,
    )

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.6, 1.0, 1.2],
        hspace=0.55, wspace=0.32,
        left=0.06, right=0.97, top=0.96, bottom=0.05,
    )

    ax_tps   = fig.add_subplot(gs[0, :])
    ax_ttft  = fig.add_subplot(gs[1, 0])
    ax_lat   = fig.add_subplot(gs[1, 1])
    ax_h_std = fig.add_subplot(gs[2, 0])
    ax_h_thk = fig.add_subplot(gs[2, 1])

    # ── Panel 1: Overall TPS ─────────────────────────────────────────────
    sns.barplot(
        data=overall, x="model", y="tps_mean", hue="mode",
        palette=MODE_PALETTE, ax=ax_tps, width=0.55,
    )
    _annotate_bars(ax_tps, "{:.1f}")
    ax_tps.set_title(
        "Overall Throughput — averaged across all categories  (tokens / sec)",
        fontweight="bold", fontsize=12,
    )
    ax_tps.tick_params(axis="x", labelsize=11)
    ax_tps.legend(title="Mode", fontsize=10, title_fontsize=10)
    _style_ax(ax_tps, ylabel="TPS (mean)")

    # ── Panel 2: TTFT p50 ────────────────────────────────────────────────
    sns.barplot(
        data=overall, x="model", y="ttft_p50", hue="mode",
        palette=MODE_PALETTE, ax=ax_ttft, width=0.55,
    )
    _annotate_bars(ax_ttft, "{:.0f}")
    ax_ttft.set_title("Time to First Token — p50 (ms)", fontweight="bold")
    ax_ttft.legend(title="Mode", fontsize=9)
    _style_ax(ax_ttft, ylabel="TTFT ms")

    # ── Panel 3: Latency p50 ─────────────────────────────────────────────
    sns.barplot(
        data=overall, x="model", y="latency_p50", hue="mode",
        palette=MODE_PALETTE, ax=ax_lat, width=0.55,
    )
    _annotate_bars(ax_lat, "{:.0f}")
    ax_lat.set_title("Total Latency — p50 (ms)", fontweight="bold")
    ax_lat.legend(title="Mode", fontsize=9)
    _style_ax(ax_lat, ylabel="Latency ms")

    # ── Panels 4 & 5: Per-category TPS heatmaps ──────────────────────────
    # Shared vmin/vmax makes standard and thinking heatmaps directly comparable.
    _heatmap_panel(agg_df, "standard", ax_h_std,
                   "TPS by Category — Standard Mode", tps_min, tps_max)
    _heatmap_panel(agg_df, "think",    ax_h_thk,
                   "TPS by Category — Thinking Mode", tps_min, tps_max)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "benchmark_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path
