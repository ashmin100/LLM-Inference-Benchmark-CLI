"""Pytest plugin: live progress + end-of-run chart saved to results/."""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_config = None
_total = 0
_current = 0
_t0 = 0.0

# accumulated per-test records
_records: list[dict] = []

# ── colours ───────────────────────────────────────────────────────────────────
BG      = "#0F1117"
GRID    = "#1E2130"
TEXT    = "#E0E0E0"
GREEN   = "#4CAF50"
RED     = "#EF5350"
YELLOW  = "#FFB74D"


# ── hooks ─────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    global _config
    _config = config


def pytest_collection_finish(session):
    global _total
    _total = len(session.items)


def pytest_runtest_logstart(nodeid, location):
    global _current, _t0
    _current += 1
    _t0 = time.monotonic()


def pytest_runtest_logreport(report):
    tw = _config.get_terminal_writer()

    if report.when == "call":
        elapsed_ms = (time.monotonic() - _t0) * 1000
        progress = f"[{_current}/{_total}]"

        if report.passed:
            tw.line(f"  {progress}  {elapsed_ms:.0f}ms", green=True)
            _records.append({"nodeid": report.nodeid, "outcome": "passed", "ms": elapsed_ms})
        elif report.failed:
            tw.line(f"  {progress}  {elapsed_ms:.0f}ms  -- failure details:", red=True)
            if report.longrepr:
                for line in str(report.longrepr).strip().splitlines():
                    tw.line(f"      {line}", red=True)
            tw.line()
            _records.append({"nodeid": report.nodeid, "outcome": "failed", "ms": elapsed_ms})
        elif report.skipped:
            _records.append({"nodeid": report.nodeid, "outcome": "skipped", "ms": elapsed_ms})

    elif report.when == "setup" and report.failed:
        tw.line(f"  [{_current}/{_total}]  setup error:", red=True)
        if report.longrepr:
            for line in str(report.longrepr).strip().splitlines():
                tw.line(f"      {line}", red=True)
        tw.line()
        _records.append({"nodeid": report.nodeid, "outcome": "error", "ms": 0.0})


def pytest_sessionfinish(session, exitstatus):
    if not _records:
        return
    out = Path("results") / "test_report.png"
    _save_chart(out)
    tw = _config.get_terminal_writer()
    tw.line(f"\n  test report → {out}", bold=True)


# ── chart ─────────────────────────────────────────────────────────────────────

def _save_chart(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # sort by duration descending for the bar chart
    records = sorted(_records, key=lambda r: r["ms"], reverse=True)
    labels  = [_short(r["nodeid"]) for r in records]
    durations = [r["ms"] for r in records]
    colors  = [_color(r["outcome"]) for r in records]

    n_pass  = sum(1 for r in _records if r["outcome"] == "passed")
    n_fail  = sum(1 for r in _records if r["outcome"] == "failed")
    n_skip  = sum(1 for r in _records if r["outcome"] in ("skipped", "error"))
    total   = len(_records)

    fig = plt.figure(figsize=(14, max(6, total * 0.28 + 3)), facecolor=BG)
    fig.suptitle("pytest — test run report", fontsize=13, fontweight="bold",
                 color=TEXT, y=0.99)

    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.35,
                          left=0.02, right=0.97, top=0.93, bottom=0.06)

    ax_bar = fig.add_subplot(gs[0])
    ax_sum = fig.add_subplot(gs[1])

    # ── left: horizontal bar chart ────────────────────────────────────────────
    y_pos = range(len(records))
    ax_bar.barh(list(y_pos), durations, color=colors, height=0.7, zorder=2)
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(labels, fontsize=7.5, color=TEXT)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("duration (ms)", color=TEXT, fontsize=9)
    ax_bar.set_facecolor(GRID)
    ax_bar.tick_params(colors=TEXT)
    ax_bar.xaxis.grid(True, color="#2A2D3E", linewidth=0.5, zorder=0)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # value labels on bars
    for i, (dur, rec) in enumerate(zip(durations, records)):
        ax_bar.text(dur + max(durations) * 0.005, i, f"{dur:.0f}ms",
                    va="center", fontsize=7, color=TEXT)

    # ── right: summary ────────────────────────────────────────────────────────
    ax_sum.set_facecolor(BG)
    ax_sum.axis("off")

    donut_vals  = [v for v in [n_pass, n_fail, n_skip] if v]
    donut_cols  = [c for v, c in [(n_pass, GREEN), (n_fail, RED), (n_skip, YELLOW)] if v]
    if donut_vals:
        wedges, _ = ax_sum.pie(
            donut_vals, colors=donut_cols,
            startangle=90, counterclock=False,
            wedgeprops={"width": 0.55, "edgecolor": BG, "linewidth": 1.5},
        )
        ax_sum.text(0, 0, f"{total}", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=TEXT)

    legend_patches = [
        mpatches.Patch(color=GREEN,  label=f"passed  {n_pass}"),
        mpatches.Patch(color=RED,    label=f"failed  {n_fail}"),
        mpatches.Patch(color=YELLOW, label=f"skipped {n_skip}"),
    ]
    ax_sum.legend(handles=legend_patches, loc="lower center",
                  fontsize=9, frameon=False,
                  labelcolor=TEXT, bbox_to_anchor=(0.5, -0.08))

    total_s = sum(r["ms"] for r in _records) / 1000
    ax_sum.set_title(f"total  {total_s:.2f}s", color=TEXT, fontsize=9, pad=12)

    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _short(nodeid: str) -> str:
    """Strip the file path prefix, keep class::method."""
    parts = nodeid.split("::")
    return "::".join(parts[1:]) if len(parts) > 1 else nodeid


def _color(outcome: str) -> str:
    return {
        "passed":  GREEN,
        "failed":  RED,
        "skipped": YELLOW,
        "error":   RED,
    }.get(outcome, TEXT)
