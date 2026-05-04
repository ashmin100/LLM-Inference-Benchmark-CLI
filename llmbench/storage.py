"""Save benchmark results to JSON and CSV."""

import csv
import json
from datetime import datetime
from pathlib import Path

from .metrics import SingleResult, AggregatedStats


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save(
    raw: list[SingleResult],
    stats: list[AggregatedStats],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save raw results as JSON and aggregated stats as CSV. Returns (json_path, csv_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()

    # ── JSON (raw) ────────────────────────────────────────────────────────
    json_path = output_dir / f"raw_{ts}.json"
    raw_data = [
        {
            "model": r.model,
            "thinking": r.thinking,
            "category": r.category,
            "length": r.length,
            "ttft_ms": r.ttft_ms,
            "tps": r.tps,
            "total_ms": r.total_ms,
            "output_tokens": r.output_tokens,
            "error": r.error,
        }
        for r in raw
    ]
    json_path.write_text(json.dumps({"timestamp": ts, "results": raw_data}, indent=2))

    # ── CSV (aggregated) ──────────────────────────────────────────────────
    csv_path = output_dir / f"summary_{ts}.csv"
    fields = [
        "model", "thinking", "category", "length", "n_samples", "error_rate",
        "ttft_p50", "ttft_p95", "ttft_p99",
        "tps_mean", "tps_std",
        "latency_p50", "latency_p95", "latency_p99",
        "avg_tokens",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in stats:
            writer.writerow({
                "model": s.model,
                "thinking": s.thinking,
                "category": s.category,
                "length": s.length,
                "n_samples": s.n_samples,
                "error_rate": round(s.error_rate, 4),
                "ttft_p50": round(s.ttft_p50, 1),
                "ttft_p95": round(s.ttft_p95, 1),
                "ttft_p99": round(s.ttft_p99, 1),
                "tps_mean": round(s.tps_mean, 2),
                "tps_std": round(s.tps_std, 2),
                "latency_p50": round(s.latency_p50, 1),
                "latency_p95": round(s.latency_p95, 1),
                "latency_p99": round(s.latency_p99, 1),
                "avg_tokens": round(s.avg_tokens, 1),
            })

    return json_path, csv_path
