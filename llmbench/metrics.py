"""Benchmark result types and aggregation statistics."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SingleResult:
    model: str
    thinking: bool
    category: str
    length: str          # short / medium / long
    prompt: str
    ttft_ms: Optional[float]
    tps: float
    total_ms: float
    output_tokens: int
    error: Optional[str] = None


@dataclass
class AggregatedStats:
    model: str
    thinking: bool
    category: str
    length: str       # "" when aggregated across multiple lengths
    n_samples: int
    error_rate: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    tps_mean: float
    tps_std: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    avg_tokens: float


def aggregate(results: list[SingleResult]) -> Optional["AggregatedStats"]:
    valid = [r for r in results if r.error is None]
    if not valid:
        return None

    ttfts = [r.ttft_ms for r in valid if r.ttft_ms is not None]
    tps_vals = [r.tps for r in valid]
    latencies = [r.total_ms for r in valid]
    token_counts = [r.output_tokens for r in valid]

    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else 0.0

    lengths = {r.length for r in valid}
    length = next(iter(lengths)) if len(lengths) == 1 else ""

    return AggregatedStats(
        model=valid[0].model,
        thinking=valid[0].thinking,
        category=valid[0].category,
        length=length,
        n_samples=len(valid),
        error_rate=(len(results) - len(valid)) / len(results),
        ttft_p50=pct(ttfts, 50),
        ttft_p95=pct(ttfts, 95),
        ttft_p99=pct(ttfts, 99),
        tps_mean=float(np.mean(tps_vals)),
        tps_std=float(np.std(tps_vals)),
        latency_p50=pct(latencies, 50),
        latency_p95=pct(latencies, 95),
        latency_p99=pct(latencies, 99),
        avg_tokens=float(np.mean(token_counts)),
    )
