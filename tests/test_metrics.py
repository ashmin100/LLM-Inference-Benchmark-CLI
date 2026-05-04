"""Tests for llmbench.metrics — SingleResult, AggregatedStats, aggregate()."""

import pytest
from llmbench.metrics import SingleResult, AggregatedStats, aggregate


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_result(
    model="model-a",
    thinking=False,
    category="qa",
    length="short",
    ttft_ms=100.0,
    tps=50.0,
    total_ms=500.0,
    output_tokens=25,
    error=None,
) -> SingleResult:
    return SingleResult(
        model=model,
        thinking=thinking,
        category=category,
        length=length,
        prompt="What is 2+2?",
        ttft_ms=ttft_ms,
        tps=tps,
        total_ms=total_ms,
        output_tokens=output_tokens,
        error=error,
    )


# ─────────────────────────────────────────────
# SingleResult
# ─────────────────────────────────────────────

class TestSingleResult:
    def test_default_error_is_none(self):
        r = make_result()
        assert r.error is None

    def test_error_result(self):
        r = make_result(error="connection refused")
        assert r.error == "connection refused"

    def test_fields_stored_correctly(self):
        r = make_result(model="qwen3:8b", thinking=True, tps=42.5)
        assert r.model == "qwen3:8b"
        assert r.thinking is True
        assert r.tps == 42.5


# ─────────────────────────────────────────────
# aggregate()
# ─────────────────────────────────────────────

class TestAggregate:
    def test_returns_none_for_all_errors(self):
        results = [make_result(error="fail") for _ in range(3)]
        assert aggregate(results) is None

    def test_basic_aggregation(self):
        results = [make_result(ttft_ms=100.0, tps=50.0, total_ms=500.0, output_tokens=25) for _ in range(3)]
        stats = aggregate(results)
        assert stats is not None
        assert stats.n_samples == 3
        assert stats.error_rate == 0.0
        assert stats.tps_mean == pytest.approx(50.0)

    def test_error_rate_calculation(self):
        results = [
            make_result(error="fail"),
            make_result(error="fail"),
            make_result(ttft_ms=100.0, tps=50.0, total_ms=500.0, output_tokens=25),
        ]
        stats = aggregate(results)
        assert stats is not None
        assert stats.n_samples == 1
        assert stats.error_rate == pytest.approx(2 / 3)

    def test_percentile_ordering(self):
        """p95 >= p50 >= p50 (monotonically non-decreasing)."""
        results = [
            make_result(ttft_ms=float(v), tps=50.0, total_ms=float(v * 5), output_tokens=25)
            for v in [10, 50, 100, 200, 500]
        ]
        stats = aggregate(results)
        assert stats.ttft_p50 <= stats.ttft_p95 <= stats.ttft_p99
        assert stats.latency_p50 <= stats.latency_p95 <= stats.latency_p99

    def test_ttft_none_skipped(self):
        """Results with ttft_ms=None should not crash aggregate."""
        results = [make_result(ttft_ms=None, tps=30.0, total_ms=300.0, output_tokens=10) for _ in range(3)]
        stats = aggregate(results)
        assert stats is not None
        assert stats.ttft_p50 == 0.0  # pct([]) → 0.0

    def test_model_category_thinking_preserved(self):
        results = [make_result(model="qwen3:14b", category="reasoning", thinking=True) for _ in range(2)]
        stats = aggregate(results)
        assert stats.model == "qwen3:14b"
        assert stats.category == "reasoning"
        assert stats.thinking is True

    def test_avg_tokens(self):
        results = [make_result(output_tokens=t) for t in [10, 20, 30]]
        stats = aggregate(results)
        assert stats.avg_tokens == pytest.approx(20.0)

    def test_tps_std_zero_when_uniform(self):
        results = [make_result(tps=40.0) for _ in range(5)]
        stats = aggregate(results)
        assert stats.tps_std == pytest.approx(0.0)

    def test_returns_none_for_empty_list(self):
        """aggregate([]) must not crash and must return None."""
        assert aggregate([]) is None

    def test_single_valid_result(self):
        """A single result should work with n_samples=1 and std=0."""
        stats = aggregate([make_result(tps=55.0, output_tokens=30)])
        assert stats is not None
        assert stats.n_samples == 1
        assert stats.tps_mean == pytest.approx(55.0)
        assert stats.avg_tokens == pytest.approx(30.0)

    def test_uniform_length_is_preserved(self):
        """When all results share the same length, AggregatedStats.length must match."""
        results = [make_result(length="long") for _ in range(3)]
        stats = aggregate(results)
        assert stats.length == "long"

    def test_mixed_lengths_produce_empty_string(self):
        """When results span multiple lengths (coarse aggregation), length must be ''."""
        results = [
            make_result(length="short"),
            make_result(length="medium"),
            make_result(length="long"),
        ]
        stats = aggregate(results)
        assert stats.length == ""
