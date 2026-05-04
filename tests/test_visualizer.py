"""Smoke tests for llmbench.visualizer — verifies save_charts() doesn't crash
and produces a PNG. Does not assert visual correctness."""

from pathlib import Path
import pytest
from llmbench.metrics import AggregatedStats
from llmbench.visualizer import save_charts


def _make_stats(model="m", thinking=False, category="qa", tps=40.0) -> AggregatedStats:
    return AggregatedStats(
        model=model, thinking=thinking, category=category, length="short",
        n_samples=3, error_rate=0.0,
        ttft_p50=100.0, ttft_p95=150.0, ttft_p99=200.0,
        tps_mean=tps, tps_std=2.0,
        latency_p50=500.0, latency_p95=700.0, latency_p99=900.0,
        avg_tokens=20.0,
    )


class TestSaveCharts:
    def test_creates_png_file(self, tmp_path):
        stats = [_make_stats("m-a"), _make_stats("m-b")]
        path = save_charts([], stats, tmp_path)
        assert path.exists()
        assert path.suffix == ".png"

    def test_empty_stats_does_not_crash(self, tmp_path):
        path = save_charts([], [], tmp_path)
        assert isinstance(path, Path)

    def test_standard_mode_only(self, tmp_path):
        """With only standard mode, the thinking heatmap panel should show 'not run'."""
        stats = [
            _make_stats("m", thinking=False, category=cat)
            for cat in ["qa", "coding", "reasoning"]
        ]
        path = save_charts([], stats, tmp_path)
        assert path.exists()

    def test_both_modes(self, tmp_path):
        """With both modes, both heatmap panels should render without error."""
        stats = [
            _make_stats("m-a", thinking=mode, category=cat, tps=40 - 10 * mode)
            for mode in [False, True]
            for cat in ["qa", "coding"]
        ]
        path = save_charts([], stats, tmp_path)
        assert path.exists()

    def test_multiple_models(self, tmp_path):
        """Three-model comparison — the most common real use case."""
        stats = [
            _make_stats(model, thinking=False, category=cat, tps=base)
            for model, base in [("qwen3:8b", 42), ("qwen3:14b", 28), ("qwen3:30b-a3b", 38)]
            for cat in ["qa", "coding", "reasoning"]
        ]
        path = save_charts([], stats, tmp_path)
        assert path.exists()

    def test_output_dir_is_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        path = save_charts([], [_make_stats()], nested)
        assert nested.is_dir()
        assert path.exists()
