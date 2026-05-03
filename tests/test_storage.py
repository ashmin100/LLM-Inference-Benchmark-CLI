"""Tests for llmbench.storage — save() JSON/CSV output."""

import csv
import json
from pathlib import Path

import pytest
from llmbench.metrics import SingleResult, AggregatedStats
from llmbench.storage import save, _timestamp


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_raw_result(**kwargs) -> SingleResult:
    defaults = dict(
        model="model-a",
        thinking=False,
        category="qa",
        length="short",
        prompt="Hello?",
        ttft_ms=120.0,
        tps=45.0,
        total_ms=600.0,
        output_tokens=30,
        error=None,
    )
    defaults.update(kwargs)
    return SingleResult(**defaults)


def make_stats(**kwargs) -> AggregatedStats:
    defaults = dict(
        model="model-a",
        thinking=False,
        category="qa",
        n_samples=3,
        error_rate=0.0,
        ttft_p50=120.0,
        ttft_p95=150.0,
        ttft_p99=180.0,
        tps_mean=45.0,
        tps_std=2.0,
        latency_p50=600.0,
        latency_p95=700.0,
        latency_p99=800.0,
        avg_tokens=30.0,
    )
    defaults.update(kwargs)
    return AggregatedStats(**defaults)


# ─────────────────────────────────────────────
# _timestamp
# ─────────────────────────────────────────────

class TestTimestamp:
    def test_format(self):
        ts = _timestamp()
        # Should be YYYYMMDD_HHMMSS (15 chars)
        assert len(ts) == 15
        assert ts[8] == "_"
        assert ts[:8].isdigit()
        assert ts[9:].isdigit()


# ─────────────────────────────────────────────
# save()
# ─────────────────────────────────────────────

class TestSave:
    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "new_dir" / "nested"
        save([make_raw_result()], [make_stats()], out)
        assert out.is_dir()

    def test_returns_two_paths(self, tmp_path):
        json_path, csv_path = save([make_raw_result()], [make_stats()], tmp_path)
        assert json_path.exists()
        assert csv_path.exists()

    def test_json_filename_pattern(self, tmp_path):
        json_path, _ = save([make_raw_result()], [make_stats()], tmp_path)
        assert json_path.name.startswith("raw_")
        assert json_path.suffix == ".json"

    def test_csv_filename_pattern(self, tmp_path):
        _, csv_path = save([make_raw_result()], [make_stats()], tmp_path)
        assert csv_path.name.startswith("summary_")
        assert csv_path.suffix == ".csv"

    def test_json_structure(self, tmp_path):
        raw = [make_raw_result(model="qwen3:8b", tps=42.0, output_tokens=10)]
        json_path, _ = save(raw, [make_stats()], tmp_path)
        data = json.loads(json_path.read_text())
        assert "timestamp" in data
        assert "results" in data
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert r["model"] == "qwen3:8b"
        assert r["tps"] == 42.0
        assert r["output_tokens"] == 10

    def test_json_error_field(self, tmp_path):
        raw = [make_raw_result(error="timeout")]
        json_path, _ = save(raw, [], tmp_path)
        data = json.loads(json_path.read_text())
        assert data["results"][0]["error"] == "timeout"

    def test_csv_headers(self, tmp_path):
        _, csv_path = save([make_raw_result()], [make_stats()], tmp_path)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        expected = [
            "model", "thinking", "category", "n_samples", "error_rate",
            "ttft_p50", "ttft_p95", "ttft_p99",
            "tps_mean", "tps_std",
            "latency_p50", "latency_p95", "latency_p99",
            "avg_tokens",
        ]
        assert headers == expected

    def test_csv_row_values(self, tmp_path):
        stats = make_stats(model="qwen3:14b", tps_mean=55.55, n_samples=5)
        _, csv_path = save([make_raw_result()], [stats], tmp_path)
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["model"] == "qwen3:14b"
        assert float(rows[0]["tps_mean"]) == pytest.approx(55.55)
        assert int(rows[0]["n_samples"]) == 5

    def test_multiple_stats_rows(self, tmp_path):
        stats_list = [
            make_stats(model="model-a", category="qa"),
            make_stats(model="model-b", category="coding"),
        ]
        _, csv_path = save([make_raw_result()], stats_list, tmp_path)
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_empty_stats_produces_header_only_csv(self, tmp_path):
        _, csv_path = save([make_raw_result()], [], tmp_path)
        with csv_path.open() as f:
            content = f.read()
        # Header line should still be present
        assert "model" in content
        lines = [l for l in content.strip().splitlines() if l]
        assert len(lines) == 1  # header only
