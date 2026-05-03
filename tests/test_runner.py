"""Tests for llmbench.runner — _run_single() and run_benchmark() with mocked Ollama."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmbench.runner import _run_single, run_benchmark
from llmbench.metrics import SingleResult


# ─────────────────────────────────────────────
# Fake Ollama chunk helpers
# ─────────────────────────────────────────────

def _make_chunk(response="token", done=False, eval_count=0, eval_duration=0):
    chunk = MagicMock()
    chunk.response = response
    chunk.done = done
    chunk.eval_count = eval_count
    chunk.eval_duration = eval_duration
    return chunk


async def _fake_stream(chunks):
    """Async generator that yields fake chunks."""
    for c in chunks:
        yield c


# ─────────────────────────────────────────────
# _run_single
# ─────────────────────────────────────────────

class TestRunSingle:
    def test_successful_run_metrics(self):
        """TPS, output_tokens, ttft_ms, model, and thinking are all computed correctly."""
        chunks = [
            _make_chunk(response="hello", done=False),
            _make_chunk(response="", done=True, eval_count=10, eval_duration=1_000_000_000),
        ]

        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value=_fake_stream(chunks))

        result = asyncio.run(_run_single(
            client=mock_client,
            model="qwen3:8b",
            prompt="Say hi.",
            thinking=False,
            category="qa",
            length="short",
        ))

        assert isinstance(result, SingleResult)
        assert result.error is None
        assert result.model == "qwen3:8b"
        assert result.thinking is False
        assert result.output_tokens == 10
        assert result.tps == pytest.approx(10.0)        # 10 tokens / 1 s
        assert result.ttft_ms is not None
        assert result.ttft_ms > 0                       # first token recorded

    def test_thinking_mode_sets_think_true_and_field(self):
        """thinking=True passes think=True to Ollama AND sets result.thinking."""
        chunks = [
            _make_chunk(response="ok", done=False),
            _make_chunk(response="", done=True, eval_count=5, eval_duration=500_000_000),
        ]
        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value=_fake_stream(chunks))

        result = asyncio.run(_run_single(
            client=mock_client,
            model="qwen3:8b",
            prompt="Think.",
            thinking=True,
            category="reasoning",
            length="long",
        ))

        call_kwargs = mock_client.generate.call_args.kwargs
        assert call_kwargs.get("think") is True
        assert result.thinking is True

    def test_standard_mode_sets_think_false_and_field(self):
        """thinking=False passes think=False to Ollama AND sets result.thinking."""
        chunks = [
            _make_chunk(response="ok", done=False),
            _make_chunk(response="", done=True, eval_count=5, eval_duration=500_000_000),
        ]
        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value=_fake_stream(chunks))

        result = asyncio.run(_run_single(
            client=mock_client,
            model="qwen3:8b",
            prompt="Hello.",
            thinking=False,
            category="qa",
            length="short",
        ))

        call_kwargs = mock_client.generate.call_args.kwargs
        assert call_kwargs.get("think") is False
        assert result.thinking is False

    def test_exception_returns_error_result(self):
        mock_client = MagicMock()
        mock_client.generate = MagicMock(side_effect=ConnectionError("ollama not running"))

        result = asyncio.run(_run_single(
            client=mock_client,
            model="qwen3:8b",
            prompt="Hello.",
            thinking=False,
            category="qa",
            length="short",
        ))

        assert result.error is not None
        assert "ollama not running" in result.error
        assert result.tps == 0.0
        assert result.output_tokens == 0

    def test_ttft_is_none_when_no_response_token(self):
        """If all chunks have empty response, ttft_ms should be None."""
        chunks = [
            _make_chunk(response="", done=False),
            _make_chunk(response="", done=True, eval_count=0, eval_duration=0),
        ]
        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value=_fake_stream(chunks))

        result = asyncio.run(_run_single(
            client=mock_client,
            model="qwen3:8b",
            prompt="Silent.",
            thinking=False,
            category="qa",
            length="short",
        ))

        assert result.ttft_ms is None

    def test_tps_zero_when_eval_duration_is_zero(self):
        chunks = [
            _make_chunk(response="hi", done=False),
            _make_chunk(response="", done=True, eval_count=5, eval_duration=0),
        ]
        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value=_fake_stream(chunks))

        result = asyncio.run(_run_single(
            client=mock_client,
            model="qwen3:8b",
            prompt="Hello.",
            thinking=False,
            category="qa",
            length="short",
        ))

        assert result.tps == 0.0

    def test_category_and_length_stored_in_result(self):
        """category and length from the call must be preserved in the result."""
        chunks = [
            _make_chunk(response="code", done=False),
            _make_chunk(response="", done=True, eval_count=8, eval_duration=800_000_000),
        ]
        mock_client = MagicMock()
        mock_client.generate = MagicMock(return_value=_fake_stream(chunks))

        result = asyncio.run(_run_single(
            client=mock_client,
            model="qwen3:14b",
            prompt="Write bubble sort.",
            thinking=False,
            category="coding",
            length="medium",
        ))

        assert result.category == "coding"
        assert result.length == "medium"
        assert result.model == "qwen3:14b"


# ─────────────────────────────────────────────
# run_benchmark
# ─────────────────────────────────────────────

class TestRunBenchmark:
    def _patch_run_single(self, result: SingleResult):
        """Patch _run_single to always return the given result."""
        return patch("llmbench.runner._run_single", new=AsyncMock(return_value=result))

    def _dummy_result(self, model="qwen3:8b") -> SingleResult:
        return SingleResult(
            model=model,
            thinking=False,
            category="qa",
            length="short",
            prompt="Hi.",
            ttft_ms=100.0,
            tps=40.0,
            total_ms=500.0,
            output_tokens=20,
        )

    def test_total_tasks_count(self):
        """2 models × 2 thinking modes × 2 prompts × 3 shots = 24 tasks."""
        prompts = [
            {"category": "qa", "length": "short", "text": "Q1"},
            {"category": "qa", "length": "short", "text": "Q2"},
        ]
        dummy = self._dummy_result()
        with self._patch_run_single(dummy) as mock_fn:
            results = asyncio.run(run_benchmark(
                models=["model-a", "model-b"],
                prompts=prompts,
                thinking_modes=[False, True],
                shots=3,
            ))
        assert len(results) == 24
        assert mock_fn.call_count == 24

    def test_each_result_has_correct_model(self):
        """Each SingleResult must carry the model name it was called with."""
        prompts = [{"category": "qa", "length": "short", "text": "Hello."}]

        async def _fake_run_single(client, model, prompt, thinking, category, length):
            return SingleResult(
                model=model, thinking=thinking, category=category,
                length=length, prompt=prompt,
                ttft_ms=80.0, tps=35.0, total_ms=400.0, output_tokens=15,
            )

        with patch("llmbench.runner._run_single", new=_fake_run_single):
            results = asyncio.run(run_benchmark(
                models=["model-a", "model-b"],
                prompts=prompts,
                thinking_modes=[False],
                shots=1,
            ))

        model_names = {r.model for r in results}
        assert model_names == {"model-a", "model-b"}

    def test_thinking_modes_propagated(self):
        """Both thinking=True and thinking=False must appear when modes=[False, True]."""
        prompts = [{"category": "qa", "length": "short", "text": "Q."}]

        async def _fake_run_single(client, model, prompt, thinking, category, length):
            return SingleResult(
                model=model, thinking=thinking, category=category,
                length=length, prompt=prompt,
                ttft_ms=80.0, tps=35.0, total_ms=400.0, output_tokens=15,
            )

        with patch("llmbench.runner._run_single", new=_fake_run_single):
            results = asyncio.run(run_benchmark(
                models=["model-a"],
                prompts=prompts,
                thinking_modes=[False, True],
                shots=1,
            ))

        modes = {r.thinking for r in results}
        assert modes == {False, True}

    def test_progress_callback_called_correct_times(self):
        prompts = [{"category": "qa", "length": "short", "text": "Hi."}]
        calls = []

        def on_progress(model, thinking, prompt_dict, shot):
            calls.append((model, thinking, shot))

        dummy = self._dummy_result()
        with self._patch_run_single(dummy):
            asyncio.run(run_benchmark(
                models=["model-a"],
                prompts=prompts,
                thinking_modes=[False],
                shots=3,
                progress_cb=on_progress,
            ))

        assert len(calls) == 3
        # shots should be 1-indexed
        assert {shot for _, _, shot in calls} == {1, 2, 3}
