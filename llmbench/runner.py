"""Async benchmark runner — measures TTFT, TPS, and latency per request."""

import asyncio
import time
from typing import Optional

import ollama

from .metrics import SingleResult


async def _run_single(
    client: ollama.AsyncClient,
    model: str,
    prompt: str,
    thinking: bool,
    category: str,
    length: str,
) -> SingleResult:
    start_ns = time.perf_counter_ns()
    first_token_ns: Optional[int] = None
    output_tokens = 0
    eval_duration_ns = 0

    try:
        kwargs = dict(model=model, prompt=prompt, stream=True)
        # Qwen3 thinking mode: supported in Ollama >= 0.6.5
        # think=False also passed explicitly to disable reasoning in non-thinking runs
        kwargs["think"] = thinking

        async for chunk in client.generate(**kwargs):
            if first_token_ns is None and chunk.response:
                first_token_ns = time.perf_counter_ns()
            if chunk.done:
                output_tokens = chunk.eval_count or 0
                eval_duration_ns = chunk.eval_duration or 0

    except Exception as exc:
        return SingleResult(
            model=model,
            thinking=thinking,
            category=category,
            length=length,
            prompt=prompt,
            ttft_ms=None,
            tps=0.0,
            total_ms=(time.perf_counter_ns() - start_ns) / 1e6,
            output_tokens=0,
            error=str(exc),
        )

    end_ns = time.perf_counter_ns()
    ttft_ms = (first_token_ns - start_ns) / 1e6 if first_token_ns else None
    total_ms = (end_ns - start_ns) / 1e6
    # Use Ollama's reported eval_duration for accurate TPS
    tps = (output_tokens / eval_duration_ns * 1e9) if eval_duration_ns > 0 else 0.0

    return SingleResult(
        model=model,
        thinking=thinking,
        category=category,
        length=length,
        prompt=prompt,
        ttft_ms=ttft_ms,
        tps=tps,
        total_ms=total_ms,
        output_tokens=output_tokens,
    )


async def run_benchmark(
    models: list[str],
    prompts: list[dict],          # [{"category", "length", "text"}, ...]
    thinking_modes: list[bool],   # e.g. [False] or [False, True]
    concurrency: int = 1,
    shots: int = 3,
    progress_cb=None,             # optional callback(model, thinking, prompt_idx)
) -> list[SingleResult]:
    """Run full benchmark matrix and return all SingleResult records."""

    client = ollama.AsyncClient()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(model, prompt_dict, thinking, shot):
        async with semaphore:
            if progress_cb:
                progress_cb(model, thinking, prompt_dict, shot)
            return await _run_single(
                client,
                model=model,
                prompt=prompt_dict["text"],
                thinking=thinking,
                category=prompt_dict["category"],
                length=prompt_dict["length"],
            )

    tasks = [
        bounded(model, p, thinking, shot)
        for model in models
        for thinking in thinking_modes
        for p in prompts
        for shot in range(1, shots + 1)   # 1-indexed for progress display
    ]

    results = await asyncio.gather(*tasks)
    return list(results)
