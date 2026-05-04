"""Async benchmark runner — measures TTFT, TPS, and latency per request."""

import asyncio
import time
from typing import Callable, Optional

import ollama

from .metrics import SingleResult
from .prompts import PromptDict


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
        async for chunk in await client.generate(
            model=model, prompt=prompt, stream=True, think=thinking
        ):
            if first_token_ns is None and chunk.response:
                first_token_ns = time.perf_counter_ns()
            if chunk.done:
                output_tokens = chunk.eval_count or 0
                eval_duration_ns = chunk.eval_duration or 0

    except Exception as exc:
        return SingleResult(
            model=model, thinking=thinking, category=category, length=length,
            prompt=prompt, ttft_ms=None, tps=0.0,
            total_ms=(time.perf_counter_ns() - start_ns) / 1e6,
            output_tokens=0, error=str(exc),
        )

    end_ns = time.perf_counter_ns()
    ttft_ms = (first_token_ns - start_ns) / 1e6 if first_token_ns else None
    total_ms = (end_ns - start_ns) / 1e6
    tps = (output_tokens / eval_duration_ns * 1e9) if eval_duration_ns > 0 else 0.0

    return SingleResult(
        model=model, thinking=thinking, category=category, length=length,
        prompt=prompt, ttft_ms=ttft_ms, tps=tps, total_ms=total_ms,
        output_tokens=output_tokens,
    )


async def run_benchmark(
    models: list[str],
    prompts: list[PromptDict],
    thinking_modes: list[bool],
    concurrency: int = 1,
    shots: int = 3,
    timeout: float = 300.0,
    progress_cb: Optional[Callable[[str, bool, PromptDict, int], None]] = None,
) -> list[SingleResult]:
    """Run benchmark with round-robin model ordering for fair thermal comparison."""

    client = ollama.AsyncClient()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(model, prompt_dict, thinking, shot):
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    _run_single(
                        client, model=model, prompt=prompt_dict["text"],
                        thinking=thinking, category=prompt_dict["category"],
                        length=prompt_dict["length"],
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                result = SingleResult(
                    model=model, thinking=thinking,
                    category=prompt_dict["category"], length=prompt_dict["length"],
                    prompt=prompt_dict["text"], ttft_ms=None, tps=0.0,
                    total_ms=timeout * 1000, output_tokens=0,
                    error=f"timeout after {timeout:.0f}s",
                )
            if progress_cb:
                progress_cb(model, thinking, prompt_dict, shot)
            return result

    # Round-robin: model is the innermost loop so each (thinking, prompt, shot)
    # slot runs all models back-to-back, minimising thermal/memory bias.
    tasks = [
        bounded(model, p, thinking, shot)
        for thinking in thinking_modes
        for p in prompts
        for shot in range(1, shots + 1)
        for model in models
    ]

    results = await asyncio.gather(*tasks)
    return list(results)
