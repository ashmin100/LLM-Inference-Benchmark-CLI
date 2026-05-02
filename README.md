<div align="center">

# 🔬 llm-bench

**Local LLM inference benchmark for Ollama — Dense vs MoE, Thinking vs Standard**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/ollama-compatible-black?logo=ollama)](https://ollama.com)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-silver?logo=apple)](https://www.apple.com/mac/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Measure what actually matters in LLM serving: **TTFT**, **throughput**, and **tail latency** — across architectures and inference modes, entirely on your local machine.

</div>

---

## Why llm-bench?

Raw parameter count is a poor proxy for real inference behavior.

A **30B MoE model** that activates only 3B parameters per token behaves very differently from a **14B dense model** — in throughput, latency distribution, and tail latency under load. Likewise, enabling **thinking mode** (chain-of-thought reasoning) reshapes the inference profile even for the *same model*.

llm-bench makes this comparison concrete and reproducible, with no API costs and no data leaving your machine.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              LLM Inference Benchmark — Summary                              │
├──────────────────────┬───────┬──────────────┬───────┬──────────┬───────────┤
│ Model                │ Think │ Category     │  TPS  │ TTFT p50 │ Lat p99   │
│                      │       │              │ (mean)│   (ms)   │   (ms)    │
├──────────────────────┼───────┼──────────────┼───────┼──────────┼───────────┤
│ qwen3:8b             │   —   │ reasoning    │  42.3 │      184 │       620 │
│ qwen3:8b             │   ✓   │ reasoning    │  38.1 │      190 │      4820 │
│ qwen3:14b            │   —   │ reasoning    │  24.7 │      310 │      1240 │
│ qwen3:14b            │   ✓   │ reasoning    │  22.9 │      318 │      9310 │
│ qwen3:30b-a3b (MoE)  │   —   │ reasoning    │  51.2 │      220 │       890 │  ← MoE beats 14B dense
│ qwen3:30b-a3b (MoE)  │   ✓   │ reasoning    │  47.8 │      228 │      7640 │
└──────────────────────┴───────┴──────────────┴───────┴──────────┴───────────┘
  🧠 = thinking mode ON | TPS: green ≥85% best | TTFT: green ≤1.3× best
```

> ⚠️ The table above uses **placeholder values** — run the benchmark on your own machine to get real numbers.

---

## Experiment Design

Tested on **Apple Silicon M4 Pro · 24 GB unified memory** · Ollama local inference (zero API cost)

### Models

| Model | Architecture | Total Params | Active params/token |
|---|---|---|---|
| `qwen3:8b` | Dense | 8B | 8B |
| `qwen3:14b` | Dense | 14B | 14B |
| `qwen3:30b-a3b` | **MoE** | 30B | ~3B |

`qwen3:30b-a3b` is the key subject: despite having 30B total parameters, each forward pass only activates ~3B — making it a direct test of whether MoE efficiency gains hold up locally against a dense model of similar quality.

### Inference Modes

Each model is tested in two modes:

| Mode | Description |
|---|---|
| **Standard** | Normal autoregressive decoding |
| **Thinking** | `think=True` — model generates internal reasoning tokens before responding |

Thinking mode is a strong stress test: it dramatically increases output token count, revealing how TTFT and tail latency behave under longer generation sequences.

### Prompt Suite

23 prompts across 5 categories at 3 difficulty levels (short / medium / long):

| Category | What it tests |
|---|---|
| `reasoning` | Logical deduction, math, multi-step inference |
| `coding` | Function generation, algorithm implementation, system design |
| `summarization` | Technical document compression (inference papers) |
| `creative` | Open-ended generation — highest TPS variance across models |
| `qa` | Factual recall and technical explanation |

Each configuration runs 3 shots per prompt for statistical stability (p95/p99 require repeated samples).

---

## Key Metrics

| Metric | What it measures | Why it matters |
|---|---|---|
| **TTFT** | Time from request to first token received | User-perceived responsiveness — the wait before output starts |
| **TPS** | Output tokens generated per second | Throughput for long-form generation |
| **p50 latency** | Median total response time | The typical experience |
| **p99 latency** | 99th percentile total response time | Worst-case behavior — determines SLA reliability |
| **Avg output tokens** | Mean tokens generated per prompt | Contextualizes TPS — a fair comparison requires similar output lengths |

> TPS is computed from Ollama's internal `eval_duration` counter, not wall-clock token counting, for higher accuracy.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ashmin100/llm-bench
cd llm-bench
pip install -e .

# 2. Pull models (one-time)
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull qwen3:30b-a3b

# 3. Run the full experiment
llm-bench compare \
  --models qwen3:8b,qwen3:14b,qwen3:30b-a3b \
  --mode both
```

Requires **Python 3.10+** and [Ollama](https://ollama.com) running locally (`ollama serve`).

---

## Usage

```bash
# Full experiment — all models, both modes, all categories
llm-bench compare --models qwen3:8b,qwen3:14b,qwen3:30b-a3b --mode both

# Quick check — standard mode only, two categories
llm-bench compare --models qwen3:8b,qwen3:14b --mode standard --category reasoning,coding

# Minimal run — 1 shot, no chart (fast iteration)
llm-bench compare --models qwen3:8b --shots 1 --no-chart

# Bring your own prompts
llm-bench compare --models qwen3:14b --prompts ./my_prompts.json

# See what's installed in Ollama
llm-bench list-models

# Run without installing
python -m llmbench compare --models qwen3:8b --mode standard
```

### All Options

```
Options:
  --models, -m        Comma-separated Ollama model names           [required]
  --category, -c      all | reasoning,coding,summarization,creative,qa
  --mode              standard | thinking | both         [default: both]
  --shots, -s         Repeats per prompt (min 2 for p99) [default: 3]
  --concurrency       Max simultaneous requests           [default: 1]
  --output, -o        Result directory                   [default: results/]
  --warmup/--no-warmup  Warmup request to avoid cold-start bias
  --no-chart          Skip PNG dashboard generation
```

---

## Output

Each run saves timestamped files to `results/` (gitignored — stays local):

```
results/
├── raw_20260502_143022.json         # Per-request measurements (all shots)
├── summary_20260502_143022.csv      # Aggregated stats per (model, mode, category)
└── benchmark_dashboard.png          # 2×2 visual dashboard
```

### Dashboard

The PNG dashboard contains four panels:

| Panel | Chart | Insight |
|---|---|---|
| Top-left | TPS mean ± std bar | Which model generates fastest? |
| Top-right | TTFT p50 bar | Which model responds first? |
| Bottom-left | Latency violin plot | How consistent is each model? |
| Bottom-right | Tail latency p99 | Who spikes under pressure? |

---

## Project Structure

```
llmbench/
├── cli.py           # Typer CLI — compare, list-models
├── runner.py        # Async benchmark engine (TTFT, TPS measurement)
├── metrics.py       # Result types + aggregation (p50/p95/p99)
├── reporter.py      # Rich color-coded terminal table
├── visualizer.py    # Matplotlib 2×2 dark-theme dashboard
├── storage.py       # JSON + CSV export
└── prompts/
    ├── reasoning.json
    ├── coding.json
    ├── summarization.json
    ├── creative.json
    └── qa.json
```

---

## Design Notes

**Why Ollama?** All inference runs locally — zero API cost, zero tokens sent externally. Any OpenAI-compatible endpoint works as a future extension.

**Warmup requests** are sent per model before measurement to eliminate cold-start bias from the first load.

**Thinking mode** inserts `<think>...</think>` reasoning blocks before the final response. TTFT includes the wait for the first thinking token — this is the user-perceived latency, which is what matters for interactive applications.

**MoE inference on Apple Silicon** uses Metal-accelerated compute. The `qwen3:30b-a3b` model fits within 24 GB unified memory while activating only ~3B parameters per forward pass, enabling a genuine comparison against dense models of similar generation quality.

---

## License

MIT

