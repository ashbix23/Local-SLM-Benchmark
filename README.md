# Local-SLM-Benchmark

A benchmark suite and FastAPI service for comparing three locally-hosted small language models on an 8 GB M2 MacBook Air: Gemma 2 (2B), Llama 3.2 (3B), and Qwen 2.5 (7B).

Measures latency, throughput, memory, and quality across reasoning, summarization, structured extraction, and code generation tasks. Produces a reproducible benchmark report with per-model and per-category findings.

**Full analysis and findings: [BENCHMARKS.md](./BENCHMARKS.md)**

## What's in here

- **An Ollama-backed FastAPI service** with `/generate`, `/compare`, and `/history` endpoints, plus auto-generated Swagger docs at `/docs`.
- **A benchmark harness** that runs 14 prompts against each of three models, scores every response, and persists results to SQLite.
- **Four scoring strategies**: LLM-as-judge via Claude Sonnet 4.5 for reasoning and summarization, Pydantic-schema enforcement via Instructor for structured extraction, and sandboxed code execution for code generation.
- **An inspection CLI** for browsing individual runs by model, category, or task.
- **A chart generator** that produces four PNG charts from the latest benchmark batch.

## Headline findings

![Quality by task category](charts/quality_by_category.png)

- Qwen 2.5 (7B) wins three of four categories but costs ~3x the latency and ~2x the memory of the smaller models.
- Llama 3.2 (3B) **underperforms the smaller Gemma 2B on reasoning** (0.45 vs 0.60). Larger parameter count did not predict better reasoning.
- Llama 3.2 (3B) **ties Qwen 7B on code generation** (1.00) at roughly a third of the footprint.
- Memory usage is non-linear with parameter count due to Q4 quantization as Gemma and Llama are both ~2.5 GB despite a 50% parameter difference.

Full methodology, qualitative failure analysis, and recommendations in [BENCHMARKS.md](./BENCHMARKS.md).

## Quickstart

**Prerequisites:** Ollama installed and running, Python 3.12+, an Anthropic API key (used by the LLM judge).

```bash
# Pull the three benchmark models
ollama pull gemma2:2b
ollama pull llama3.2:3b
ollama pull qwen2.5:7b

# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

**Run the full benchmark sweep** (~15–25 minutes on M2 Air, close other apps first):

```bash
python scripts/run_benchmark.py
```

**Generate charts** from the latest results:

```bash
python scripts/generate_charts.py
```

**Browse individual runs:**

```bash
python scripts/inspect_runs.py --failures
python scripts/inspect_runs.py --task reasoning_05
python scripts/inspect_runs.py --model qwen2.5:7b --category extraction
```

**Start the API:**

```bash
uvicorn app.main:app --reload --reload-dir app
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger docs.

## Project structure

```
local-slm-benchmark/
├── app/
│   ├── main.py              # FastAPI service
│   ├── ollama_client.py     # Streaming HTTP client with timing capture
│   ├── benchmark.py         # Single-run orchestrator with warmup + memory tracking
│   ├── database.py          # SQLite persistence layer
│   ├── models.py            # Pydantic schemas (API + extraction targets + benchmark rows)
│   ├── prompts.py           # The 14-prompt benchmark suite
│   └── scoring/
│       ├── __init__.py      # Scoring dispatcher
│       ├── code.py          # Code execution scorer
│       ├── extraction.py    # Instructor + Pydantic scorer
│       └── judge.py         # LLM-as-judge (Claude Sonnet 4.5)
├── scripts/
│   ├── run_benchmark.py     # Full benchmark sweep
│   ├── generate_charts.py   # PNG chart generation
│   └── inspect_runs.py      # Run browser with filtering
├── charts/                  # Generated PNGs (checked in)
├── data/runs.db             # SQLite store (gitignored)
├── BENCHMARKS.md            # Full report
└── README.md
```

## Design notes

A few non-obvious decisions worth calling out:

- **Memory measurement via Ollama's `/api/ps` endpoint, not process RSS.** On Apple Silicon, the Metal framework manages model weights in GPU-shared memory outside any process's RSS accounting. Using `/api/ps` gives the actual VRAM footprint, which is what matters for hardware planning.
- **Sequential model loading with explicit unload between sweeps.** On 8 GB hardware, running Qwen 7B without unloading the previous models would trigger swap and contaminate latency measurements. The runner unloads via `keep_alive: 0` between models.
- **Warmup before measurement.** The first generation against a freshly-loaded model includes a 3–5 second cold-start that would otherwise skew TTFT numbers. Each model is warmed with a throwaway prompt before its real prompts run.
- **LLM-as-judge uses a stronger model than the models under test.** Claude Sonnet 4.5 scores Gemma/Llama/Qwen outputs. A judge that shares systematic weaknesses with the models it's scoring will miss errors.

## Stack

Python 3.12, FastAPI, Ollama, SQLite, Pydantic, Instructor, Anthropic SDK, httpx, matplotlib, rich.
