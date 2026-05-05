# Local-SLM-Benchmark

A benchmark suite and FastAPI service for comparing three locally-hosted small language models on an 8 GB M2 MacBook Air: Gemma 2 (2B), Llama 3.2 (3B), and Qwen 2.5 (7B).

Measures latency, throughput, memory, and quality across reasoning, summarization, structured extraction, and code generation tasks. Produces a reproducible benchmark report with per-model and per-category findings.

**Full analysis and findings: [BENCHMARKS.md](./BENCHMARKS.md)**

## What's in here

- **An Ollama-backed FastAPI service** with `/generate`, `/compare`, `/route`, `/routing/policy`, `/routing/decisions`, and `/history` endpoints, plus auto-generated Swagger docs at `/docs`.
- **A benchmark harness** that runs 14 prompts against each of three models, scores every response, and persists results to SQLite.
- **Four scoring strategies**: LLM-as-judge via Claude Sonnet 4.5 for reasoning and summarization, Pydantic-schema enforcement via Instructor for structured extraction, and sandboxed code execution for code generation.
- **A benchmark-driven request router** that classifies inbound prompts and dispatches them to the model the benchmark identifies as best for that category, with a configurable fallback chain on validation failure.
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

**Run the routing acceptance tests:**

```bash
python scripts/eval_router.py             # classifier-only, 20-prompt eval
python scripts/eval_router.py --full      # also exercises a few full /route calls
python scripts/eval_router.py --force-fb  # forced-failure smoke test for fallback
```

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
├── app/routing/
│   ├── __init__.py          # Public surface (Router, RoutingPolicy, PromptClassifier)
│   ├── classifier.py        # Heuristic + Gemma 2B zero-shot prompt classifier
│   ├── policy.py            # Reads runs.db; derives best-model-per-category mapping
│   └── router.py            # Classify → select → generate → validate → fallback
├── scripts/
│   ├── run_benchmark.py     # Full benchmark sweep
│   ├── generate_charts.py   # PNG chart generation
│   ├── inspect_runs.py      # Run browser with filtering
│   └── eval_router.py       # Routing acceptance tests (classifier accuracy + p95)
├── charts/                  # Generated PNGs (checked in)
├── data/runs.db             # SQLite store (gitignored)
├── BENCHMARKS.md            # Full report
└── README.md
```

## Routing

The `/route` endpoint operationalizes the benchmark findings. Instead of callers having to read the report and pick a model by hand, the router classifies each prompt and dispatches it to whichever model the latest benchmark batch identifies as best for that category.

### Flow

```
              ┌─────────────────┐
   prompt ───▶│   classifier    │  heuristic pre-pass (regex), then
              │  (Gemma 2B)     │  Gemma 2B zero-shot via Instructor
              └────────┬────────┘  if heuristic abstains
                       │
                       ▼
              ┌─────────────────┐
              │     policy      │  reads latest batch from SQLite,
              │ (mean quality)  │  picks highest-quality model per
              └────────┬────────┘  category, builds fallback chain
                       │
                       ▼
              ┌─────────────────┐
              │   generation    │  primary model in the chain
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐  passes ─▶ return response + trace
              │   validation    │
              └────────┬────────┘  fails ──▶ next model in chain;
                       │                     if chain exhausted, return
                       ▼                     last response with the
                  (loop on fail)             failure flagged in the trace
```

### Categories

Five buckets: `reasoning`, `summarization`, `extraction`, `code`, and `general`. The first four mirror the benchmark categories. `general` is the safe-default bucket, used when classifier confidence is below threshold (default 0.6) or when the prompt clearly fits none of the others. `general` always routes to Qwen 7B.

### Selection rule

For each of the four real categories, the policy picks the model with the **highest mean `quality_score`** in the latest benchmark batch. Ties are broken by lower mean latency. The mapping is read from the DB at request time, so re-running the benchmark immediately changes routing; no code changes, no restart.

Pareto-aware and configurable selection policies are deliberately out of scope for v1; the policy module is structured so they slot in later without touching callers.

### Fallback chain

For each category, the chain is the full ranked list of models (best → worst by quality) plus Qwen 7B as a last resort if it isn't already in the list. When a validator returns `passed=False`, the router walks to the next model in the chain. If the whole chain fails, the last response is still returned but the trace flags `validation_passed=false` so the caller can decide what to do.

The validation hook is the runtime validator from the [Output validation pipeline](#output-validation-pipeline) section. The router treats the hook as opaque, so swapping in a different validator is a one-line change in the FastAPI startup wiring.

### Persistence

Every routing decision is logged to a `routing_decisions` table with timestamp, prompt hash, classified category, classifier confidence, latency, chosen model, fallback chain, fallback-triggered flag, and validation outcome. Browse via `GET /routing/decisions?limit=N`.

### Example

```bash
curl -s -X POST http://localhost:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Write a Python function to reverse a string."}' | jq
```

```json
{
  "response": "def reverse(s: str) -> str:\n    return s[::-1]\n",
  "trace": {
    "classified_category": "code",
    "classifier_confidence": 0.85,
    "classifier_method": "heuristic",
    "classifier_latency_ms": 0.4,
    "chosen_model": "llama3.2:3b",
    "fallback_chain": ["llama3.2:3b", "qwen2.5:7b", "gemma2:2b"],
    "fallback_triggered": false,
    "final_model": "llama3.2:3b",
    "validation_passed": true,
    "policy_notes": "Derived from 12 (model, category) groups in latest batch."
  }
}
```

To inspect the current category → model mapping:

```bash
curl -s http://localhost:8000/routing/policy | jq
```

## Output validation pipeline

The benchmark scoring layer runs offline against a fixed prompt suite. That works for measurement but ships unvalidated output to runtime callers. The validation pipeline closes that gap: every response from `/generate` and `/route` is checked against a category-aware lightweight validator before it leaves the API.

The benchmark scorers stay where they are; the runtime validators are deliberately faster and more conservative. Schema match plus full test-case execution are appropriate for offline batches; JSON parsability and an AST check plus a sandboxed smoke run are appropriate for the request path.

### Decision flow

```
              ┌─────────────────┐
   response ─▶│  category-aware │  extraction → JSON parse
              │    validator    │  code       → ast.parse + sandbox exec
              └────────┬────────┘  summarization → length / no fences
                       │            reasoning → length / not-a-refusal
                       │            general → non-empty / length cap
                       ▼
              ┌─────────────────┐ pass ─▶ return response + validation block
              │  outcome + log  │ fail ─▶ recommended_action per config:
              │  (one DB row    │           retry      → caller retries
              │   per attempt)  │           fallback   → router walks chain
              └─────────────────┘           hard_error → /generate returns 422
```

### Categories and checks

Each category has a tight, cheap check. The validators are intentionally narrow: anything more expensive belongs in offline scoring, not on the request path.

- **extraction**: strip any markdown fence, then `json.loads`. Object or array required. JSON-schema match is left for callers that opt in.
- **code**: `ast.parse` for syntax, then a sandboxed subprocess `exec` at module level with a 5s wall-clock timeout, a 200 MB address-space cap, and `RLIMIT_CPU` / `RLIMIT_NOFILE` ceilings. PYTHONPATH is cleaned and the working directory is a fresh tempdir.
- **summarization**: word count between 5 and 500 (configurable), no markdown code fences.
- **reasoning**: non-empty, within length bounds, doesn't open with a refusal-style marker (`"I cannot answer"` and friends).
- **general**: non-empty after strip, within length bounds. The catch-all bucket; we don't try to be opinionated about chit-chat shape.

### Failure actions

When validation fails, the system takes one of three configured actions:

- **retry** — caller retries the same model with adjusted parameters (not yet wired into endpoints; reserved for callers that want it).
- **fallback** — the router walks to the next model in the category's fallback chain. Default for `/route`.
- **hard_error** — the endpoint returns HTTP 422 with the validator name, category, notes, and request ID. Default for `/generate`.

Defaults live in `config/validation.json` and are re-read on every request, so a config change takes effect immediately. Env-var overrides (`VALIDATION_CODE_EXEC_TIMEOUT_SECONDS`, `VALIDATION_CODE_MEMORY_LIMIT_MB`, `VALIDATION_CODE_EXECUTE_IN_SANDBOX`, `VALIDATION_DEFAULT_ON_FAILURE`) layer on top.

### Sandbox notes

The code executor uses `subprocess.run` with a wall-clock timeout plus POSIX `setrlimit` calls in the child. This is the conventional non-containerised pattern; the threat model is "model output that shouldn't run for 30s and shouldn't allocate 8 GB," not "actively malicious code trying to exfiltrate." On Linux all three rlimits are reliably enforced; on macOS Apple Silicon `RLIMIT_AS` is best-effort (the wall-clock timeout is the practical safety net).

If you need stronger isolation, swap `app/validation/sandbox.py` for a containerised executor; the public surface (a single `execute(code, *, timeout_seconds, memory_limit_mb)` function returning a `SandboxResult`) was kept narrow specifically so this swap is a contained change.

### Persistence

Every validation outcome is logged to a `validation_outcomes` table with timestamp, request ID, endpoint, category, validator name, pass/fail, action taken, latency, and notes. Browse via `GET /validation/outcomes?limit=N`. The router writes one row per attempt, so the chain walk is fully visible: a request that started on Llama 3.2 3B and ended on Qwen 7B has two rows linked by the same `request_id`.

### Example: validation failure on `/generate`

```bash
curl -i -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma2:2b",
    "prompt": "Extract the name and age as JSON: Dr. Patel, 47.",
    "category": "extraction"
  }'
```

If the model returns prose instead of JSON, the response is HTTP 422 with the failure detail in the body and the outcome persisted to `validation_outcomes`. If it returns valid JSON, the response is the usual 200 with a `validation` block containing the outcome.

## Design notes

A few non-obvious decisions worth calling out:

- **Warmup before measurement.** The first generation against a freshly-loaded model includes a 3–5 second cold-start that would otherwise skew TTFT numbers. Each model is warmed with a throwaway prompt before its real prompts run.
- **LLM-as-judge uses a stronger model than the models under test.** Claude Sonnet 4.5 scores Gemma/Llama/Qwen outputs. A judge that shares systematic weaknesses with the models it's scoring will miss errors.

## Stack

Python 3.12, FastAPI, Ollama, SQLite, Pydantic, Instructor, Anthropic SDK, httpx, matplotlib, rich.
