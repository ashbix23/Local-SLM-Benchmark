"""
FastAPI surface for the local SLM benchmark project.

Exposes the Ollama-backed generation pipeline as a usable HTTP API:
the same code paths the benchmark exercises are reachable from any
HTTP client (curl, Postman, frontend, another service).

Endpoints:
    GET  /              — health check
    GET  /models        — list models currently pulled in Ollama
    POST /generate      — one model, one prompt, full metrics in response
    POST /compare       — same prompt against all three benchmark models in
                          parallel; returns side-by-side metrics
    GET  /history?limit — recent runs from the SQLite DB

Run:
    uvicorn app.main:app --reload

The --reload flag picks up code changes without restarting; useful while
iterating. For "production" demo use, drop --reload and increase workers.
"""

import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.ollama_client import OllamaClient
from app.models import GenerateRequest, GenerateResponse, CompareRequest, CompareResponse
from app.database import init_db, get_connection, DB_PATH


# Models we expose via /compare. Mirrors the benchmark lineup so the API
# story matches the report story.
COMPARE_MODELS = ["gemma2:2b", "llama3.2:3b", "qwen2.5:7b"]


app = FastAPI(
    title="Local SLM Benchmark API",
    description=(
        "HTTP surface over three Ollama-served small language models "
        "(Gemma 2B, Llama 3.2 3B, Qwen 2.5 7B). Wraps the same generation "
        "and metrics pipeline used by the benchmark suite."
    ),
    version="0.1.0",
)

# Permissive CORS — this is a local-only dev API. If you ever expose it
# beyond localhost, lock this down to specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_client = OllamaClient()


@app.on_event("startup")
async def _startup() -> None:
    """Make sure the DB exists before the first request."""
    init_db()


# =============================================================================
# Health & metadata
# =============================================================================

@app.get("/")
async def root() -> dict:
    """Health check + basic project info."""
    return {
        "service": "Local SLM Benchmark API",
        "status": "ok",
        "models": COMPARE_MODELS,
    }


@app.get("/models")
async def list_models() -> dict:
    """Models currently pulled in the local Ollama instance."""
    try:
        available = await _client.list_models()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")
    return {"available": available, "benchmark_lineup": COMPARE_MODELS}


# =============================================================================
# Generation
# =============================================================================

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Single-model generation. Returns the response text plus full timing
    metrics — same data captured by the benchmark layer.
    """
    try:
        result = await _client.generate(
            model=request.model,
            prompt=request.prompt,
            system=request.system,
            temperature=request.temperature,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Generation failed: {type(exc).__name__}: {exc}",
        )

    return GenerateResponse(
        model=result.model,
        response=result.response_text,
        time_to_first_token=result.time_to_first_token,
        total_latency=result.total_latency,
        tokens_generated=result.tokens_generated,
        tokens_per_second=result.tokens_per_second,
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest) -> CompareResponse:
    """
    Run the same prompt against all three benchmark models in parallel.

    Note on parallelism: on 8 GB hardware, parallel generation across
    three models would force swap. Ollama serializes generation requests
    internally per loaded model, but cross-model parallelism is fine
    *if* the models are all already loaded. In practice for an 8 GB Air,
    expect this endpoint to load models on demand — first call after a
    cold start will be slow as Ollama swaps models in/out.
    """
    async def _run_one(model: str) -> Optional[GenerateResponse]:
        try:
            result = await _client.generate(
                model=model,
                prompt=request.prompt,
                system=request.system,
                temperature=request.temperature,
            )
            return GenerateResponse(
                model=result.model,
                response=result.response_text,
                time_to_first_token=result.time_to_first_token,
                total_latency=result.total_latency,
                tokens_generated=result.tokens_generated,
                tokens_per_second=result.tokens_per_second,
            )
        except Exception:
            # Per-model failure shouldn't break the whole comparison
            return None

    results = await asyncio.gather(*[_run_one(model) for model in COMPARE_MODELS])
    successful = [response for response in results if response is not None]

    if not successful:
        raise HTTPException(
            status_code=502,
            detail="All models failed to generate. Is Ollama running?",
        )

    return CompareResponse(prompt=request.prompt, results=successful)


# =============================================================================
# History
# =============================================================================

class HistoryRow(BaseModel):
    """One row from the runs DB, API-friendly shape."""
    id: int
    run_timestamp: str
    model: str
    task_category: str
    task_id: str
    total_latency: float
    tokens_per_second: float
    quality_score: Optional[float] = None
    quality_notes: Optional[str] = None


class HistoryResponse(BaseModel):
    count: int
    rows: list[HistoryRow]


@app.get("/history", response_model=HistoryResponse)
async def history(limit: int = Query(50, ge=1, le=500)) -> HistoryResponse:
    """
    Recent benchmark runs from the SQLite store.

    `limit` clamps between 1 and 500. Default 50 is enough for a dashboard
    or quick look without flooding the response.
    """
    with get_connection(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT id, run_timestamp, model, task_category, task_id,
                   total_latency, tokens_per_second, quality_score, quality_notes
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history_rows = [HistoryRow(**dict(row)) for row in rows]
    return HistoryResponse(count=len(history_rows), rows=history_rows)
