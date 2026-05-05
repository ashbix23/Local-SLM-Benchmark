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
    POST /route         : classify the prompt, dispatch to the benchmark-
                          best model for that category, with fallback chain
    GET  /history?limit — recent runs from the SQLite DB
    GET  /routing/decisions?limit : recent routing decisions for auditing
    GET  /routing/policy : current category → model mapping derived from DB
    GET  /validation/outcomes?limit : recent validation outcomes for auditing

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
from app.models import (
    GenerateRequest, GenerateResponse, ValidationBlock,
    CompareRequest, CompareResponse,
    RouteRequest, RouteResponse, RouteTrace, RouteAttempt,
)
from app.database import (
    init_db, get_connection, DB_PATH,
    fetch_recent_routing_decisions, fetch_recent_validation_outcomes,
)
from app.routing import Router
from app.routing.policy import derive_mapping
from app.validation import validate as validate_output, hook as validation_hook


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
_router = Router(client=_client, validator=validation_hook)


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
    metrics, same data captured by the benchmark layer.

    Runtime validation runs on the output before this returns. The
    `category` field on the request controls which validator fires; if
    omitted, the output is treated as `general` (minimal well-formedness
    check). On validation failure the endpoint returns 422 by default
    (see `endpoint_defaults.generate.on_failure_override` in
    `config/validation.json`).
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

    category = request.category or "general"
    outcome = validate_output(
        category,
        prompt=request.prompt,
        response=result.response_text,
        endpoint="generate",
    )

    if not outcome.passed and outcome.recommended_action == "hard_error":
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validation_failed",
                "validator": outcome.validator,
                "category": outcome.category,
                "notes": outcome.notes,
                "request_id": outcome.request_id,
            },
        )

    return GenerateResponse(
        model=result.model,
        response=result.response_text,
        time_to_first_token=result.time_to_first_token,
        total_latency=result.total_latency,
        tokens_generated=result.tokens_generated,
        tokens_per_second=result.tokens_per_second,
        validation=ValidationBlock(
            passed=outcome.passed,
            notes=outcome.notes,
            validator=outcome.validator,
            category=outcome.category,
            latency_ms=outcome.latency_ms,
            action_taken=outcome.recommended_action,
            request_id=outcome.request_id,
        ),
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


# =============================================================================
# Routing
# =============================================================================

@app.post("/route", response_model=RouteResponse)
async def route(request: RouteRequest) -> RouteResponse:
    """
    Benchmark-driven routing.

    Classifies the prompt, looks up the highest-quality model for that
    category from the SQLite benchmark store, and dispatches generation.
    On validation failure, walks the fallback chain.

    The response includes a `trace` field exposing the full decision:
    classified category, confidence, chosen model, fallback chain, which
    attempt(s) ran, and final model. Audit data is also persisted.
    """
    try:
        result = await _router.route(
            prompt=request.prompt,
            system=request.system,
            temperature=request.temperature,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Routing failed: {type(exc).__name__}: {exc}",
        )

    trace_dict = result.trace_dict()
    return RouteResponse(
        response=result.response,
        trace=RouteTrace(
            classified_category=trace_dict["classified_category"],
            classifier_confidence=trace_dict["classifier_confidence"],
            classifier_method=trace_dict["classifier_method"],
            classifier_latency_ms=trace_dict["classifier_latency_ms"],
            chosen_model=trace_dict["chosen_model"],
            fallback_chain=trace_dict["fallback_chain"],
            fallback_triggered=trace_dict["fallback_triggered"],
            final_model=trace_dict["final_model"],
            attempts=[RouteAttempt(**attempt) for attempt in trace_dict["attempts"]],
            total_latency_seconds=trace_dict["total_latency_seconds"],
            validation_passed=trace_dict["validation_passed"],
            validation_notes=trace_dict["validation_notes"],
            policy_notes=trace_dict["policy_notes"],
        ),
    )


@app.get("/routing/policy")
async def routing_policy() -> dict:
    """
    Current category → model mapping derived from the latest benchmark
    batch. Useful for confirming the router is reading the data you
    expect, especially after re-running the benchmark.
    """
    policy = derive_mapping()
    return {
        "mapping": policy.mapping,
        "fallback_chains": policy.chains,
        "derived_from_rows": policy.derived_from_rows,
        "notes": policy.derivation_notes,
    }


@app.get("/routing/decisions")
async def routing_decisions(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Recent routing decisions for auditing and drift detection."""
    rows = fetch_recent_routing_decisions(limit=limit)
    return {"count": len(rows), "rows": rows}


# =============================================================================
# Validation
# =============================================================================

@app.get("/validation/outcomes")
async def validation_outcomes(limit: int = Query(50, ge=1, le=500)) -> dict:
    """
    Recent runtime-validation outcomes for auditing and drift detection.

    Each row corresponds to one validator call against one model output.
    The router writes one row per attempt (so the chain walk is
    visible); /generate writes exactly one row per request.
    """
    rows = fetch_recent_validation_outcomes(limit=limit)
    return {"count": len(rows), "rows": rows}
