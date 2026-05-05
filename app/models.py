"""
Pydantic schemas.

Two categories live in this file:

1. API layer schemas — request/response types for FastAPI endpoints.
2. Structured extraction targets — schemas that Instructor enforces on
   the local models during the extraction benchmark task. If a model's
   output doesn't fit the schema, Instructor retries until it does (or
   gives up after max_retries). This is the production-grade pattern for
   getting reliable structured data out of LLMs.
"""

from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# API layer — request/response schemas
# =============================================================================

class GenerateRequest(BaseModel):
    """POST /generate — single model, single prompt."""
    model: str = Field(..., description="Ollama model tag, e.g. 'llama3.2:3b'")
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    """What /generate returns. Mirrors GenerationResult but API-friendly."""
    model: str
    response: str
    time_to_first_token: float
    total_latency: float
    tokens_generated: int
    tokens_per_second: float


class CompareRequest(BaseModel):
    """POST /compare — same prompt, all three models in parallel."""
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.7


class CompareResponse(BaseModel):
    """Results from all models, keyed by model name."""
    prompt: str
    results: list[GenerateResponse]


class RouteRequest(BaseModel):
    """POST /route — classify and dispatch to the benchmark-best model."""
    prompt: str
    system: Optional[str] = None
    temperature: float = 0.7
    force_validation_failure: bool = Field(
        False,
        description=(
            "Test affordance: when true, force the first attempt's validation "
            "to fail so the fallback chain is exercised. Used by the routing "
            "test script; production callers should leave this false."
        ),
    )


class RouteAttempt(BaseModel):
    model: str
    latency_seconds: float
    validation_passed: bool
    validation_notes: str


class RouteTrace(BaseModel):
    """Full trace of a routing decision — what was tried and why."""
    classified_category: str
    classifier_confidence: float
    classifier_method: str
    classifier_latency_ms: float
    chosen_model: str
    fallback_chain: list[str]
    fallback_triggered: bool
    final_model: str
    attempts: list[RouteAttempt]
    total_latency_seconds: float
    validation_passed: bool
    validation_notes: str
    policy_notes: str


class RouteResponse(BaseModel):
    """POST /route — generated response plus routing trace."""
    response: str
    trace: RouteTrace


# =============================================================================
# Structured extraction targets
# =============================================================================
# These schemas are fed to Instructor during the extraction benchmark task.
# The benchmark measures: (a) does the model return valid structured data?
# (b) how many retries did it take? (c) are the field values actually correct?

class PersonInfo(BaseModel):
    """
    Extract person details from unstructured bio text.

    A canonical extraction task — biographical text is messy and models
    have to decide what's present, what's missing, and how to normalize.
    """
    name: str = Field(..., description="Full name of the person")
    age: Optional[int] = Field(None, description="Age in years, if mentioned")
    occupation: Optional[str] = Field(None, description="Current job or role")
    location: Optional[str] = Field(None, description="City and/or country of residence")


class MeetingAction(BaseModel):
    """A single action item extracted from meeting notes."""
    task: str = Field(..., description="What needs to be done")
    assignee: Optional[str] = Field(None, description="Person responsible")
    due_date: Optional[date] = Field(None, description="Deadline, if mentioned")
    priority: Literal["low", "medium", "high"] = Field(
        "medium", description="Urgency level inferred from tone"
    )


class MeetingExtraction(BaseModel):
    """
    Extract structured action items from a meeting notes blob.

    Harder than PersonInfo because it requires: parsing an unknown-length
    list, inferring priority from tone, and date normalization. This task
    is where small models usually start to break down — a useful stress test.
    """
    meeting_topic: str
    action_items: list[MeetingAction]
    next_meeting_date: Optional[date] = None


# =============================================================================
# Benchmark result schemas
# =============================================================================
# Not exposed over the API directly — used internally to pass benchmark
# results between the runner, the database, and the chart generator.

class BenchmarkRun(BaseModel):
    """One prompt × one model = one BenchmarkRun."""
    model: str
    task_category: Literal["reasoning", "summarization", "extraction", "code"]
    task_id: str
    prompt: str
    response: str
    time_to_first_token: float
    total_latency: float
    tokens_generated: int
    tokens_per_second: float
    peak_memory_mb: float
    quality_score: Optional[float] = Field(
        None, description="0.0–1.0, scoring method depends on task category"
    )
    quality_notes: Optional[str] = Field(
        None, description="LLM-judge rationale or structured-output error details"
    )
