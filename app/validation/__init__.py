"""
Runtime output validation pipeline.

Sits between model generation and the response leaving the API. For each
output we run a category-aware lightweight check (JSON parse for
extraction, ast.parse plus optional sandboxed execution for code, length
and format sanity for summarization/reasoning, minimum well-formedness
for general). Failures are surfaced via a configurable action: retry,
fallback, or hard_error.

The benchmark scoring layer (`app.scoring`) is offline and uses an LLM
judge plus full test-case execution. That's appropriate for batch
evaluation but too slow and too expensive for the runtime path. This
package is the fast, conservative cousin: catches malformed or obviously
broken outputs without needing a judge.

Public surface:
    validate(category, prompt, response, *, ...)  -> ValidationOutcome
    hook(prompt, response, category)               -> (passed, notes)

`hook` is a drop-in for the ValidationHook contract in
`app.routing.router`, so the router replaces its placeholder validator
by passing this function in.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.database import insert_validation_outcome
from app.validation.config import load_config, get_category_config, resolve_action
from app.validation.validators import (
    validate_code,
    validate_extraction,
    validate_general,
    validate_reasoning,
    validate_summarization,
)


VALID_CATEGORIES = ("reasoning", "summarization", "extraction", "code", "general")


@dataclass
class ValidationOutcome:
    """Result of validating a single response."""
    passed: bool
    notes: str
    validator: str
    category: str
    latency_ms: float
    recommended_action: str
    request_id: str


_VALIDATORS = {
    "extraction": validate_extraction,
    "code": validate_code,
    "summarization": validate_summarization,
    "reasoning": validate_reasoning,
    "general": validate_general,
}


def validate(
    category: str,
    prompt: str,
    response: str,
    *,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    persist: bool = True,
) -> ValidationOutcome:
    """
    Run the category-appropriate validator on a response.

    `request_id` correlates the outcome with the request that produced it.
    Generated as a fresh UUID4 if not supplied. `endpoint` is "generate"
    or "route" and influences the recommended action when the validator
    fails (via endpoint_defaults in config).
    """
    if category not in VALID_CATEGORIES:
        category = "general"

    if request_id is None:
        request_id = uuid.uuid4().hex[:16]

    config = load_config()
    category_config = get_category_config(config, category)
    validator = _VALIDATORS[category]

    start = time.perf_counter()
    passed, notes, validator_name = validator(prompt, response, category_config)
    latency_ms = (time.perf_counter() - start) * 1000.0

    recommended_action = (
        "passed" if passed else resolve_action(config, category, endpoint)
    )

    outcome = ValidationOutcome(
        passed=passed,
        notes=notes,
        validator=validator_name,
        category=category,
        latency_ms=latency_ms,
        recommended_action=recommended_action,
        request_id=request_id,
    )

    if persist:
        try:
            insert_validation_outcome({
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "endpoint": endpoint or "unknown",
                "category": category,
                "validator": validator_name,
                "passed": passed,
                "action_taken": recommended_action,
                "latency_ms": latency_ms,
                "notes": notes,
            })
        except Exception:
            # Persistence failure must not break the request path.
            pass

    return outcome


def hook(prompt: str, response: str, category: str) -> tuple[bool, str]:
    """
    Drop-in ValidationHook for `app.routing.router.Router`.

    The router calls this once per fallback attempt; each call persists a
    validation_outcomes row, so the audit trail captures every attempt
    even when the chain walks past the first model.
    """
    outcome = validate(category, prompt, response, endpoint="route")
    return outcome.passed, outcome.notes


__all__ = [
    "ValidationOutcome",
    "validate",
    "hook",
    "VALID_CATEGORIES",
]
