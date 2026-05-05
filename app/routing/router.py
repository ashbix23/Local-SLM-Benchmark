"""
Request router: classify → select → generate → validate → maybe fall back.

Public surface is the `Router` class with one async method, `route`. The
return value is a `RoutingResult` carrying both the generated response
and the full routing trace (classification, chosen model, fallback chain
walked, validation outcome, latencies). The trace is what the API
exposes so callers can audit decisions, and what we persist to SQLite.

Validation hook contract:
  A `ValidationHook` is any callable mapping (prompt, response, category)
  to a (passed: bool, notes: str) tuple. The Router walks the fallback
  chain when the hook returns False. Until SLM-202 lands, the default
  hook always returns True (placeholder), per the ticket's instructions.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

from app.database import DB_PATH, insert_routing_decision
from app.ollama_client import OllamaClient, GenerationResult
from app.routing.classifier import Classification, PromptClassifier
from app.routing.policy import RoutingPolicy, derive_mapping


# A validation hook returns (passed, notes). When it returns False, the
# router walks the fallback chain.
ValidationHook = Callable[[str, str, str], tuple[bool, str]]


def _placeholder_validator(prompt: str, response: str, category: str) -> tuple[bool, str]:
    """
    Default validation hook. Always passes.

    Replaced by the real validator from SLM-202 once that ships. The
    router treats the hook as opaque, so swapping in the real one is a
    one-line change in main.py.
    """
    return True, "placeholder validator: always passes"


@dataclass
class FallbackAttempt:
    """One model attempt inside route()."""
    model: str
    response: str
    latency_seconds: float
    validation_passed: bool
    validation_notes: str


@dataclass
class RoutingResult:
    """Full trace of a routing decision plus the final response."""
    prompt: str
    response: str
    classification: Classification
    chosen_model: str
    fallback_chain: list[str]
    fallback_triggered: bool
    final_model: str
    attempts: list[FallbackAttempt] = field(default_factory=list)
    total_latency_seconds: float = 0.0
    validation_passed: bool = True
    validation_notes: str = ""
    policy_notes: str = ""

    def trace_dict(self) -> dict:
        """API-friendly view of the trace (no Python objects)."""
        return {
            "classified_category": self.classification.category,
            "classifier_confidence": self.classification.confidence,
            "classifier_method": self.classification.method,
            "classifier_latency_ms": self.classification.latency_ms,
            "chosen_model": self.chosen_model,
            "fallback_chain": self.fallback_chain,
            "fallback_triggered": self.fallback_triggered,
            "final_model": self.final_model,
            "attempts": [
                {
                    "model": attempt.model,
                    "latency_seconds": attempt.latency_seconds,
                    "validation_passed": attempt.validation_passed,
                    "validation_notes": attempt.validation_notes,
                }
                for attempt in self.attempts
            ],
            "total_latency_seconds": self.total_latency_seconds,
            "validation_passed": self.validation_passed,
            "validation_notes": self.validation_notes,
            "policy_notes": self.policy_notes,
        }


class Router:
    """
    Stateless orchestrator. Holds references to the client, classifier,
    and validator; resolves the policy on every request so DB updates
    take effect without a restart.
    """

    def __init__(
        self,
        client: OllamaClient,
        classifier: Optional[PromptClassifier] = None,
        validator: Optional[ValidationHook] = None,
        db_path: Path = DB_PATH,
    ):
        self.client = client
        self.classifier = classifier or PromptClassifier()
        self.validator = validator or _placeholder_validator
        self.db_path = db_path

    def _resolve_policy(self) -> RoutingPolicy:
        """Cheap aggregation query; re-run on every request for freshness."""
        return derive_mapping(self.db_path)

    async def route(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        force_validation_failure: bool = False,
    ) -> RoutingResult:
        """
        Classify → pick → generate → validate → fall back if needed.

        ``force_validation_failure`` is a test affordance: when True, the
        first attempt's validation is forced to fail regardless of what
        the validator returns. This is how the test script verifies the
        fallback path triggers on a forced failure. It does not bypass
        validation on subsequent attempts; those still go through the
        real validator.
        """
        overall_start = time.perf_counter()

        classification = self.classifier.classify(prompt)
        policy = self._resolve_policy()
        chain = policy.chain_for(classification.category)
        chosen_model = chain[0]

        attempts: list[FallbackAttempt] = []
        final_response = ""
        final_model = chosen_model
        validation_passed = False
        validation_notes = ""

        for index, model in enumerate(chain):
            generation = await self._generate(model, prompt, system, temperature)

            if force_validation_failure and index == 0:
                passed, notes = False, "forced validation failure (test mode)"
            else:
                passed, notes = self.validator(prompt, generation.response_text, classification.category)

            attempts.append(FallbackAttempt(
                model=model,
                response=generation.response_text,
                latency_seconds=generation.total_latency,
                validation_passed=passed,
                validation_notes=notes,
            ))

            if passed:
                final_response = generation.response_text
                final_model = model
                validation_passed = True
                validation_notes = notes
                break

        if not validation_passed:
            # Whole chain failed. Surface the last attempt as the response,
            # but flag validation as failed so the caller can decide what
            # to do (the response is still usable; we just couldn't verify).
            last = attempts[-1]
            final_response = last.response
            final_model = last.model
            validation_notes = f"all {len(chain)} models failed validation; last: {last.validation_notes}"

        total_latency = time.perf_counter() - overall_start
        result = RoutingResult(
            prompt=prompt,
            response=final_response,
            classification=classification,
            chosen_model=chosen_model,
            fallback_chain=chain,
            fallback_triggered=(final_model != chosen_model) or not validation_passed,
            final_model=final_model,
            attempts=attempts,
            total_latency_seconds=total_latency,
            validation_passed=validation_passed,
            validation_notes=validation_notes,
            policy_notes=policy.derivation_notes,
        )

        self._persist(result)
        return result

    async def _generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str],
        temperature: float,
    ) -> GenerationResult:
        return await self.client.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
        )

    def _persist(self, result: RoutingResult) -> None:
        """
        Insert the routing decision into SQLite.

        Failures here must not break the request; observability shouldn't
        be on the critical path. We swallow the exception and move on; if
        it becomes a problem we'll add a logger.
        """
        prompt_hash = hashlib.sha256(result.prompt.encode("utf-8")).hexdigest()[:16]
        prompt_preview = result.prompt[:200]

        try:
            insert_routing_decision(
                {
                    "decided_at": datetime.now(timezone.utc).isoformat(),
                    "prompt_hash": prompt_hash,
                    "prompt_preview": prompt_preview,
                    "classified_category": result.classification.category,
                    "classifier_confidence": result.classification.confidence,
                    "classifier_latency_ms": result.classification.latency_ms,
                    "chosen_model": result.chosen_model,
                    "fallback_chain": ",".join(result.fallback_chain),
                    "fallback_triggered": result.fallback_triggered,
                    "final_model": result.final_model,
                    "total_latency": result.total_latency_seconds,
                    "validation_passed": result.validation_passed,
                    "validation_notes": result.validation_notes,
                },
                db_path=self.db_path,
            )
        except Exception:
            pass
