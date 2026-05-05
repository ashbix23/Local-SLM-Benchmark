"""
Prompt classifier for the router.

Maps an inbound prompt to one of the five routing categories. Two layers:

1. CHEAP HEURISTIC — regex/keyword pre-pass. Catches obvious cases (a
   prompt with ``def`` or ```` ``` ```` is almost certainly code, a prompt
   with "summarize" is almost certainly summarization). Heuristic hits
   short-circuit the LLM call entirely, which keeps p95 latency in check
   and saves Ollama load.

2. LLM ZERO-SHOT — Gemma 2B via Instructor, returning a small Pydantic
   schema (category + confidence). Used only when the heuristic abstains.

The heuristic is intentionally conservative: it returns confidently or
not at all. Anything ambiguous goes to the LLM. That keeps the cheap
path from biasing classification toward whatever keywords we picked.

Why Gemma 2B and not Qwen 7B for classification:
  Per the benchmark, classification is a discriminative task — it's about
  putting prompts into the right bucket, not generating high-quality
  content. The 7B's quality advantage doesn't help here, and using it
  would add 5+ seconds to every request. Gemma 2B's TTFT (under a second
  warm) is a much better fit for an inline routing decision.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


CLASSIFIER_MODEL = "gemma2:2b"
OLLAMA_OPENAI_COMPAT_URL = "http://localhost:11434/v1"

# Below this confidence we route to `general` — the safe default bucket.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


# =============================================================================
# Heuristic pre-pass
# =============================================================================
# Each heuristic is a (compiled_regex, category, confidence) tuple. We hand-
# tuned confidences down from 1.0 because keyword hits are signals, not
# certainties — a prompt that says "summarize this code" should still go
# to the LLM rather than getting stuck on the first matching bucket.

_CODE_PATTERNS = [
    re.compile(r"```"),                                         # fenced code
    re.compile(r"\bdef\s+\w+\s*\(", re.IGNORECASE),
    re.compile(r"\bclass\s+\w+\s*[:(]"),
    re.compile(r"\b(write|implement|fix)\s+(a\s+)?(function|method|program)", re.IGNORECASE),
    re.compile(r"\b(python|javascript|typescript|rust|go|java)\b\s+(function|code|snippet)", re.IGNORECASE),
]

_SUMMARIZATION_PATTERNS = [
    re.compile(r"\bsummari[sz]e\b", re.IGNORECASE),
    re.compile(r"\btl;?dr\b", re.IGNORECASE),
    re.compile(r"\bin\s+\d+\s+(sentences?|words?|bullets?)\b", re.IGNORECASE),
    re.compile(r"\bcondense\b", re.IGNORECASE),
]

_EXTRACTION_PATTERNS = [
    re.compile(r"\bextract\b", re.IGNORECASE),
    re.compile(r"\bparse\b.*\b(into|as)\b", re.IGNORECASE),
    re.compile(r"\b(json|schema|structured|fields?)\b.*\b(from|out\s+of)\b", re.IGNORECASE),
]

_REASONING_PATTERNS = [
    re.compile(r"\b(how\s+many|how\s+much)\b", re.IGNORECASE),
    re.compile(r"\bif\b.*\b(then|how|what|when)\b", re.IGNORECASE),
    re.compile(r"\b(solve|calculate|compute)\b", re.IGNORECASE),
    re.compile(r"\b(puzzle|riddle)\b", re.IGNORECASE),
]


def _heuristic_classify(prompt: str) -> Optional[tuple[str, float]]:
    """
    Quick pre-pass. Returns (category, confidence) when at least two
    patterns hit for a single category — that's enough signal to skip the
    LLM. Single-pattern hits are returned with lower confidence and only
    if no other category has competing hits.
    """
    counts = {
        "code": sum(1 for p in _CODE_PATTERNS if p.search(prompt)),
        "summarization": sum(1 for p in _SUMMARIZATION_PATTERNS if p.search(prompt)),
        "extraction": sum(1 for p in _EXTRACTION_PATTERNS if p.search(prompt)),
        "reasoning": sum(1 for p in _REASONING_PATTERNS if p.search(prompt)),
    }
    nonzero = [(cat, n) for cat, n in counts.items() if n > 0]
    if not nonzero:
        return None

    nonzero.sort(key=lambda kv: -kv[1])
    top_cat, top_n = nonzero[0]

    # Strong signal: 2+ hits in one category and nothing tied at the top.
    if top_n >= 2 and (len(nonzero) == 1 or nonzero[1][1] < top_n):
        return top_cat, 0.85

    # Weak signal: a single hit, no competing categories. Return it but
    # with low confidence so the threshold in the router can downshift to
    # `general` if we want to be cautious.
    if top_n == 1 and len(nonzero) == 1:
        return top_cat, 0.55

    # Mixed signals — defer to the LLM.
    return None


# =============================================================================
# LLM zero-shot fallback
# =============================================================================

class _LLMClassification(BaseModel):
    """Schema Instructor enforces on the classifier model's output."""
    category: str = Field(
        ...,
        description=(
            "One of: reasoning, summarization, extraction, code, general. "
            "Use 'general' only when the prompt clearly fits none of the others."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0–1.0, your confidence in this classification.",
    )


_CLASSIFIER_SYSTEM = """You are a prompt classifier. Read the user's prompt and decide which task category it belongs to.

Categories:
- reasoning: math, logic puzzles, multi-step reasoning, common-sense traps.
- summarization: condense a passage into fewer words, sentences, or bullets.
- extraction: pull structured fields (names, dates, amounts) out of unstructured text.
- code: write, fix, or explain code.
- general: the prompt clearly fits none of the above.

Respond with the category name and a confidence score from 0.0 to 1.0. Be honest about uncertainty — if the prompt is ambiguous, return 'general' with a moderate confidence."""


@dataclass
class Classification:
    """Result of classifying a single prompt."""
    category: str
    confidence: float
    method: str          # "heuristic" or "llm"
    latency_ms: float


class PromptClassifier:
    """
    Two-layer classifier (heuristic → LLM).

    The LLM call uses Instructor for schema enforcement so we get a clean
    Pydantic object even when Gemma decides to add prose around the JSON.
    """

    def __init__(
        self,
        model: str = CLASSIFIER_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._client: Optional[instructor.Instructor] = None

    def _get_client(self) -> instructor.Instructor:
        if self._client is None:
            openai_client = OpenAI(
                base_url=OLLAMA_OPENAI_COMPAT_URL,
                api_key="ollama",
                timeout=20.0,
            )
            self._client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)
        return self._client

    def classify(self, prompt: str) -> Classification:
        """
        Return a Classification. Always returns — never raises. On any
        LLM failure, falls back to ('general', 0.0, 'llm-failed').
        """
        start = time.perf_counter()

        heuristic = _heuristic_classify(prompt)
        if heuristic is not None:
            category, confidence = heuristic
            return Classification(
                category=category,
                confidence=confidence,
                method="heuristic",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        try:
            result = self._get_client().chat.completions.create(
                model=self.model,
                response_model=_LLMClassification,
                max_retries=1,
                messages=[
                    {"role": "system", "content": _CLASSIFIER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            return Classification(
                category="general",
                confidence=0.0,
                method="llm-failed",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        category = result.category.strip().lower()
        if category not in ("reasoning", "summarization", "extraction", "code", "general"):
            category = "general"

        confidence = float(result.confidence)
        # Apply the threshold here so callers see a uniform contract: any
        # below-threshold classification gets demoted to `general`.
        if confidence < self.confidence_threshold and category != "general":
            category = "general"

        return Classification(
            category=category,
            confidence=confidence,
            method="llm",
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
