"""
Scoring for structured-extraction prompts.

Two-stage scoring:
  1. STRUCTURAL — does the model's output conform to the target Pydantic
     schema? Instructor handles this by prompting the model in a schema-
     aware way and retrying on validation failures. We capture whether it
     succeeded and how many retries it needed.
  2. FIELD — are the extracted field values actually correct? We use
     fuzzy matchers (contains, None-check, list-length-min) because exact
     equality is too brittle for natural-language extraction — "Dr. Rajesh
     Patel" and "Rajesh Patel" should both count as correct names.

Why Instructor and not raw JSON-mode prompting:
  Instructor is the production pattern for this. It (a) reformats the
  prompt to include the schema, (b) parses and validates the response
  against Pydantic, (c) feeds validation errors back to the model for a
  retry if the first attempt fails. For this benchmark it gives us a
  clean data point: did this model handle structured extraction at all?
"""

from typing import Any
import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from app.models import PersonInfo, MeetingExtraction


# Schema registry — maps the `schema_name` string on BenchmarkPrompt to
# the actual Pydantic class. Extend here if you add new extraction targets.
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "PersonInfo": PersonInfo,
    "MeetingExtraction": MeetingExtraction,
}


OLLAMA_OPENAI_COMPAT_URL = "http://localhost:11434/v1"


def _make_instructor_client() -> instructor.Instructor:
    """
    Instructor talks to Ollama via its OpenAI-compatible endpoint.

    Ollama exposes /v1/chat/completions that speaks the OpenAI API dialect,
    so we point the openai client at localhost:11434/v1 and wrap it with
    instructor.from_openai. No actual OpenAI API key is used — we pass a
    dummy string because the openai client requires one to initialize.
    """
    openai_client = OpenAI(
        base_url=OLLAMA_OPENAI_COMPAT_URL,
        api_key="ollama",  # dummy — Ollama ignores auth
    )
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


def _score_person_info(extracted: PersonInfo, expected: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Fuzzy-match a PersonInfo extraction against expectations.

    Supported expected-field keys:
      - name / name_contains
      - age (exact, including None to test "don't hallucinate")
      - occupation_contains
      - location_contains
    """
    checks: list[tuple[bool, str]] = []

    if "name" in expected:
        checks.append((
            extracted.name == expected["name"],
            f"name exact: got {extracted.name!r}, expected {expected['name']!r}",
        ))
    if "name_contains" in expected:
        checks.append((
            expected["name_contains"].lower() in extracted.name.lower(),
            f"name contains {expected['name_contains']!r}: got {extracted.name!r}",
        ))
    if "age" in expected:
        checks.append((
            extracted.age == expected["age"],
            f"age: got {extracted.age!r}, expected {expected['age']!r}",
        ))
    if "occupation_contains" in expected:
        got = (extracted.occupation or "").lower()
        checks.append((
            expected["occupation_contains"].lower() in got,
            f"occupation contains {expected['occupation_contains']!r}: got {extracted.occupation!r}",
        ))
    if "location_contains" in expected:
        got = (extracted.location or "").lower()
        checks.append((
            expected["location_contains"].lower() in got,
            f"location contains {expected['location_contains']!r}: got {extracted.location!r}",
        ))

    passed = sum(1 for ok, _ in checks if ok)
    total = len(checks)
    score = passed / total if total > 0 else 0.0
    failures = [msg for ok, msg in checks if not ok]
    return score, failures


def _score_meeting_extraction(extracted: MeetingExtraction, expected: dict[str, Any]) -> tuple[float, list[str]]:
    """
    Fuzzy-match a MeetingExtraction against expectations.

    Supported expected-field keys:
      - meeting_topic_contains
      - action_item_count_min
      - has_high_priority  (at least one action with priority='high')
      - next_meeting_date_present
    """
    checks: list[tuple[bool, str]] = []

    if "meeting_topic_contains" in expected:
        got = extracted.meeting_topic.lower()
        checks.append((
            expected["meeting_topic_contains"].lower() in got,
            f"meeting_topic contains {expected['meeting_topic_contains']!r}: got {extracted.meeting_topic!r}",
        ))
    if "action_item_count_min" in expected:
        count = len(extracted.action_items)
        checks.append((
            count >= expected["action_item_count_min"],
            f"action_item_count >= {expected['action_item_count_min']}: got {count}",
        ))
    if "has_high_priority" in expected:
        has_high = any(item.priority == "high" for item in extracted.action_items)
        checks.append((
            has_high == expected["has_high_priority"],
            f"has_high_priority: expected {expected['has_high_priority']}, got {has_high}",
        ))
    if "next_meeting_date_present" in expected:
        present = extracted.next_meeting_date is not None
        checks.append((
            present == expected["next_meeting_date_present"],
            f"next_meeting_date_present: expected {expected['next_meeting_date_present']}, got {present}",
        ))

    passed = sum(1 for ok, _ in checks if ok)
    total = len(checks)
    score = passed / total if total > 0 else 0.0
    failures = [msg for ok, msg in checks if not ok]
    return score, failures


def score_extraction(
    model: str,
    prompt_text: str,
    schema_name: str,
    expected_fields: dict[str, Any],
) -> tuple[float, str]:
    """
    Run Instructor-guided extraction and score the result.

    Unlike the other scorers, this one re-calls the model — Instructor
    needs to own the request/response loop to handle schema injection and
    retries. That's fine: the benchmark runner still logs TTFT/latency/
    tokens from the original free-form call; this step measures a
    separate, scoring-specific dimension (can it produce structured
    output at all?).

    Returns (score in [0.0, 1.0], human-readable notes).
    """
    schema_class = SCHEMA_REGISTRY.get(schema_name)
    if schema_class is None:
        return 0.0, f"Unknown schema: {schema_name}"

    client = _make_instructor_client()

    # Instructor returns the validated Pydantic object on success, or raises
    # on final failure after retries exhausted.
    try:
        extracted = client.chat.completions.create(
            model=model,
            response_model=schema_class,
            max_retries=2,
            messages=[{"role": "user", "content": prompt_text}],
        )
    except ValidationError as exc:
        return 0.0, f"Pydantic validation failed after retries: {exc.errors()[:2]}"
    except Exception as exc:
        return 0.0, f"Instructor call failed: {type(exc).__name__}: {str(exc)[:200]}"

    # Structural success — score the fields
    if isinstance(extracted, PersonInfo):
        field_score, failures = _score_person_info(extracted, expected_fields)
    elif isinstance(extracted, MeetingExtraction):
        field_score, failures = _score_meeting_extraction(extracted, expected_fields)
    else:
        return 0.0, f"No field scorer for schema {schema_name}"

    # Blend: half the score is "produced valid schema," half is "fields correct"
    final_score = 0.5 + 0.5 * field_score
    notes_parts = [f"Schema valid. Field score: {field_score:.2f}"]
    if failures:
        notes_parts.append("Issues: " + "; ".join(failures[:3]))
    return final_score, " | ".join(notes_parts)
