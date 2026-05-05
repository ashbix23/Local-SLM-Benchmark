"""
Extraction-bucket validator.

The model's job for extraction is "return structured data" — usually
JSON. The runtime check is JSON-validity: strip any markdown fences the
model wrapped its output in, then parse. Pydantic schema match is
intentionally out of scope for v1 because /generate and /route don't
carry a schema name on the request; that's a benchmark-time concept.
We can wire schema match in a follow-up if production callers start
passing one.

Why JSON and not "any structured form": the benchmark's extraction task
already operates in JSON via Instructor. Aligning the runtime validator
with the benchmark target keeps the two layers consistent.
"""

from __future__ import annotations

import json
import re


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Pull JSON out of a markdown code block if the model wrapped it."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate(prompt: str, response: str, config: dict) -> tuple[bool, str, str]:
    name = "extraction.json_parse"

    max_length = int(config.get("max_response_length", 8000))
    require_json = bool(config.get("require_json", True))

    if len(response) > max_length:
        return False, f"response too long ({len(response)} > {max_length} chars)", name

    if not response.strip():
        return False, "empty response", name

    if not require_json:
        return True, "json-check disabled by config; non-empty pass", name

    candidate = _strip_fences(response)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return False, f"not valid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}", name

    if not isinstance(parsed, (dict, list)):
        return False, f"JSON must be object or array; got {type(parsed).__name__}", name

    return True, "valid JSON, parsable", name
