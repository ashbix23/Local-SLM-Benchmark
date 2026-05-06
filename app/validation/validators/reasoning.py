"""
Reasoning-bucket validator.

For reasoning we don't have a structural shape we can check (the answer
might be a number, a sentence, or a multi-paragraph explanation). The
runtime check is well-formedness only: non-empty, within length bounds,
and not a refusal-style empty response.
"""

from __future__ import annotations


# Phrases that indicate the model gave up rather than reasoned. These
# are heuristics, not a comprehensive list, deliberately conservative
# so we don't false-positive on legitimate hedging.
_REFUSAL_MARKERS = (
    "i cannot answer",
    "i can't answer",
    "i don't know how to",
    "as an ai language model, i",
)


def validate(prompt: str, response: str, config: dict) -> tuple[bool, str, str]:
    name = "reasoning.well_formed"

    min_length = int(config.get("min_length", 1))
    max_length = int(config.get("max_response_length", 4000))

    stripped = response.strip()

    if len(stripped) < min_length:
        return False, f"response too short ({len(stripped)} < {min_length} chars)", name

    if len(response) > max_length:
        return False, f"response too long ({len(response)} > {max_length} chars)", name

    lower = stripped.lower()
    for marker in _REFUSAL_MARKERS:
        if lower.startswith(marker):
            return False, f"response opens with refusal-style marker: {marker!r}", name

    return True, "non-empty, within bounds, not a refusal", name
