"""
General-bucket validator.

The `general` category is what we route to when the classifier wasn't
confident or when the prompt didn't match any of the four real
categories. Validation here is intentionally minimal: non-empty, within
length bounds. We don't want to reject open-ended chit-chat for failing
some category-specific shape check.
"""

from __future__ import annotations


def validate(prompt: str, response: str, config: dict) -> tuple[bool, str, str]:
    name = "general.well_formed"

    min_length = int(config.get("min_length", 1))
    max_length = int(config.get("max_response_length", 8000))

    stripped = response.strip()

    if len(stripped) < min_length:
        return False, f"response too short ({len(stripped)} < {min_length} chars)", name

    if len(response) > max_length:
        return False, f"response too long ({len(response)} > {max_length} chars)", name

    return True, "non-empty and within length bounds", name
