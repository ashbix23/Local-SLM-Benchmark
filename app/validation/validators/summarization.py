"""
Summarization-bucket validator.

A "summary" that's longer than the source is a failure mode we've seen
in practice (the model regurgitates the input). A summary that's empty
or that contains code fences is usually wrong shape. These are cheap to
check and catch the obvious failures without needing semantic scoring.
"""

from __future__ import annotations

import re


_CODE_FENCE = re.compile(r"```")


def _word_count(text: str) -> int:
    return len(text.split())


def validate(prompt: str, response: str, config: dict) -> tuple[bool, str, str]:
    name = "summarization.shape"

    min_words = int(config.get("min_words", 5))
    max_words = int(config.get("max_words", 500))
    forbid_fences = bool(config.get("forbid_code_fences", True))

    stripped = response.strip()
    if not stripped:
        return False, "empty response", name

    words = _word_count(stripped)
    if words < min_words:
        return False, f"too few words ({words} < {min_words})", name
    if words > max_words:
        return False, f"too many words ({words} > {max_words})", name

    if forbid_fences and _CODE_FENCE.search(response):
        return False, "summary contains a markdown code fence", name

    return True, f"shape ok ({words} words, no fences)", name
