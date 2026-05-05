"""
Per-category validators.

Each validator is a callable with signature:
    (prompt: str, response: str, config: dict) -> (passed: bool, notes: str, validator_name: str)

The `validator_name` is a short identifier (e.g. "extraction.json_parse")
that lets logs disambiguate which check failed.
"""

from app.validation.validators.code import validate as validate_code
from app.validation.validators.extraction import validate as validate_extraction
from app.validation.validators.general import validate as validate_general
from app.validation.validators.reasoning import validate as validate_reasoning
from app.validation.validators.summarization import validate as validate_summarization

__all__ = [
    "validate_code",
    "validate_extraction",
    "validate_general",
    "validate_reasoning",
    "validate_summarization",
]
