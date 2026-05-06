"""
Validation config loader.

Layered: JSON file at `config/validation.json` (the source of truth)
overlaid with env-var overrides for the few knobs ops most often want
to flip without redeploying. Env vars use a fixed prefix so they don't
collide with anything else.

Supported env-var overrides:
    VALIDATION_CODE_EXEC_TIMEOUT_SECONDS   float
    VALIDATION_CODE_MEMORY_LIMIT_MB        int
    VALIDATION_CODE_EXECUTE_IN_SANDBOX     "true" or "false"
    VALIDATION_DEFAULT_ON_FAILURE          "retry", "fallback", "hard_error"

The config is re-read on every call (it's a tiny JSON file), so a config
change takes effect immediately without restarting the server.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional


CONFIG_PATH = Path("config/validation.json")


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _read_file(path: Path) -> dict:
    if not path.exists():
        return _DEFAULT_CONFIG.copy()
    with open(path) as f:
        return json.load(f)


def _apply_env_overrides(config: dict) -> dict:
    """
    Layer in env-var overrides. Conservative: we only honor a small set
    of keys, so a typo in an env var doesn't silently disable validation.
    """
    code_cfg = config.setdefault("categories", {}).setdefault("code", {})

    if (raw := os.environ.get("VALIDATION_CODE_EXEC_TIMEOUT_SECONDS")):
        try:
            code_cfg["exec_timeout_seconds"] = float(raw)
        except ValueError:
            pass

    if (raw := os.environ.get("VALIDATION_CODE_MEMORY_LIMIT_MB")):
        try:
            code_cfg["memory_limit_mb"] = int(raw)
        except ValueError:
            pass

    if (raw := os.environ.get("VALIDATION_CODE_EXECUTE_IN_SANDBOX")):
        code_cfg["execute_in_sandbox"] = _str_to_bool(raw)

    if (raw := os.environ.get("VALIDATION_DEFAULT_ON_FAILURE")):
        if raw in ("retry", "fallback", "hard_error"):
            for cat_cfg in config.get("categories", {}).values():
                cat_cfg["on_failure"] = raw

    return config


def load_config(path: Path = CONFIG_PATH) -> dict:
    """
    Load and return the layered config. Cheap; safe to call per request.
    """
    return _apply_env_overrides(_read_file(path))


def get_category_config(config: dict, category: str) -> dict:
    """
    Return the per-category config dict, falling back to `general` if the
    requested category isn't in the file.
    """
    categories = config.get("categories", {})
    return categories.get(category) or categories.get("general") or {}


def resolve_action(config: dict, category: str, endpoint: Optional[str]) -> str:
    """
    Return the recommended action when this category fails on this endpoint.

    Endpoint defaults can override the per-category action (so /generate
    can hard-error while /route falls back, even if both share the same
    category-level config).
    """
    category_action = get_category_config(config, category).get("on_failure", "fallback")

    if endpoint:
        endpoint_defaults = config.get("endpoint_defaults", {}).get(endpoint, {})
        override = endpoint_defaults.get("on_failure_override")
        if override:
            return override

    return category_action


# In-source default. Used when config/validation.json is missing (e.g.
# unit tests or a fresh checkout). Mirrors the JSON file's structure.
_DEFAULT_CONFIG: dict[str, Any] = {
    "categories": {
        "extraction": {"max_response_length": 8000, "require_json": True, "on_failure": "fallback"},
        "code": {
            "max_response_length": 8000,
            "require_syntactic_validity": True,
            "execute_in_sandbox": True,
            "exec_timeout_seconds": 5.0,
            "memory_limit_mb": 200,
            "on_failure": "fallback",
        },
        "summarization": {"min_words": 5, "max_words": 500, "forbid_code_fences": True, "on_failure": "fallback"},
        "reasoning": {"min_length": 1, "max_response_length": 4000, "on_failure": "fallback"},
        "general": {"min_length": 1, "max_response_length": 8000, "on_failure": "fallback"},
    },
    "endpoint_defaults": {
        "generate": {"on_failure_override": "hard_error"},
        "route": {"on_failure_override": None},
    },
}
