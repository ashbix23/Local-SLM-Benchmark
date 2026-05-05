"""
Benchmark-driven request router.

Takes an inbound prompt, classifies it into one of the four benchmark
categories (or `general`), looks up the benchmark-best model for that
category from SQLite, and dispatches generation. On validation failure,
walks a configurable fallback chain.

Submodules:
    policy:      reads runs.db, returns best-model-per-category mapping
    classifier:  zero-shot prompt classifier (Gemma 2B + Instructor)
    router:      orchestrates classify → select → generate → validate
"""

from app.routing.router import Router, RoutingResult, ValidationHook
from app.routing.policy import RoutingPolicy
from app.routing.classifier import PromptClassifier, Classification

__all__ = [
    "Router",
    "RoutingResult",
    "ValidationHook",
    "RoutingPolicy",
    "PromptClassifier",
    "Classification",
]
