"""
Routing policy: which model wins for which category.

The policy reads from `runs.db` rather than hardcoding a mapping. When a
new benchmark batch lands, the router immediately reflects the new data
without code changes.

V1 selection rule: highest mean quality score per (model, category) over
the latest benchmark batch. Ties broken by lower mean latency.

Future hooks (not built, deliberately scoped out):
  - Pareto-aware selection (quality + latency joint front)
  - Configurable selection policies (`quality_only`, `cost_aware`, etc.)
The selection logic lives behind a single function (`derive_mapping`)
specifically so future policies can drop in without changing callers.

Fallback chain:
  For each category, the chain is the ranked list of models for that
  category, plus the global "safe default" (Qwen 7B) as the last resort.
  The router walks this chain when validation fails.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from app.database import DB_PATH, get_connection


# The five category labels the router accepts. The first four match the
# benchmark categories; `general` is the low-confidence fallback bucket.
ROUTING_CATEGORIES = ("reasoning", "summarization", "extraction", "code", "general")

# Safe default, used as the last item in any fallback chain, and as the
# pick for `general` when we have no benchmark data to consult.
SAFE_DEFAULT_MODEL = "qwen2.5:7b"


@dataclass
class CategoryStats:
    """Per-(model, category) aggregate from the latest benchmark batch."""
    model: str
    category: str
    mean_quality: float
    mean_latency: float
    sample_count: int


@dataclass
class RoutingPolicy:
    """
    Resolved policy at a moment in time.

    `mapping` is category → best model. `chains` is category → ordered
    fallback list (best → safe default). The Router consults both: the
    first entry of `chains[category]` is always the same as
    `mapping[category]`.
    """
    mapping: dict[str, str]
    chains: dict[str, list[str]]
    derived_from_rows: int  # how many benchmark rows fed the policy
    derivation_notes: str = ""

    def model_for(self, category: str) -> str:
        """Return the chosen model for a category, falling back to safe default."""
        return self.mapping.get(category, SAFE_DEFAULT_MODEL)

    def chain_for(self, category: str) -> list[str]:
        """Return the fallback chain for a category."""
        return self.chains.get(category, [SAFE_DEFAULT_MODEL])


def _load_latest_batch_stats(db_path: Path) -> list[CategoryStats]:
    """
    Aggregate the latest benchmark batch into per-(model, category) stats.

    A "batch" here matches the convention in database.fetch_latest_run_batch:
    runs sharing the same YYYY-MM-DDTHH prefix. Rows with no quality_score
    are skipped; they couldn't have informed the ranking.
    """
    with get_connection(db_path) as connection:
        latest_timestamp = connection.execute(
            "SELECT MAX(run_timestamp) FROM runs WHERE quality_score IS NOT NULL"
        ).fetchone()[0]

        if not latest_timestamp:
            return []

        batch_prefix = latest_timestamp[:13]
        rows = connection.execute(
            """
            SELECT model, task_category,
                   AVG(quality_score) AS mean_quality,
                   AVG(total_latency) AS mean_latency,
                   COUNT(*)            AS n
            FROM runs
            WHERE run_timestamp LIKE ?
              AND quality_score IS NOT NULL
            GROUP BY model, task_category
            """,
            (f"{batch_prefix}%",),
        ).fetchall()

    return [
        CategoryStats(
            model=row["model"],
            category=row["task_category"],
            mean_quality=row["mean_quality"],
            mean_latency=row["mean_latency"],
            sample_count=row["n"],
        )
        for row in rows
    ]


def derive_mapping(db_path: Path = DB_PATH) -> RoutingPolicy:
    """
    Build a fresh RoutingPolicy from whatever's in the DB right now.

    Called at server startup and (cheaply) on each request; it's a single
    aggregation query. If the DB is empty (no benchmark has been run yet),
    every category routes to SAFE_DEFAULT_MODEL.
    """
    stats = _load_latest_batch_stats(db_path)

    if not stats:
        # No benchmark data; everything goes to the safe default.
        mapping = {cat: SAFE_DEFAULT_MODEL for cat in ROUTING_CATEGORIES}
        chains = {cat: [SAFE_DEFAULT_MODEL] for cat in ROUTING_CATEGORIES}
        return RoutingPolicy(
            mapping=mapping,
            chains=chains,
            derived_from_rows=0,
            derivation_notes="No benchmark data found; defaulted to qwen2.5:7b for all categories.",
        )

    # Bucket stats by category, sort by quality desc then latency asc.
    by_category: dict[str, list[CategoryStats]] = {}
    for entry in stats:
        by_category.setdefault(entry.category, []).append(entry)
    for category, entries in by_category.items():
        entries.sort(key=lambda s: (-s.mean_quality, s.mean_latency))

    mapping: dict[str, str] = {}
    chains: dict[str, list[str]] = {}

    for category in ("reasoning", "summarization", "extraction", "code"):
        ranked = by_category.get(category, [])
        if not ranked:
            mapping[category] = SAFE_DEFAULT_MODEL
            chains[category] = [SAFE_DEFAULT_MODEL]
            continue

        ranked_models = [s.model for s in ranked]
        mapping[category] = ranked_models[0]
        # Fallback chain: ranked models for this category, then the safe
        # default if it isn't already there.
        chain = list(ranked_models)
        if SAFE_DEFAULT_MODEL not in chain:
            chain.append(SAFE_DEFAULT_MODEL)
        chains[category] = chain

    # `general` always routes to the safe default, by design, since it's
    # the bucket for "we're not confident what this is."
    mapping["general"] = SAFE_DEFAULT_MODEL
    chains["general"] = [SAFE_DEFAULT_MODEL]

    return RoutingPolicy(
        mapping=mapping,
        chains=chains,
        derived_from_rows=sum(s.sample_count for s in stats),
        derivation_notes=(
            f"Derived from {len(stats)} (model, category) groups in latest batch."
        ),
    )
