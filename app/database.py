"""
SQLite persistence for benchmark runs.

Every (model × prompt) generation gets logged as a row. The benchmark
runner writes; the chart generator and analysis scripts read. Keeping
the data in SQLite (vs. JSON files or in-memory) means:

  - Re-running charts without re-running the benchmark (cheap iteration)
  - Easy ad-hoc queries ("which model had highest variance on reasoning?")
  - Historical comparison if we ever re-run on new hardware

Schema is intentionally denormalized — one flat `runs` table. At this
scale, normalization would add joins without meaningful benefit.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from app.models import BenchmarkRun


DB_PATH = Path("data/runs.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp   TEXT NOT NULL,
    model           TEXT NOT NULL,
    task_category   TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    response        TEXT NOT NULL,
    time_to_first_token  REAL NOT NULL,
    total_latency        REAL NOT NULL,
    tokens_generated     INTEGER NOT NULL,
    tokens_per_second    REAL NOT NULL,
    peak_memory_mb       REAL NOT NULL,
    quality_score        REAL,
    quality_notes        TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model);
CREATE INDEX IF NOT EXISTS idx_runs_category ON runs(task_category);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(run_timestamp);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at          TEXT NOT NULL,
    prompt_hash         TEXT NOT NULL,
    prompt_preview      TEXT NOT NULL,
    classified_category TEXT NOT NULL,
    classifier_confidence REAL NOT NULL,
    classifier_latency_ms REAL NOT NULL,
    chosen_model        TEXT NOT NULL,
    fallback_chain      TEXT NOT NULL,
    fallback_triggered  INTEGER NOT NULL,
    final_model         TEXT NOT NULL,
    total_latency       REAL NOT NULL,
    validation_passed   INTEGER NOT NULL,
    validation_notes    TEXT
);

CREATE INDEX IF NOT EXISTS idx_routing_decided_at ON routing_decisions(decided_at);
CREATE INDEX IF NOT EXISTS idx_routing_category ON routing_decisions(classified_category);
CREATE INDEX IF NOT EXISTS idx_routing_prompt_hash ON routing_decisions(prompt_hash);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the DB file and schema if they don't exist. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA)


@contextmanager
def get_connection(db_path: Path = DB_PATH):
    """
    Context-managed SQLite connection.

    sqlite3 doesn't auto-commit on context exit the way some expect —
    we do it explicitly so callers don't have to remember.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row  # dict-like row access
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def insert_run(run: BenchmarkRun, db_path: Path = DB_PATH) -> int:
    """Persist a single BenchmarkRun. Returns the new row's id."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs (
                run_timestamp, model, task_category, task_id,
                prompt, response,
                time_to_first_token, total_latency,
                tokens_generated, tokens_per_second,
                peak_memory_mb, quality_score, quality_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                run.model,
                run.task_category,
                run.task_id,
                run.prompt,
                run.response,
                run.time_to_first_token,
                run.total_latency,
                run.tokens_generated,
                run.tokens_per_second,
                run.peak_memory_mb,
                run.quality_score,
                run.quality_notes,
            ),
        )
        return cursor.lastrowid


def fetch_all_runs(db_path: Path = DB_PATH) -> list[dict]:
    """Return every run in the DB as a list of dicts."""
    with get_connection(db_path) as connection:
        rows = connection.execute("SELECT * FROM runs ORDER BY id").fetchall()
        return [dict(row) for row in rows]


def fetch_runs_by_model(model: str, db_path: Path = DB_PATH) -> list[dict]:
    """All runs for one model. Useful for per-model deep dives."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM runs WHERE model = ? ORDER BY id", (model,)
        ).fetchall()
        return [dict(row) for row in rows]


def fetch_latest_run_batch(db_path: Path = DB_PATH) -> list[dict]:
    """
    Return only the most recent benchmark batch.

    Charts should use the latest run, not mix historical data. We identify
    a "batch" as all runs sharing the same run_timestamp date+hour, since
    a full benchmark sweep takes under an hour.
    """
    with get_connection(db_path) as connection:
        latest_timestamp = connection.execute(
            "SELECT MAX(run_timestamp) FROM runs"
        ).fetchone()[0]

        if not latest_timestamp:
            return []

        # Grab everything within 1 hour of the latest run
        batch_prefix = latest_timestamp[:13]  # YYYY-MM-DDTHH
        rows = connection.execute(
            "SELECT * FROM runs WHERE run_timestamp LIKE ? ORDER BY id",
            (f"{batch_prefix}%",),
        ).fetchall()
        return [dict(row) for row in rows]


def clear_all_runs(db_path: Path = DB_PATH) -> int:
    """Wipe the runs table. Returns rows deleted. Use with caution."""
    with get_connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM runs")
        return cursor.rowcount


def insert_routing_decision(
    decision: dict,
    db_path: Path = DB_PATH,
) -> int:
    """
    Persist a routing decision row.

    `decision` keys must match the routing_decisions schema. We pass a dict
    rather than a Pydantic model because the routing module owns its own
    schema and we want database.py to stay decoupled from routing internals.
    """
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO routing_decisions (
                decided_at, prompt_hash, prompt_preview,
                classified_category, classifier_confidence, classifier_latency_ms,
                chosen_model, fallback_chain, fallback_triggered,
                final_model, total_latency,
                validation_passed, validation_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision["decided_at"],
                decision["prompt_hash"],
                decision["prompt_preview"],
                decision["classified_category"],
                decision["classifier_confidence"],
                decision["classifier_latency_ms"],
                decision["chosen_model"],
                decision["fallback_chain"],
                int(decision["fallback_triggered"]),
                decision["final_model"],
                decision["total_latency"],
                int(decision["validation_passed"]),
                decision.get("validation_notes"),
            ),
        )
        return cursor.lastrowid


def fetch_recent_routing_decisions(limit: int = 50, db_path: Path = DB_PATH) -> list[dict]:
    """Return most recent routing decisions, newest first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM routing_decisions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
