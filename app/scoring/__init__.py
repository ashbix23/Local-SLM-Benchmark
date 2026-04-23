"""
Scoring dispatcher.

Routes a BenchmarkRun to the right scorer based on the prompt's category,
then updates the run's quality_score and quality_notes in the database.

Keeping a single public entry point (score_run) means the runner script
doesn't need to know about individual scorers — it just calls score_run
for every completed generation.
"""

import sqlite3
from pathlib import Path
from app.models import BenchmarkRun
from app.prompts import BenchmarkPrompt
from app.database import DB_PATH, get_connection
from app.scoring.code import score_code
from app.scoring.extraction import score_extraction
from app.scoring.judge import score_reasoning, score_summarization


def score_run(
    run: BenchmarkRun,
    prompt: BenchmarkPrompt,
    db_path: Path = DB_PATH,
) -> BenchmarkRun:
    """
    Score a completed run and persist the result back to the database.

    Mutates `run` in place (sets quality_score and quality_notes) and
    returns it for caller convenience.
    """
    category = prompt.category

    if category == "reasoning":
        score, notes = score_reasoning(
            prompt=prompt.prompt,
            expected_answer=prompt.expected_answer or "",
            response=run.response,
        )

    elif category == "summarization":
        score, notes = score_summarization(
            prompt=prompt.prompt,
            source_text=prompt.source_text or "",
            response=run.response,
        )

    elif category == "extraction":
        score, notes = score_extraction(
            model=run.model,
            prompt_text=prompt.prompt,
            schema_name=prompt.schema_name or "",
            expected_fields=prompt.expected_fields,
        )

    elif category == "code":
        score, notes = score_code(
            response=run.response,
            prompt_text=prompt.prompt,
            test_cases=prompt.test_cases,
        )

    else:
        score, notes = 0.0, f"No scorer for category: {category}"

    run.quality_score = score
    run.quality_notes = notes

    # Persist the updated score/notes for this run. We match on (model,
    # task_id, latest row) — there's one row per (model, task_id) per
    # sweep, and we want to update the most recent.
    _update_score_in_db(run, db_path)

    return run


def _update_score_in_db(run: BenchmarkRun, db_path: Path) -> None:
    """Update the quality fields on the most recent matching row."""
    with get_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE runs
            SET quality_score = ?, quality_notes = ?
            WHERE id = (
                SELECT id FROM runs
                WHERE model = ? AND task_id = ?
                ORDER BY id DESC
                LIMIT 1
            )
            """,
            (run.quality_score, run.quality_notes, run.model, run.task_id),
        )
