"""
Inspect benchmark runs.

Read-only browser over the SQLite database. Use after a benchmark sweep
to look at what the models actually said — the qualitative material you
need for BENCHMARKS.md (specific failure examples, surprise wins, judge
rationales).

Usage:
    python scripts/inspect_runs.py                       # everything, latest batch
    python scripts/inspect_runs.py --model llama3.2:3b   # filter by model
    python scripts/inspect_runs.py --category reasoning  # filter by category
    python scripts/inspect_runs.py --failures            # only runs with score < 1.0
    python scripts/inspect_runs.py --task reasoning_03   # one specific prompt across all models

Combine flags freely:
    python scripts/inspect_runs.py --model llama3.2:3b --category reasoning --failures
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, ".")

from app.database import fetch_latest_run_batch


console = Console()


MODEL_DISPLAY = {
    "gemma2:2b": "Gemma 2 (2B)",
    "llama3.2:3b": "Llama 3.2 (3B)",
    "qwen2.5:7b": "Qwen 2.5 (7B)",
}


def _truncate(text: str, max_chars: int = 800) -> str:
    """Trim very long responses for display, preserving the start."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, full length {len(text)} chars]"


def _score_color(score: float | None) -> str:
    """Color-code scores by quality tier for quick visual scanning."""
    if score is None:
        return "dim"
    if score >= 0.85:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _render_run(row: dict) -> None:
    """Render a single run as a rich Panel — readable in terminal."""
    score = row["quality_score"]
    score_text = f"{score:.2f}" if score is not None else "—"
    color = _score_color(score)

    header = (
        f"[bold cyan]{MODEL_DISPLAY.get(row['model'], row['model'])}[/bold cyan]"
        f"  •  [yellow]{row['task_category']}/{row['task_id']}[/yellow]"
        f"  •  Score: [bold {color}]{score_text}[/bold {color}]"
    )

    body = Text()
    body.append("PROMPT:\n", style="bold")
    body.append(_truncate(row["prompt"], max_chars=400) + "\n\n", style="dim")
    body.append("RESPONSE:\n", style="bold")
    body.append(_truncate(row["response"]) + "\n\n")
    body.append("METRICS:\n", style="bold")
    body.append(
        f"  TTFT {row['time_to_first_token']:.2f}s  •  "
        f"Total {row['total_latency']:.2f}s  •  "
        f"{row['tokens_generated']} tokens at "
        f"{row['tokens_per_second']:.1f} tok/s\n",
        style="dim",
    )
    if row["quality_notes"]:
        body.append("\nJUDGE NOTES:\n", style="bold")
        body.append(row["quality_notes"], style="italic")

    console.print(Panel(body, title=header, border_style=color, expand=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect benchmark runs from the latest sweep.")
    parser.add_argument("--model", help="Filter by model (e.g., gemma2:2b)")
    parser.add_argument("--category", help="Filter by category (reasoning, summarization, extraction, code)")
    parser.add_argument("--task", help="Filter by exact task_id (e.g., reasoning_03)")
    parser.add_argument("--failures", action="store_true",
                        help="Only show runs with quality_score < 1.0")
    args = parser.parse_args()

    rows = fetch_latest_run_batch()
    if not rows:
        console.print("[red]No runs in DB. Run scripts/run_benchmark.py first.[/red]")
        return

    # Apply filters in order — short-circuit empty results with a useful message
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]
    if args.category:
        rows = [r for r in rows if r["task_category"] == args.category]
    if args.task:
        rows = [r for r in rows if r["task_id"] == args.task]
    if args.failures:
        rows = [r for r in rows if r["quality_score"] is not None and r["quality_score"] < 1.0]

    if not rows:
        console.print("[yellow]No runs match those filters.[/yellow]")
        return

    # Sort: category, task_id, model — so related runs cluster together
    rows.sort(key=lambda r: (r["task_category"], r["task_id"], r["model"]))

    console.rule(f"[bold]Showing {len(rows)} runs[/bold]")
    for row in rows:
        _render_run(row)
        console.print()  # blank line between panels

    console.rule(f"[dim]End — {len(rows)} runs displayed[/dim]")


if __name__ == "__main__":
    main()
