"""
End-to-end benchmark runner.

Sweeps all configured models against the full prompt suite, scores every
generation, and writes results to SQLite. Designed for 8 GB M2 Air — runs
models strictly sequentially with explicit unload between sweeps so we
never have two models resident at once.

Usage:
    python scripts/run_benchmark.py

Preconditions:
    - ollama serve (or Ollama.app) running
    - gemma2:2b, llama3.2:3b, qwen2.5:7b pulled
    - ANTHROPIC_API_KEY in .env (for the LLM judge on reasoning/summarization)

Total runtime: ~15-25 min on M2 Air with browser/apps closed.
"""

import asyncio
import sys
import time
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Make the app/ package importable when running this file directly
sys.path.insert(0, ".")

from app.database import init_db, fetch_latest_run_batch, clear_all_runs
from app.benchmark import BenchmarkRunner
from app.prompts import ALL_PROMPTS, BenchmarkPrompt
from app.scoring import score_run
from app.ollama_client import OLLAMA_BASE_URL


MODELS_TO_BENCHMARK = [
    "gemma2:2b",
    "llama3.2:3b",
    "qwen2.5:7b",
]


console = Console()


def _build_prompt_text(prompt: BenchmarkPrompt) -> str:
    """Materialize the prompt string, substituting source text for summarization."""
    if prompt.source_text:
        return prompt.prompt.format(source=prompt.source_text)
    return prompt.prompt


async def unload_model(model: str) -> None:
    """
    Force Ollama to unload a model immediately.

    Ollama's default keep_alive is 5 minutes. Passing keep_alive=0 on a
    generate call triggers immediate unload — we send a near-empty
    generate request solely for this side effect.
    """
    payload = {"model": model, "prompt": "", "keep_alive": 0, "stream": False}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        except Exception:
            # Unload failure is non-fatal — Ollama will evict eventually
            pass


async def run_model_sweep(
    runner: BenchmarkRunner,
    model: str,
    prompts: list[BenchmarkPrompt],
    progress: Progress,
    task_id,
) -> tuple[int, int]:
    """
    Run all prompts against one model. Returns (successes, failures).

    Errors on individual prompts are logged but don't abort the sweep —
    we'd rather have 13/14 data points for this model than zero.
    """
    successes = 0
    failures = 0

    for prompt in prompts:
        prompt_text = _build_prompt_text(prompt)
        try:
            run = await runner.run_single(
                model=model,
                task_category=prompt.category,
                task_id=prompt.task_id,
                prompt=prompt_text,
            )
            score_run(run, prompt)
            successes += 1
        except Exception as exc:
            console.print(
                f"  [red]✗ {prompt.task_id}:[/red] {type(exc).__name__}: {exc}"
            )
            failures += 1

        progress.update(task_id, advance=1)

    return successes, failures


def print_summary() -> None:
    """Print a per-model summary table from the latest run batch."""
    rows = fetch_latest_run_batch()
    if not rows:
        console.print("[yellow]No runs to summarize.[/yellow]")
        return

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    table = Table(title="Benchmark Summary — Latest Sweep")
    table.add_column("Model", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Avg Latency (s)", justify="right")
    table.add_column("Avg Tok/sec", justify="right")
    table.add_column("Memory (MB)", justify="right")

    for model, model_rows in by_model.items():
        scored = [r for r in model_rows if r["quality_score"] is not None]
        avg_score = sum(r["quality_score"] for r in scored) / len(scored) if scored else 0.0
        avg_latency = sum(r["total_latency"] for r in model_rows) / len(model_rows)
        avg_throughput = sum(r["tokens_per_second"] for r in model_rows) / len(model_rows)
        peak_memory = max(r["peak_memory_mb"] for r in model_rows)

        table.add_row(
            model,
            str(len(model_rows)),
            f"{avg_score:.2f}",
            f"{avg_latency:.2f}",
            f"{avg_throughput:.1f}",
            f"{peak_memory:.0f}",
        )

    console.print(table)


async def main() -> None:
    console.rule("[bold cyan]Local SLM Benchmark Suite[/bold cyan]")
    console.print(f"Models: {', '.join(MODELS_TO_BENCHMARK)}")
    console.print(f"Prompts: {len(ALL_PROMPTS)} across 4 categories")
    console.print()

    init_db()

    # Optional reset — ask so re-running doesn't silently append to old data
    response = console.input("[yellow]Clear existing runs from DB? (y/N): [/yellow]")
    if response.strip().lower() == "y":
        deleted = clear_all_runs()
        console.print(f"[green]Cleared {deleted} previous runs.[/green]")
    console.print()

    runner = BenchmarkRunner()
    overall_start = time.perf_counter()

    for index, model in enumerate(MODELS_TO_BENCHMARK, start=1):
        console.rule(f"[bold]Model {index}/{len(MODELS_TO_BENCHMARK)}: {model}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {model}", total=len(ALL_PROMPTS))
            model_start = time.perf_counter()
            successes, failures = await run_model_sweep(
                runner, model, ALL_PROMPTS, progress, task
            )
            elapsed = time.perf_counter() - model_start

        console.print(
            f"  [green]{successes} succeeded[/green], "
            f"[red]{failures} failed[/red] in {elapsed:.0f}s"
        )

        # Unload before the next model so VRAM is free
        if index < len(MODELS_TO_BENCHMARK):
            console.print(f"  Unloading {model}...")
            await unload_model(model)
            console.print()

    total_elapsed = time.perf_counter() - overall_start
    console.rule(f"[bold green]Sweep complete in {total_elapsed:.0f}s[/bold green]")
    console.print()
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
