"""
Routing acceptance test.

Runs a curated 20-prompt eval set through the classifier and (optionally)
the full router, and reports:

  - Per-prompt: expected category, predicted category, method, confidence
  - Aggregate: classification accuracy and p95 classifier latency
  - Acceptance: pass/fail against the 85% accuracy and 500ms p95 thresholds
    set in the routing ticket

Run modes:
  python scripts/eval_router.py             # classifier-only (fast, default)
  python scripts/eval_router.py --full      # classifier + full route() call
                                            # (slower; exercises generation)
  python scripts/eval_router.py --force-fb  # forced-fallback smoke test on
                                            # one prompt to verify the chain
                                            # walks correctly

The eval set is intentionally a mix: obvious cases that the heuristic
should catch, plus borderline cases that exercise the LLM fallback.
"""

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

# Make the app/ package importable when running this file directly
sys.path.insert(0, ".")

from app.ollama_client import OllamaClient
from app.routing import Router
from app.routing.classifier import PromptClassifier


@dataclass
class EvalCase:
    expected: str
    prompt: str
    notes: str = ""


# 20 prompts: 5 reasoning, 4 summarization, 4 extraction, 4 code, 3 general.
# Counts skew toward the four real categories because that's where the
# acceptance criterion lives ("on a curated set of 20 test prompts spanning
# all four categories").
EVAL_SET: list[EvalCase] = [
    # --- Reasoning (5) ---
    EvalCase("reasoning", "If a train leaves at 2pm going 50mph and another at 3pm going 70mph from 200 miles away, when do they meet?", "math word problem"),
    EvalCase("reasoning", "A farmer has 17 sheep. All but 9 die. How many are left?", "trick question"),
    EvalCase("reasoning", "There are three switches outside a room with one bulb. You can flip switches as often as you want, but you can only enter the room once. How do you tell which switch controls the bulb?", "classic puzzle"),
    EvalCase("reasoning", "I have 5 apples, eat 2, then buy 3 more. How many do I have?", "simple arithmetic"),
    EvalCase("reasoning", "Alice is older than Bob. Bob is older than Carol. Is Alice older than Carol?", "transitivity"),

    # --- Summarization (4) ---
    EvalCase("summarization", "Summarize the following passage in two sentences: Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and supports multiple programming paradigms.", "explicit 'summarize'"),
    EvalCase("summarization", "Give me a one-paragraph TL;DR of this article: The transformer architecture has become the dominant approach in NLP since 2017...", "TL;DR variant"),
    EvalCase("summarization", "In 3 bullet points, condense the key takeaways from this meeting transcript: [transcript content here about Q3 planning]", "bullets variant"),
    EvalCase("summarization", "Read this and tell me the main idea in under 50 words. The Industrial Revolution transformed manufacturing through mechanization, beginning in Britain in the late 18th century and spreading globally over the next hundred years.", "implicit summarization via length cap"),

    # --- Extraction (4) ---
    EvalCase("extraction", "Extract the person's name, age, and city from: Dr. Rajesh Patel, 47, lives in Mumbai and works as a cardiologist.", "explicit extract"),
    EvalCase("extraction", "Pull the action items, owners, and deadlines from this meeting note as JSON: Sarah will draft the proposal by Friday. Mike owns testing.", "structured output keyword"),
    EvalCase("extraction", "Parse the following invoice into structured fields (date, amount, vendor): Invoice #4421, $1,250.00 from Acme Corp dated March 14, 2026.", "parse + structured"),
    EvalCase("extraction", "Get the company name, ticker, and headquarters from: Acme Corporation (NYSE: ACME) is headquartered in San Francisco.", "no 'extract' keyword but extraction-shaped"),

    # --- Code (4) ---
    EvalCase("code", "Write a Python function called is_palindrome(s) that returns True if s reads the same forwards and backwards.", "explicit code request"),
    EvalCase("code", "Implement two_sum in JavaScript: given an array of integers and a target, return indices of two numbers that add up to the target.", "implement keyword"),
    EvalCase("code", "Fix this code:\n```python\ndef add(a, b)\n  return a + b\n```", "fenced code block"),
    EvalCase("code", "Write a function to find the longest substring without repeating characters.", "no language specified"),

    # --- General (3) ---
    EvalCase("general", "Hi, how are you today?", "small talk"),
    EvalCase("general", "What's the weather like in Paris?", "factual; not really any of the four"),
    EvalCase("general", "Tell me an interesting fact about octopuses.", "open-ended creative"),
]


P95_BUDGET_MS = 500.0
ACCURACY_TARGET = 0.85


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = int(round((p / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[k]


async def run_classifier_only(console: Console, heuristic_only: bool = False) -> int:
    classifier = PromptClassifier(heuristic_only=heuristic_only)
    mode = "heuristic-only" if heuristic_only else "heuristic + Gemma 2B fallback"
    console.print(f"[bold]Classifying {len(EVAL_SET)} prompts ({mode})...[/bold]\n")
    console.print(
        f"  {'#':>3}  {'expected':<14} {'predicted':<14} {'method':<12} "
        f"{'conf':>5}  {'ms':>6}  hit"
    )
    console.print("  " + "-" * 70)

    correct = 0
    latencies: list[float] = []

    for index, case in enumerate(EVAL_SET, start=1):
        result = classifier.classify(case.prompt)
        latencies.append(result.latency_ms)
        hit = result.category == case.expected
        if hit:
            correct += 1
        marker = "[green]PASS[/green]" if hit else "[red]FAIL[/red]"
        console.print(
            f"  {index:>3}  {case.expected:<14} {result.category:<14} "
            f"{result.method:<12} {result.confidence:>5.2f}  {result.latency_ms:>6.0f}  {marker}"
        )

    accuracy = correct / len(EVAL_SET)
    p95 = _percentile(latencies, 95)
    mean_ms = statistics.mean(latencies)

    console.print(
        f"\nAccuracy: [bold]{accuracy:.0%}[/bold] ({correct}/{len(EVAL_SET)})  "
        f"target: {ACCURACY_TARGET:.0%}"
    )
    console.print(
        f"Classifier latency: mean={mean_ms:.0f}ms p95={p95:.0f}ms  "
        f"target: p95 < {P95_BUDGET_MS:.0f}ms"
    )

    failed = []
    if accuracy < ACCURACY_TARGET:
        failed.append(f"accuracy {accuracy:.0%} < {ACCURACY_TARGET:.0%}")
    if p95 >= P95_BUDGET_MS:
        failed.append(f"p95 {p95:.0f}ms >= {P95_BUDGET_MS:.0f}ms")

    if failed:
        console.print(f"[bold red]ACCEPTANCE FAILED[/bold red]: {', '.join(failed)}")
        return 1
    console.print("[bold green]ACCEPTANCE PASSED[/bold green]")
    return 0


async def run_forced_fallback(console: Console) -> int:
    """Fire one prompt with force_validation_failure=True and verify the chain walked."""
    client = OllamaClient()
    router = Router(client=client)
    prompt = "Write a Python function to reverse a string."

    console.print(f"[bold]Forced-fallback test[/bold] on: {prompt!r}")
    result = await router.route(prompt=prompt, force_validation_failure=True)
    console.print(f"  classified: {result.classification.category} ({result.classification.confidence:.2f})")
    console.print(f"  chain: {' -> '.join(result.fallback_chain)}")
    console.print(f"  attempts: {len(result.attempts)}")
    for index, attempt in enumerate(result.attempts):
        console.print(
            f"    [{index}] model={attempt.model} "
            f"validation_passed={attempt.validation_passed} "
            f"notes={attempt.validation_notes!r}"
        )
    console.print(f"  final_model: {result.final_model}")
    console.print(f"  fallback_triggered: {result.fallback_triggered}")

    # The chain must have at least 2 attempts (forced fail on the first, then a fallback).
    if len(result.fallback_chain) < 2:
        console.print(
            "[yellow]NOTE: fallback chain has length 1 — this category only has one model "
            "in the policy. Forced failure cannot demonstrate fallback. Re-run after a full "
            "benchmark sweep to populate richer chains.[/yellow]"
        )
        return 0

    if not result.fallback_triggered or len(result.attempts) < 2:
        console.print("[bold red]FORCED FALLBACK DID NOT TRIGGER[/bold red]")
        return 1
    console.print("[bold green]FORCED FALLBACK TRIGGERED CORRECTLY[/bold green]")
    return 0


async def run_full(console: Console) -> int:
    """Classifier eval + a few full /route calls to spot-check end-to-end."""
    rc = await run_classifier_only(console)
    console.print("\n[bold]Spot-checking 3 full route() calls...[/bold]")
    client = OllamaClient()
    router = Router(client=client)
    samples = [EVAL_SET[0], EVAL_SET[5], EVAL_SET[12]]
    for case in samples:
        result = await router.route(prompt=case.prompt)
        preview = result.response[:80].replace("\n", " ")
        console.print(
            f"  expected={case.expected:<14} "
            f"chosen={result.chosen_model:<14} "
            f"latency={result.total_latency_seconds:.1f}s  "
            f"resp={preview!r}..."
        )
    return rc


async def main() -> int:
    parser = argparse.ArgumentParser(description="Router acceptance tests")
    parser.add_argument("--full", action="store_true", help="Also run full route() calls")
    parser.add_argument("--force-fb", action="store_true", help="Run forced-fallback smoke test")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Heuristic-only classifier (skip Gemma 2B fallback). Useful on "
             "memory-constrained machines or for measuring the heuristic's "
             "ceiling in isolation.",
    )
    args = parser.parse_args()

    console = Console()

    if args.force_fb:
        return await run_forced_fallback(console)
    if args.full:
        return await run_full(console)
    return await run_classifier_only(console, heuristic_only=args.no_llm)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
