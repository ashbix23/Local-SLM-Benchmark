"""
Chart generator.

Reads the latest benchmark batch from SQLite and renders four PNGs into
charts/. These get embedded directly in BENCHMARKS.md, so we optimize
for clarity at small render sizes (GitHub renders markdown images at
modest widths) and for accessibility (color-distinct, labeled axes).

Run anytime after a benchmark sweep — does not re-run the benchmark.

Usage:
    python scripts/generate_charts.py
"""

import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, ".")

from app.database import fetch_latest_run_batch


CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# Consistent color per model across all charts — readers learn the mapping once
MODEL_COLORS = {
    "gemma2:2b": "#4285F4",      # Google blue
    "llama3.2:3b": "#0668E1",    # Meta blue (close to but distinct from Google's)
    "qwen2.5:7b": "#FF6A00",     # Alibaba orange
}

# Friendly display names — drop the Ollama tag for chart labels
MODEL_DISPLAY = {
    "gemma2:2b": "Gemma 2 (2B)",
    "llama3.2:3b": "Llama 3.2 (3B)",
    "qwen2.5:7b": "Qwen 2.5 (7B)",
}


def _setup_style() -> None:
    """Consistent visual style across all charts."""
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.axisbelow"] = True


def _group_by_model(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    return dict(grouped)


def _ordered_models(grouped: dict[str, list[dict]]) -> list[str]:
    """Return models in the canonical lineup order, skipping any that aren't present."""
    canonical = ["gemma2:2b", "llama3.2:3b", "qwen2.5:7b"]
    return [model for model in canonical if model in grouped]


def chart_latency(rows: list[dict]) -> Path:
    """Average total latency per model. Single bar per model — clean and readable."""
    grouped = _group_by_model(rows)
    models = _ordered_models(grouped)
    avg_latency = [
        sum(r["total_latency"] for r in grouped[m]) / len(grouped[m])
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [MODEL_DISPLAY[m] for m in models],
        avg_latency,
        color=[MODEL_COLORS[m] for m in models],
        edgecolor="white",
        linewidth=2,
    )
    ax.set_ylabel("Avg latency per response (seconds)")
    ax.set_title("Response latency — lower is better", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fs"))

    for bar, value in zip(bars, avg_latency):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    output = CHARTS_DIR / "latency.png"
    plt.savefig(output)
    plt.close(fig)
    return output


def chart_throughput(rows: list[dict]) -> Path:
    """Tokens per second per model. Higher is better — fast generation."""
    grouped = _group_by_model(rows)
    models = _ordered_models(grouped)
    avg_throughput = [
        sum(r["tokens_per_second"] for r in grouped[m]) / len(grouped[m])
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [MODEL_DISPLAY[m] for m in models],
        avg_throughput,
        color=[MODEL_COLORS[m] for m in models],
        edgecolor="white",
        linewidth=2,
    )
    ax.set_ylabel("Tokens per second")
    ax.set_title("Generation throughput — higher is better", fontweight="bold")

    for bar, value in zip(bars, avg_throughput):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    output = CHARTS_DIR / "throughput.png"
    plt.savefig(output)
    plt.close(fig)
    return output


def chart_memory(rows: list[dict]) -> Path:
    """Memory footprint per model — relevant for hardware planning."""
    grouped = _group_by_model(rows)
    models = _ordered_models(grouped)
    memory_mb = [
        max(r["peak_memory_mb"] for r in grouped[m])
        for m in models
    ]
    memory_gb = [m / 1024 for m in memory_mb]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        [MODEL_DISPLAY[m] for m in models],
        memory_gb,
        color=[MODEL_COLORS[m] for m in models],
        edgecolor="white",
        linewidth=2,
    )
    ax.set_ylabel("Memory footprint (GB)")
    ax.set_title("VRAM usage — lower is better for constrained hardware", fontweight="bold")

    for bar, value in zip(bars, memory_gb):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f} GB",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    output = CHARTS_DIR / "memory.png"
    plt.savefig(output)
    plt.close(fig)
    return output


def chart_quality_by_category(rows: list[dict]) -> Path:
    """
    Grouped bars: quality score per model per task category.

    This is the chart that surfaces which models are strong/weak at WHICH
    kinds of tasks — the most narratively useful chart in the set. A model
    that wins overall might still lose on, say, code generation.
    """
    grouped = _group_by_model(rows)
    models = _ordered_models(grouped)
    categories = ["reasoning", "summarization", "extraction", "code"]
    category_labels = ["Reasoning", "Summarization", "Extraction", "Code"]

    # scores[model][category] = average score
    scores: dict[str, dict[str, float]] = {}
    for model in models:
        scores[model] = {}
        for category in categories:
            relevant = [
                r for r in grouped[model]
                if r["task_category"] == category and r["quality_score"] is not None
            ]
            scores[model][category] = (
                sum(r["quality_score"] for r in relevant) / len(relevant)
                if relevant else 0.0
            )

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.25
    x_positions = list(range(len(categories)))

    for index, model in enumerate(models):
        offsets = [x + (index - 1) * bar_width for x in x_positions]
        values = [scores[model][cat] for cat in categories]
        bars = ax.bar(
            offsets,
            values,
            bar_width,
            label=MODEL_DISPLAY[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=1.5,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(category_labels)
    ax.set_ylabel("Quality score (0–1)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Quality by task category", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)

    output = CHARTS_DIR / "quality_by_category.png"
    plt.savefig(output)
    plt.close(fig)
    return output


def main() -> None:
    rows = fetch_latest_run_batch()
    if not rows:
        print("No runs found in DB. Run scripts/run_benchmark.py first.")
        return

    print(f"Generating charts from {len(rows)} runs...")
    _setup_style()

    outputs = [
        chart_latency(rows),
        chart_throughput(rows),
        chart_memory(rows),
        chart_quality_by_category(rows),
    ]

    print("\nCharts written to:")
    for path in outputs:
        print(f"  {path}")
    print("\nEmbed in BENCHMARKS.md with:")
    for path in outputs:
        print(f"  ![{path.stem}]({path})")


if __name__ == "__main__":
    main()
