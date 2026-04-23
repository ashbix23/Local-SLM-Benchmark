"""
Benchmark execution layer.

Runs a single (model × prompt) generation and persists the result. Two
non-obvious pieces:

  1. WARMUP. Ollama loads models into memory on first request. The cold
     load adds 3–5 seconds of latency and contaminates TTFT metrics. We
     pay that cost once per model with a throwaway prompt, then measure
     against warm models.

  2. MEMORY VIA OLLAMA API. On Apple Silicon, model weights live in GPU-
     shared memory managed by Metal, which doesn't show up in any process
     RSS accounting. We use Ollama's own /api/ps endpoint instead — it
     reports size_vram (bytes on the GPU), which is the honest number.
     This is also what `ollama ps` shows users.
"""

from typing import Optional
from app.ollama_client import OllamaClient, GenerationResult
from app.models import BenchmarkRun
from app.database import insert_run


class BenchmarkRunner:
    """Orchestrates a single benchmarked generation end-to-end."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        self._warmed_up_models: set[str] = set()

    async def warmup(self, model: str) -> None:
        """
        Pay the cold-load cost so subsequent measurements are fair.

        Ollama keeps models in memory for ~5 minutes after last use. As
        long as benchmark runs for a given model stay within that window,
        one warmup per model per sweep is sufficient.
        """
        if model in self._warmed_up_models:
            return

        await self.client.generate(
            model=model,
            prompt="Hi",
            temperature=0.0,
        )
        self._warmed_up_models.add(model)

    async def run_single(
        self,
        model: str,
        task_category: str,
        task_id: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        persist: bool = True,
    ) -> BenchmarkRun:
        """
        Execute one benchmarked generation.

        Memory is read from /api/ps after generation completes — the
        runner is guaranteed to be alive and the model loaded at that
        point. quality_score stays None; scoring is task-specific and
        happens later in the runner script.
        """
        await self.warmup(model)

        generation: GenerationResult = await self.client.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
        )

        memory_mb = await self.client.get_model_memory_mb(model)

        run = BenchmarkRun(
            model=model,
            task_category=task_category,
            task_id=task_id,
            prompt=prompt,
            response=generation.response_text,
            time_to_first_token=generation.time_to_first_token,
            total_latency=generation.total_latency,
            tokens_generated=generation.tokens_generated,
            tokens_per_second=generation.tokens_per_second,
            peak_memory_mb=memory_mb,
            quality_score=None,
            quality_notes=None,
        )

        if persist:
            insert_run(run)

        return run
