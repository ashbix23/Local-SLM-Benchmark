"""
Ollama HTTP client wrapper.

Talks to the local Ollama server on port 11434. Captures timing metrics
(time-to-first-token, total latency, tokens/sec) on every generation so
the benchmark layer can consume them directly. Also exposes memory
accounting via /api/ps, which on Apple Silicon is the only honest way
to measure model footprint (weights live in GPU-shared memory managed
by Metal, invisible to process RSS).
"""

import time
import json
import httpx
from dataclasses import dataclass
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 300.0  # 5 min — 7B model on CPU-heavy tasks can be slow


@dataclass
class GenerationResult:
    """Everything the benchmark layer needs from a single generation."""
    model: str
    prompt: str
    response_text: str
    time_to_first_token: float   # seconds
    total_latency: float         # seconds
    tokens_generated: int
    tokens_per_second: float
    raw_ollama_metrics: dict     # Ollama's own timing fields, for debugging


class OllamaClient:
    """Async client for Ollama's local HTTP API with timing capture."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout

    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """
        Generate with streaming so we can capture time-to-first-token.

        We stream the response even though we return the full text at the end —
        streaming is the only way to measure TTFT, which is a key UX metric
        (how fast does the user see *something*?).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        response_chunks: list[str] = []
        final_metrics: dict = {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    chunk = json.loads(line)

                    # First chunk with actual text = time to first token
                    if first_token_time is None and chunk.get("response"):
                        first_token_time = time.perf_counter()

                    if "response" in chunk:
                        response_chunks.append(chunk["response"])

                    # The final chunk has done=True and includes Ollama's own metrics
                    if chunk.get("done"):
                        final_metrics = chunk

        end_time = time.perf_counter()
        total_latency = end_time - start_time
        time_to_first_token = (
            first_token_time - start_time if first_token_time else total_latency
        )

        response_text = "".join(response_chunks)
        tokens_generated = final_metrics.get("eval_count", 0)

        # Ollama reports eval_duration in nanoseconds — convert for tokens/sec
        eval_duration_ns = final_metrics.get("eval_duration", 0)
        if eval_duration_ns > 0 and tokens_generated > 0:
            tokens_per_second = tokens_generated / (eval_duration_ns / 1e9)
        else:
            tokens_per_second = 0.0

        return GenerationResult(
            model=model,
            prompt=prompt,
            response_text=response_text,
            time_to_first_token=time_to_first_token,
            total_latency=total_latency,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            raw_ollama_metrics=final_metrics,
        )

    async def list_models(self) -> list[str]:
        """Return names of models currently pulled. Useful for sanity-checking."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]

    async def get_model_memory_mb(self, model: str) -> float:
        """
        Return currently-loaded memory footprint for a model, in MB.

        Hits Ollama's /api/ps endpoint, which reports the same info as the
        `ollama ps` CLI. We use `size_vram` (bytes on the GPU) since on
        Apple Silicon all inference is GPU-resident. Returns 0.0 if the
        model isn't currently loaded — callers should call this after a
        generation so the runner is guaranteed to be alive.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            data = response.json()

        for entry in data.get("models", []):
            if entry.get("name") == model or entry.get("model") == model:
                # Prefer VRAM (GPU) bytes on Apple Silicon; fall back to total
                memory_bytes = entry.get("size_vram") or entry.get("size", 0)
                return memory_bytes / (1024 * 1024)

        return 0.0
