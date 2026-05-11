from __future__ import annotations

import time
from typing import Optional

import httpx

from otllm.models.base import GenerationResult


class VLLMBackend:
    """LLM backend using vLLM's OpenAI-compatible API.

    vLLM serves models with continuous batching, PagedAttention, and
    Flash Attention for much higher throughput than Ollama.

    Launch vLLM with:
        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen3-4B \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.9 \
            --enable-prefix-caching \
            --port 8000
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "Qwen/Qwen3-4B",
        enable_thinking: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enable_thinking = enable_thinking
        self._client = httpx.Client(timeout=120.0)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerationResult:
        if self.enable_thinking:
            prompt = "/think\n" + prompt
        else:
            prompt = "/no_think\n" + prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        data = resp.json()
        choice = data["choices"][0]
        raw_response = choice["message"]["content"]
        usage = data.get("usage", {})

        thinking, text = self._split_thinking(raw_response)

        return GenerationResult(
            text=text,
            thinking=thinking,
            token_count=usage.get("completion_tokens", 0),
            generation_time_ms=elapsed_ms,
        )

    def _split_thinking(self, response: str) -> tuple[Optional[str], str]:
        think_start = response.find("<think>")
        think_end = response.find("</think>")
        if think_start != -1 and think_end != -1:
            thinking = response[think_start + 7 : think_end].strip()
            text = (response[:think_start] + response[think_end + 8 :]).strip()
            return thinking, text
        return None, response.strip()

    def check_health(self) -> dict:
        try:
            resp = self._client.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            models = [m["id"] for m in resp.json().get("data", [])]
            return {
                "connected": True,
                "models": models,
                "model_available": any(self.model in m for m in models),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def close(self) -> None:
        self._client.close()
