from __future__ import annotations

import time
from typing import Optional

import httpx

from otllm.models.base import GenerationResult


class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        enable_thinking: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.enable_thinking = enable_thinking
        self._client = httpx.Client(timeout=120.0)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> GenerationResult:
        if self.enable_thinking:
            prompt = "/think\n" + prompt
        else:
            prompt = "/no_think\n" + prompt

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        t0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        data = resp.json()
        raw_response = data.get("response", "")
        thinking, text = self._split_thinking(raw_response)

        return GenerationResult(
            text=text,
            thinking=thinking,
            token_count=data.get("eval_count", 0),
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
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return {
                "connected": True,
                "models": models,
                "model_available": any(self.model in m for m in models),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def close(self) -> None:
        self._client.close()
