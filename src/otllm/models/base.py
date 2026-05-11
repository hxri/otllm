from __future__ import annotations

import dataclasses
from typing import List, Optional, Protocol

import numpy as np


@dataclasses.dataclass
class GenerationResult:
    text: str
    thinking: Optional[str] = None
    token_count: int = 0
    generation_time_ms: float = 0.0


class LLMBackend(Protocol):
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> GenerationResult: ...


class EmbeddingBackend(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]: ...
