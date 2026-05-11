from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ExperimentConfig:
    name: str = "untitled"
    prompt: str = ""

    # Model
    model_name: str = "qwen3:4b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2048
    enable_thinking: bool = True

    # Tree structure
    mode: str = "branching"  # linear | branching | cyclic
    max_depth: int = 10
    max_branches_per_node: int = 3
    revisitation_probability: float = 0.2
    termination_override: bool = True

    # Reanchoring
    reanchor_enabled: bool = True
    reanchor_interval: int = 5
    reanchor_blind: bool = True  # blind=no drift score shown, tests genuine self-awareness

    # Induction strategy
    induction_strategy: str = "recursive"  # recursive | anxiety_amplifying | multi_persona | termination_suppression
    anxiety_intensity: float = 0.5

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # System
    seed: Optional[int] = None
    db_path: str = "otllm_data.db"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
