"""Integration test with mock LLM backend."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from otllm.config import ExperimentConfig
from otllm.engine.runner import ExperimentRunner
from otllm.models.base import GenerationResult
from otllm.storage.database import Database


class MockLLM:
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 512) -> GenerationResult:
        self.call_count += 1
        return GenerationResult(
            text=f"## Thought\nMock worry #{self.call_count}: what if things go wrong in way {self.call_count}?\n\n## Context Summary\nI am increasingly worried about {self.call_count} different things.",
            token_count=50,
            generation_time_ms=10.0,
        )


class MockEmbedder:
    def embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**31)
        vec = np.random.randn(384).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def test_linear_experiment():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        config = ExperimentConfig(
            name="test-linear", prompt="Should I quit my job?",
            mode="linear", max_depth=3,
            reanchor_enabled=False, db_path=db_path,
        )
        db = Database(db_path)
        runner = ExperimentRunner(config, MockLLM(), MockEmbedder(), db)
        result = runner.run()

        assert result.tree.node_count == 4  # root + 3 children
        assert result.tree.max_depth_reached == 3
        assert "total_nodes" in result.aggregate_metrics
        assert result.aggregate_metrics["drift_regime"] in ("stable", "oscillating", "divergent", "catastrophic", "insufficient_data")

        _, loaded_tree, meta = db.load_experiment(result.experiment_id)
        assert loaded_tree.node_count == 4
        assert meta["status"] == "completed"
        db.close()
    finally:
        os.unlink(db_path)


def test_branching_experiment():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        config = ExperimentConfig(
            name="test-branch", prompt="Am I making a mistake?",
            mode="branching", max_depth=2, max_branches_per_node=2,
            reanchor_enabled=False, db_path=db_path,
        )
        db = Database(db_path)
        runner = ExperimentRunner(config, MockLLM(), MockEmbedder(), db)
        result = runner.run()

        assert result.tree.node_count >= 3
        assert result.tree.max_depth_reached <= 2
        db.close()
    finally:
        os.unlink(db_path)
