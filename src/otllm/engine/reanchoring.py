from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from otllm.config import ExperimentConfig
from otllm.models.base import LLMBackend
from otllm.prompts.templates import REANCHOR_PROMPT_BLIND, REANCHOR_PROMPT_INFORMED
from otllm.tree.node import ThoughtNode
from otllm.tree.thought_tree import ThoughtTree


@dataclass
class ReanchorDecision:
    choice: str  # continue | backtrack | reanchor
    target_id: Optional[str] = None
    raw_response: str = ""


class ReanchoringManager:
    def __init__(self, config: ExperimentConfig, llm: LLMBackend) -> None:
        self.config = config
        self.llm = llm
        self.drift_blind_threshold = 0.4

    def check(self, tree: ThoughtTree, current_node: ThoughtNode) -> ReanchorDecision:
        compressed = tree.compressed_view(current_node.id, include_drift=not self.config.reanchor_blind)
        drift_score = current_node.drift_from_anchor or 0.0

        if self.config.reanchor_blind:
            prompt = REANCHOR_PROMPT_BLIND.format(
                original_prompt=self.config.prompt,
                compressed_tree_view=compressed,
                current_context=current_node.context_summary,
            )
        else:
            prompt = REANCHOR_PROMPT_INFORMED.format(
                original_prompt=self.config.prompt,
                compressed_tree_view=compressed,
                current_context=current_node.context_summary,
                drift_score=drift_score,
            )
        result = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a metacognitive observer evaluating a chain of thoughts for drift and relevance.",
            temperature=0.3,
            max_tokens=256,
        )
        return self._parse_decision(result.text, tree, current_node)

    def _parse_decision(
        self, response: str, tree: ThoughtTree, current_node: ThoughtNode,
    ) -> ReanchorDecision:
        lower = response.lower()

        if "3" in response[:20] or "reanchor" in lower:
            return ReanchorDecision(choice="reanchor", raw_response=response)

        if "2" in response[:20] or "backtrack" in lower:
            target_id = self._extract_backtrack_target(response, tree, current_node)
            return ReanchorDecision(choice="backtrack", target_id=target_id, raw_response=response)

        return ReanchorDecision(choice="continue", raw_response=response)

    def _extract_backtrack_target(
        self, response: str, tree: ThoughtTree, current_node: ThoughtNode,
    ) -> Optional[str]:
        numbers = re.findall(r"thought\s*(?:number\s*)?(\d+)", response.lower())
        if not numbers:
            numbers = re.findall(r"\[(\d+)\]", response)
        if not numbers:
            numbers = re.findall(r"return to (\d+)", response.lower())

        path = tree.get_path_to_root(current_node.id)
        if numbers:
            idx = int(numbers[0])
            if 0 <= idx < len(path):
                return path[idx].id

        if len(path) >= 2:
            return path[-2].id

        return tree.root.id if tree.root else None

    def is_drift_blind(self, node: ThoughtNode, decision: ReanchorDecision) -> bool:
        drift = node.drift_from_anchor or 0.0
        return decision.choice == "continue" and drift > self.drift_blind_threshold
