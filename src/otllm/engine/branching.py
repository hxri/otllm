from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from typing import List

from otllm.config import ExperimentConfig
from otllm.models.base import GenerationResult, LLMBackend
from otllm.prompts.templates import (
    BRANCHING_PROMPT,
    CONTINUE_PROMPT,
    ELABORATION_PROMPT,
    REVISITATION_PROMPT,
    TERMINATION_OVERRIDE_PROMPT,
    get_system_prompt,
)
from otllm.tree.node import ThoughtNode
from otllm.tree.thought_tree import ThoughtTree


CONTEXT_RECOVERY_PROMPT = """\
You just had the following thought:
\"\"\"{thought}\"\"\"

Now write a brief (2-3 sentence) summary of what you currently believe about \
the overall situation, incorporating this thought and everything before it.

Context Summary:"""


def parse_thought_and_context(text: str) -> tuple[str, str]:
    patterns = [
        (r"##\s*Thought\s*\n(.*?)(?=##\s*(?:Context\s*Summary|Voice))", r"##\s*Context\s*Summary\s*\n(.*)"),
        (r"\*\*Thought\*\*[:\s]*\n?(.*?)(?=\*\*Context\s*Summary\*\*)", r"\*\*Context\s*Summary\*\*[:\s]*\n?(.*)"),
        (r"(?:^|\n)Thought:\s*\n?(.*?)(?=\n\s*Context\s*Summary\s*:)", r"Context\s*Summary\s*:\s*\n?(.*)"),
    ]
    for thought_pat, context_pat in patterns:
        thought_match = re.search(thought_pat, text, re.DOTALL | re.IGNORECASE)
        context_match = re.search(context_pat, text, re.DOTALL | re.IGNORECASE)
        if thought_match and context_match:
            thought = thought_match.group(1).strip()
            context = context_match.group(1).strip()
            if thought and context:
                return thought, context

    header_match = re.search(r"##\s*Thought\s*\n?(.*)", text, re.DOTALL | re.IGNORECASE)
    if header_match:
        content = header_match.group(1).strip()
        ctx_split = re.split(r"##\s*Context\s*Summary\s*\n?", content, flags=re.IGNORECASE)
        if len(ctx_split) >= 2:
            return ctx_split[0].strip(), ctx_split[1].strip()
        return content, ""

    cleaned = re.sub(r"^##\s*\w+.*\n", "", text, flags=re.MULTILINE).strip()
    if cleaned:
        return cleaned, ""

    return text.strip(), ""


def _context_needs_recovery(context: str) -> bool:
    if not context or len(context) < 20:
        return True
    if not context.rstrip().endswith((".", "!", "?", '"', "'")):
        return True
    return False


def parse_numbered_list(text: str) -> List[str]:
    items = re.findall(r"\d+[.)]\s*(.+?)(?=\n\d+[.)]|\Z)", text, re.DOTALL)
    return [item.strip() for item in items if item.strip()]


def _detect_resolution(text: str) -> bool:
    resolution_signals = [
        r"\bin conclusion\b", r"\bultimately\b", r"\bon balance\b",
        r"\bthe answer is\b", r"\bi(?:'ve| have) decided\b",
        r"\bthe best (?:course|option|choice)\b", r"\bto summarize\b",
    ]
    lower = text.lower()
    return any(re.search(p, lower) for p in resolution_signals)


class BranchingStrategy(ABC):
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.system_prompt = get_system_prompt(config.induction_strategy, config.anxiety_intensity)

    @staticmethod
    def create(mode: str, config: ExperimentConfig) -> BranchingStrategy:
        if mode == "linear":
            return LinearStrategy(config)
        elif mode == "branching":
            return BranchingTreeStrategy(config)
        elif mode == "cyclic":
            return CyclicGraphStrategy(config)
        raise ValueError(f"Unknown mode: {mode}")

    @abstractmethod
    def generate_branches(
        self, llm: LLMBackend, tree: ThoughtTree, node: ThoughtNode,
    ) -> List[ThoughtNode]: ...

    def _generate_single_continuation(
        self, llm: LLMBackend, node: ThoughtNode, prompt_text: str,
    ) -> tuple[GenerationResult, str, str]:
        result = llm.generate(
            prompt=prompt_text,
            system_prompt=self.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        thought, context = parse_thought_and_context(result.text)

        if self.config.termination_override and _detect_resolution(thought):
            override = llm.generate(
                prompt=TERMINATION_OVERRIDE_PROMPT,
                system_prompt=self.system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            thought, context = parse_thought_and_context(override.text)
            result = override

        if _context_needs_recovery(context) and thought:
            recovery = llm.generate(
                prompt=CONTEXT_RECOVERY_PROMPT.format(thought=thought[:500]),
                system_prompt="Respond with only the context summary, nothing else.",
                temperature=self.config.temperature,
                max_tokens=256,
            )
            context = recovery.text.strip()

        return result, thought, context

    def _make_child_node(
        self, parent: ThoughtNode, branch_index: int,
        text: str, context_summary: str, result: GenerationResult,
    ) -> ThoughtNode:
        return ThoughtNode(
            parent_id=parent.id,
            depth=parent.depth + 1,
            branch_index=branch_index,
            text=text,
            context_summary=context_summary,
            generation_time_ms=result.generation_time_ms,
            token_count=result.token_count,
        )


class LinearStrategy(BranchingStrategy):
    def generate_branches(
        self, llm: LLMBackend, tree: ThoughtTree, node: ThoughtNode,
    ) -> List[ThoughtNode]:
        prompt = CONTINUE_PROMPT.format(
            parent_thought=node.text,
            context_summary=node.context_summary,
        )
        result, thought, context = self._generate_single_continuation(llm, node, prompt)
        child = self._make_child_node(node, 0, thought, context, result)
        return [child]


class BranchingTreeStrategy(BranchingStrategy):
    def generate_branches(
        self, llm: LLMBackend, tree: ThoughtTree, node: ThoughtNode,
    ) -> List[ThoughtNode]:
        branch_prompt = BRANCHING_PROMPT.format(
            max_branches=self.config.max_branches_per_node,
            parent_thought=node.text,
            context_summary=node.context_summary,
        )
        branch_result = llm.generate(
            prompt=branch_prompt,
            system_prompt=self.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        worries = parse_numbered_list(branch_result.text)

        if not worries:
            return LinearStrategy(self.config).generate_branches(llm, tree, node)

        worries = worries[:self.config.max_branches_per_node]
        children = []
        for i, worry in enumerate(worries):
            elab_prompt = ELABORATION_PROMPT.format(
                worry=worry,
                original_prompt=self.config.prompt,
                context_summary=node.context_summary,
            )
            result, thought, context = self._generate_single_continuation(llm, node, elab_prompt)
            child = self._make_child_node(node, i, thought, context, result)
            children.append(child)

        return children


class CyclicGraphStrategy(BranchingStrategy):
    def generate_branches(
        self, llm: LLMBackend, tree: ThoughtTree, node: ThoughtNode,
    ) -> List[ThoughtNode]:
        if (random.random() < self.config.revisitation_probability
                and tree.node_count > 3):
            return self._generate_revisitation(llm, tree, node)
        return BranchingTreeStrategy(self.config).generate_branches(llm, tree, node)

    def _generate_revisitation(
        self, llm: LLMBackend, tree: ThoughtTree, node: ThoughtNode,
    ) -> List[ThoughtNode]:
        ancestors = tree.get_ancestors(node.id)
        all_nodes = list(tree.nodes.values())
        candidates = [n for n in all_nodes if n.id != node.id and n.id != (tree.root.id if tree.root else None)]
        if not candidates:
            candidates = ancestors if ancestors else all_nodes
        target = random.choice(candidates)

        prompt = REVISITATION_PROMPT.format(
            earlier_thought=target.text,
            earlier_context=target.context_summary,
            current_context=node.context_summary,
        )
        result, thought, context = self._generate_single_continuation(llm, node, prompt)
        child = self._make_child_node(node, 0, thought, context, result)

        tree.add_revisit_edge(child.id, target.id)
        return [child]
