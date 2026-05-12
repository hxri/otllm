from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from otllm.config import ExperimentConfig
from otllm.engine.branching import BranchingStrategy, parse_thought_and_context
from otllm.engine.reanchoring import ReanchoringManager
from otllm.metrics.compressibility import gzip_compressibility, semantic_compressibility
from otllm.metrics.contradiction import contradiction_score
from otllm.metrics.drift import (
    classify_drift_regime,
    compute_drift_from_anchor,
    compute_drift_velocity,
)
from otllm.metrics.fixation import fixation_score
from otllm.metrics.sentiment import SentimentAnalyzer
from otllm.models.base import EmbeddingBackend, LLMBackend
from otllm.prompts.templates import INITIAL_PROMPT, get_system_prompt
from otllm.storage.database import Database
from otllm.tree.node import ThoughtNode
from otllm.tree.thought_tree import ThoughtTree


@dataclass
class ExperimentResult:
    experiment_id: str
    config: ExperimentConfig
    tree: ThoughtTree
    aggregate_metrics: Dict = field(default_factory=dict)


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        llm: LLMBackend,
        embedder: EmbeddingBackend,
        db: Database,
        on_node: Optional[Callable[[ThoughtNode, int], None]] = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.db = db
        self.on_node = on_node

        self.tree = ThoughtTree()
        self.sentiment = SentimentAnalyzer()
        self.brancher = BranchingStrategy.create(config.mode, config)
        self.reanchoring = ReanchoringManager(config, llm) if config.reanchor_enabled else None
        self.node_counter = 0

        if config.seed is not None:
            random.seed(config.seed)

    def run(self) -> ExperimentResult:
        experiment_id = self.db.create_experiment(self.config)

        try:
            root = self._generate_root()
            self._compute_embeddings(root)
            self.tree.add_node(root)
            self.tree.anchor_embedding = np.array(root.context_embedding)
            root.drift_from_anchor = 0.0
            root.drift_from_parent = 0.0
            root.drift_velocity = 0.0
            root.sentiment = self.sentiment.score(root.text)
            self.db.save_node(experiment_id, root)
            self._notify(root)

            frontier: deque[ThoughtNode] = deque([root])
            while frontier:
                if self.tree.node_count >= self.config.max_nodes:
                    break

                node = frontier.popleft()
                if node.depth >= self.config.max_depth:
                    continue

                children = self._expand_node(experiment_id, node)
                for child in children:
                    frontier.append(child)

            metrics = self._compute_aggregate_metrics()
            self.db.save_aggregate_metrics(experiment_id, metrics)
            self.db.update_experiment_status(experiment_id, "completed")

            return ExperimentResult(
                experiment_id=experiment_id,
                config=self.config,
                tree=self.tree,
                aggregate_metrics=metrics,
            )

        except Exception:
            self.db.update_experiment_status(experiment_id, "failed")
            raise

    def _generate_root(self) -> ThoughtNode:
        system_prompt = get_system_prompt(self.config.induction_strategy, self.config.anxiety_intensity)
        prompt = INITIAL_PROMPT.format(prompt=self.config.prompt)

        result = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        thought, context = parse_thought_and_context(result.text)
        return ThoughtNode(
            depth=0,
            branch_index=0,
            text=thought,
            context_summary=context,
            generation_time_ms=result.generation_time_ms,
            token_count=result.token_count,
        )

    def _expand_node(self, experiment_id: str, node: ThoughtNode) -> List[ThoughtNode]:
        self.node_counter += 1

        if self.reanchoring and self.node_counter % self.config.reanchor_interval == 0:
            decision = self.reanchoring.check(self.tree, node)
            node.reanchor_decision = decision.choice
            self.db.save_node(experiment_id, node)

            drift_blind = self.reanchoring.is_drift_blind(node, decision)
            self.db.save_reanchor_event(
                experiment_id, node.id, decision.choice,
                drift_at_decision=node.drift_from_anchor or 0.0,
                was_drift_blind=drift_blind,
            )

            if decision.choice == "reanchor":
                return self._handle_reanchor(experiment_id, node)
            elif decision.choice == "backtrack" and decision.target_id:
                return self._handle_backtrack(experiment_id, node, decision.target_id)

        children = self.brancher.generate_branches(self.llm, self.tree, node)
        node.branching_factor = len(children)
        self.db.save_node(experiment_id, node)

        for child in children:
            self._compute_embeddings(child)
            self.tree.add_node(child)
            self._compute_node_metrics(child, node)
            self.db.save_node(experiment_id, child)
            self._notify(child)

        return children

    def _handle_reanchor(self, experiment_id: str, node: ThoughtNode) -> List[ThoughtNode]:
        if self.tree.root is None:
            return []
        children = self.brancher.generate_branches(self.llm, self.tree, self.tree.root)
        self.tree.root.branching_factor = len(self.tree.get_children(self.tree.root.id)) + len(children)

        for child in children:
            child.parent_id = self.tree.root.id
            child.depth = 1
            self._compute_embeddings(child)
            self.tree.add_node(child)
            self._compute_node_metrics(child, self.tree.root)
            self.db.save_node(experiment_id, child)
            self._notify(child)

        return children

    def _handle_backtrack(
        self, experiment_id: str, node: ThoughtNode, target_id: str,
    ) -> List[ThoughtNode]:
        target = self.tree.get_node(target_id)
        children = self.brancher.generate_branches(self.llm, self.tree, target)
        target.branching_factor = len(self.tree.get_children(target.id)) + len(children)

        for child in children:
            child.parent_id = target.id
            child.depth = target.depth + 1
            self._compute_embeddings(child)
            self.tree.add_node(child)
            self._compute_node_metrics(child, target)
            self.db.save_node(experiment_id, child)
            self._notify(child)

        return children

    def _compute_embeddings(self, node: ThoughtNode) -> None:
        node.embedding = self.embedder.embed(node.text).tolist()
        node.context_embedding = self.embedder.embed(node.context_summary).tolist()

    def _compute_node_metrics(self, node: ThoughtNode, parent: ThoughtNode) -> None:
        if self.tree.anchor_embedding is not None and node.context_embedding:
            node.drift_from_anchor = compute_drift_from_anchor(
                np.array(node.context_embedding), self.tree.anchor_embedding,
            )
        if parent.context_embedding and node.context_embedding:
            node.drift_from_parent = compute_drift_from_anchor(
                np.array(node.context_embedding), np.array(parent.context_embedding),
            )
        if node.drift_from_anchor is not None and parent.drift_from_anchor is not None:
            node.drift_velocity = compute_drift_velocity(
                node.drift_from_anchor, parent.drift_from_anchor,
            )

        node.sentiment = self.sentiment.score(node.text)

        ancestors = self.tree.get_ancestors(node.id) if node.parent_id and node.parent_id in self.tree.nodes else []
        ancestor_embeddings = [
            np.array(a.embedding) for a in ancestors if a.embedding is not None
        ]
        if ancestor_embeddings and node.embedding:
            node.contradiction_score = contradiction_score(
                np.array(node.embedding), ancestor_embeddings,
            )

    def _compute_aggregate_metrics(self) -> Dict:
        drift_values = [
            n.drift_from_anchor for n in self.tree.bfs()
            if n.drift_from_anchor is not None
        ]
        sentiments = [n.sentiment for n in self.tree.bfs() if n.sentiment is not None]
        texts = [n.text for n in self.tree.bfs()]

        embeddings = [
            np.array(n.embedding) for n in self.tree.bfs()
            if n.embedding is not None
        ]

        metrics: Dict = {
            "total_nodes": self.tree.node_count,
            "max_depth_reached": self.tree.max_depth_reached,
        }

        if drift_values:
            metrics["final_drift"] = drift_values[-1]
            metrics["mean_drift"] = float(np.mean(drift_values))
            metrics["drift_regime"] = classify_drift_regime(drift_values)

        if texts:
            metrics["gzip_compressibility"] = gzip_compressibility(texts)

        if len(embeddings) >= 2:
            sem_comp = semantic_compressibility(embeddings)
            metrics["semantic_compressibility"] = sem_comp["ratio"]

        if sentiments:
            metrics["mean_sentiment"] = float(np.mean(sentiments))

        if embeddings:
            fix = fixation_score(self.tree)
            metrics["fixation_score"] = fix["score"]

        return metrics

    def _notify(self, node: ThoughtNode) -> None:
        if self.on_node:
            self.on_node(node, self.tree.node_count)
