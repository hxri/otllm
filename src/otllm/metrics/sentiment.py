from __future__ import annotations

from typing import List, Tuple

from otllm.tree.thought_tree import ThoughtTree


class SentimentAnalyzer:
    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return self._analyzer.polarity_scores(text)["compound"]

    def trajectory(self, tree: ThoughtTree) -> List[Tuple[str, float]]:
        return [(node.id, node.sentiment) for node in tree.bfs() if node.sentiment is not None]
