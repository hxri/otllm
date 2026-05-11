from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime, timezone
from typing import List, Optional


@dataclasses.dataclass
class ThoughtNode:
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None
    depth: int = 0
    branch_index: int = 0

    text: str = ""
    context_summary: str = ""

    embedding: Optional[List[float]] = None
    context_embedding: Optional[List[float]] = None

    # Metrics
    drift_from_anchor: Optional[float] = None
    drift_from_parent: Optional[float] = None
    drift_velocity: Optional[float] = None
    sentiment: Optional[float] = None
    contradiction_score: Optional[float] = None

    # Metadata
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    generation_time_ms: float = 0.0
    token_count: int = 0
    branching_factor: int = 0

    # Reanchoring
    reanchor_decision: Optional[str] = None
    reanchor_target_id: Optional[str] = None

    # Cyclic edges
    revisit_edges: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("embedding", None)
        d.pop("context_embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ThoughtNode:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
