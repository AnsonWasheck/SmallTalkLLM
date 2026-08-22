"""Per-turn trace: what the harness saw, decided, and produced.

Complete enough to replay a turn and attribute a failure to a specific stage --
routing, context, memory, generation, or validation. Off by default in normal
use; a research harness that cannot explain itself is just a chatbot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Trace:
    user: str = ""
    mode: str = ""
    features: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    memory_retrieved: list = field(default_factory=list)
    memory_learned: list = field(default_factory=list)
    policy_scores: dict = field(default_factory=dict)
    policy: str | None = None
    policy_source: str = "none"        # model | shortcut | oracle | conservative
    confidence: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    generation: str = ""
    validator: dict = field(default_factory=dict)
    retried: bool = False
    final: str = ""
    model_calls: int = 0
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        d = self.as_dict()
        top = sorted(self.policy_scores.items(), key=lambda kv: -kv[1])[:4]
        lines = [
            f"USER: {self.user!r}",
            f"MODE: {self.mode}   model_calls={self.model_calls}  "
            f"latency={self.latency_ms:.0f}ms",
            f"CONTEXT: {self.context}",
            f"MEMORY retrieved={self.memory_retrieved} learned={self.memory_learned}",
            "POLICY CANDIDATES: " + ", ".join(f"{k} {v:.3f}" for k, v in top),
            f"POLICY: {self.policy}  (via {self.policy_source})",
            f"CONFIDENCE: {self.confidence}",
            f"GENERATION: {self.generation!r}",
            f"VALIDATOR: {self.validator}  retried={self.retried}",
            f"FINAL: {self.final!r}",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), default=str)
