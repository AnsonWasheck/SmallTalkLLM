"""Canonical conversation schema.

Everything in the pipeline -- DailyDialog, EmpatheticDialogues, arbitrary JSONL,
and teacher-generated synthetic data -- is normalised into this one JSONL record:

    {
      "id": "dd-000123",
      "source": "dailydialog",
      "messages": [{"role": "user", "content": "hey"},
                   {"role": "assistant", "content": "hey, what's up?"}],
      "meta": {"emotion": "neutral", "topic": "greeting"}
    }

Stage-3 (distillation / rejection sampling) records add candidates:

    {
      "id": "syn-000045#3",
      "source": "teacher",
      "messages": [... context ending with a user turn ...],
      "candidates": [{"content": "long day?", "scores": {...}, "score": 4.2}, ...],
      "chosen": 0
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ROLES = ("system", "user", "assistant")


@dataclass
class Turn:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        self.content = self.content.strip()

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Candidate:
    content: str
    scores: dict[str, float] = field(default_factory=dict)
    score: float | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {})}


@dataclass
class Conversation:
    id: str
    messages: list[Turn]
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    chosen: int | None = None

    # ---- properties --------------------------------------------------------
    @property
    def num_turns(self) -> int:
        return sum(1 for m in self.messages if m.role != "system")

    @property
    def assistant_turns(self) -> list[Turn]:
        return [m for m in self.messages if m.role == "assistant"]

    def text(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)

    def alternates(self) -> bool:
        """True if user/assistant strictly alternate (system prefix allowed)."""
        seq = [m.role for m in self.messages if m.role != "system"]
        return all(a != b for a, b in zip(seq, seq[1:]))

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.meta:
            d["meta"] = self.meta
        if self.candidates:
            d["candidates"] = [c.to_dict() for c in self.candidates]
        if self.chosen is not None:
            d["chosen"] = self.chosen
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Conversation":
        return cls(
            id=str(d.get("id", "")),
            messages=[Turn(m["role"], m["content"]) for m in d["messages"]],
            source=d.get("source", "unknown"),
            meta=d.get("meta", {}) or {},
            candidates=[
                Candidate(
                    content=c["content"],
                    scores=c.get("scores", {}) or {},
                    score=c.get("score"),
                    source=c.get("source"),
                )
                for c in d.get("candidates", []) or []
            ],
            chosen=d.get("chosen"),
        )


def write_jsonl(path: str | Path, conversations: Iterable[Conversation]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Conversation]:
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Conversation.from_dict(json.loads(line))
            except Exception as exc:  # keep going; report the bad line
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def load_conversations(path: str | Path) -> list[Conversation]:
    return list(read_jsonl(path))


SYNTHETIC_SCHEMA_DOC = """\
Synthetic teacher-data request/response schema (see data/seed/*.jsonl for examples).

Generation request (what we send to the teacher model):
  {"topic": "work", "scenario": "user had a rough day of meetings",
   "emotion": "tired", "num_turns": 8, "style": "casual, 3-25 word replies"}

Accepted teacher output -- any of:
  1. canonical  : {"messages": [{"role": "user", "content": "..."}, ...]}
  2. transcript : {"conversation": "User: hey\\nAssistant: hey, what's up?"}
  3. candidates : {"messages": [...], "candidates": ["long day?", "oof, rough one"]}
Adapters in smalltalk/data/adapters.py normalise all three.
"""
