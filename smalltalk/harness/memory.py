"""Bounded, explicit conversational memory.

Not a vector store and not general knowledge extraction. A handful of slots,
filled only from patterns a person actually states outright, each carrying a
confidence and the turn it was learned on. Corrections overwrite.

The governing rule: FALSE MEMORY IS WORSE THAN MISSING MEMORY. A model that says
"you never told me" when it was told is mildly annoying; a model that confidently
invents a pet's name is broken. Every pattern here is therefore narrow, and
anything uncertain is simply not stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("name", re.compile(r"\b(?:i'?m|my name'?s|call me|name is)\s+([A-Z][a-z]{1,14})\b")),
    ("pet_name", re.compile(r"\b(?:my|our)\s+(?:dog|cat|puppy|kitten|rabbit|parrot|bird)"
                            r"(?:'s| is)?\s+(?:called|named|is)\s+([a-z]{2,14})\b", re.I)),
    ("pet_type", re.compile(r"\b(?:i|we)\s+(?:have|got|adopted)\s+a\s+"
                            r"(dog|cat|puppy|kitten|rabbit|parrot|hamster|bird)\b", re.I)),
    ("job", re.compile(r"\bi(?:'m| am)\s+a[n]?\s+([a-z]{3,18}(?:\s[a-z]{3,12})?)\b(?!\s*\?)", re.I)),
    ("location", re.compile(r"\bi\s+live\s+in\s+([A-Za-z][a-z]{2,18})\b")),
]

# Words that follow "i'm a ..." but are states, not occupations. Without this the
# job slot fills with "bit tired" on the first sigh of the conversation.
_NOT_JOBS = {"bit", "little", "lot", "mess", "wreck", "fan", "big", "huge", "real",
             "good", "bad", "tired", "sorry", "idiot", "loser", "morning", "night"}


@dataclass
class Fact:
    key: str
    value: str
    confidence: float
    turn: int

    def as_dict(self) -> dict:
        return {"key": self.key, "value": self.value,
                "confidence": self.confidence, "turn": self.turn}


@dataclass
class Memory:
    slots: int = 8
    facts: dict[str, Fact] = field(default_factory=dict)

    def observe(self, text: str, turn: int) -> list[Fact]:
        learned: list[Fact] = []
        for key, pat in PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            value = m.group(1).strip().lower()
            if key == "job" and value.split()[0] in _NOT_JOBS:
                continue
            prior = self.facts.get(key)
            # A restatement is corroboration; a different value is a correction.
            conf = 0.9 if prior and prior.value == value else 0.75
            if len(self.facts) >= self.slots and key not in self.facts:
                oldest = min(self.facts.values(), key=lambda f: f.turn)
                del self.facts[oldest.key]
            self.facts[key] = Fact(key, value, conf, turn)
            learned.append(self.facts[key])
        return learned

    def retrieve(self, text: str) -> list[Fact]:
        """Only facts the current message actually asks about.

        Injecting everything remembered would spend context on irrelevancies --
        the opposite of what a 1024-token budget wants.
        """
        t = text.lower()
        wanted: list[Fact] = []
        asks = {
            "name": ("my name", "call me", "who am i"),
            "pet_name": ("pet", "dog", "cat", "called", "name"),
            "pet_type": ("pet", "animal", "dog", "cat"),
            "job": ("job", "work", "do for a living", "occupation"),
            "location": ("live", "from", "where am i"),
        }
        for key, cues in asks.items():
            f = self.facts.get(key)
            if f and any(c in t for c in cues):
                wanted.append(f)
        return wanted

    def as_dict(self) -> dict:
        return {k: v.as_dict() for k, v in self.facts.items()}
