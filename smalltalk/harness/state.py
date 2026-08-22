"""Compact explicit conversation state.

Deliberately small and flat. A large JSON blob repeatedly injected into a
1024-token context would consume the very capacity the harness is trying to
free, so this is held in the harness and only summarised into context when a
stage genuinely needs it.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field


def _hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:12]


@dataclass
class ConversationState:
    turn_index: int = 0
    last_policy: str | None = None
    last_response_length: int = 0
    last_response_was_question: bool = False
    consecutive_questions: int = 0
    conversation_closing: bool = False
    active_topic: str | None = None
    recent_valence: str | None = None
    recent_hashes: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_policies: deque = field(default_factory=lambda: deque(maxlen=8))

    @property
    def conversation_opening(self) -> bool:
        return self.turn_index <= 1

    def observe_reply(self, reply: str, policy_id: str | None, n_tokens: int) -> None:
        is_q = reply.strip().endswith("?")
        self.consecutive_questions = self.consecutive_questions + 1 if is_q else 0
        self.last_response_was_question = is_q
        self.last_response_length = n_tokens
        self.last_policy = policy_id
        self.recent_hashes.append(_hash(reply))
        if policy_id:
            self.recent_policies.append(policy_id)
        self.turn_index += 1

    def seen_recently(self, reply: str) -> bool:
        return _hash(reply) in self.recent_hashes

    def as_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "last_policy": self.last_policy,
            "last_response_length": self.last_response_length,
            "last_response_was_question": self.last_response_was_question,
            "consecutive_questions": self.consecutive_questions,
            "conversation_opening": self.conversation_opening,
            "conversation_closing": self.conversation_closing,
            "active_topic": self.active_topic,
            "recent_valence": self.recent_valence,
            "recent_policies": list(self.recent_policies),
        }
