"""Deterministic post-generation validation.

Catches system-level failures, not semantic ones. It does not rewrite meaning and
it does not judge whether a reply is a good conversational move -- that would be
the harness quietly authoring the response. It checks that the output is
well-formed and consistent with the policy the harness itself chose.

At most ONE controlled retry. Unbounded reflection loops are how a small model
turns a bad reply into three bad replies and a latency problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import repetition
from .policy import Length, LENGTH_TOKENS, Policy, QuestionPolicy

AI_BOILERPLATE = re.compile(
    r"\b(as an ai|as a language model|i'?m an ai|i cannot|i can't assist|"
    r"how can i help|is there anything else|let me know if)\b", re.I)
SPECIAL_TOKEN = re.compile(r"<\|[a-z_0-9]+\|>")


@dataclass
class Verdict:
    ok: bool
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "failures": self.failures}


def validate(reply: str, *, policy: Policy | None, n_tokens: int,
             recent: list[str], closing: bool) -> Verdict:
    fails: list[str] = []
    text = reply.strip()

    if not text:
        fails.append("empty")
    if SPECIAL_TOKEN.search(text):
        fails.append("special_token_leak")
    if AI_BOILERPLATE.search(text):
        fails.append("ai_boilerplate")
    if repetition.is_exact_repeat(text, recent):
        fails.append("exact_repeat")
    if repetition.internal_loop(text):
        fails.append("internal_loop")

    if policy is not None:
        cap = LENGTH_TOKENS[policy.length]
        if n_tokens > cap * 2:
            fails.append(f"length_{n_tokens}>{cap * 2}")
        if policy.question is QuestionPolicy.NO_QUESTION and text.endswith("?"):
            fails.append("question_when_forbidden")
        if policy.action.value == "CLOSE" and text.endswith("?"):
            fails.append("question_while_closing")
    if closing and text.endswith("?"):
        fails.append("question_after_farewell")

    return Verdict(not fails, fails)
