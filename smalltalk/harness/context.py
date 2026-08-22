"""Context selection.

The model accepts 1024 tokens. That is not a reason to send 1024 tokens. This
project has already measured that irrelevant context actively hurts -- the v0.2
corpus paired probes with unrelated preambles and the model learned to ignore
history altogether. Here we test the converse: whether a smaller, cleaner window
improves reliability at inference time.

Selection is by turn count and token budget, oldest dropped first, and always
ends on a user turn so the generation prompt is well-formed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Selection:
    messages: list[dict]
    n_turns: int
    n_tokens: int
    dropped: int

    def as_dict(self) -> dict:
        return {"n_turns": self.n_turns, "n_tokens": self.n_tokens,
                "dropped": self.dropped}


def select(history: list[dict], tokenizer, *, max_turns: int, token_budget: int,
           memory_hint: str | None = None) -> Selection:
    kept = [m for m in history if m["role"] != "system"]
    # The generation prompt assumes the conversation ends with the user. In normal
    # operation reply() appends the user turn before selecting, but a caller can
    # hand us history ending on an assistant turn, and silently building a prompt
    # that asks the model to follow its own last reply would be a subtle bug.
    while kept and kept[-1]["role"] != "user":
        kept.pop()
    dropped = max(0, len(kept) - max_turns)
    kept = kept[-max_turns:] if max_turns > 0 else kept

    # Drop from the front until the budget is met. Never drop the final user
    # turn: without it there is nothing to reply to.
    while len(kept) > 1:
        ids, _ = tokenizer.encode_conversation(kept, add_bos=True,
                                               add_generation_prompt=True)
        if len(ids) <= token_budget:
            break
        kept = kept[1:]
        dropped += 1

    msgs = list(kept)
    if memory_hint:
        msgs = [{"role": "system", "content": memory_hint}] + msgs

    ids, _ = tokenizer.encode_conversation(msgs, add_bos=True,
                                           add_generation_prompt=True)
    return Selection(msgs, len(kept), len(ids), dropped)
