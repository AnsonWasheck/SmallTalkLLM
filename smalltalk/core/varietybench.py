"""VarietyBench: does the model hold a conversation, or repeat one good reply?

Core-Bench scores a single turn against an accept set, so a model that emits the
single highest-scoring phrase every time scores well and is unbearable to talk
to. Measured on v0.3.1-r004: 40% of replies within a conversation were verbatim
repeats of an earlier one, while Core-Bench registered nothing.

This closes that hole. Fixed multi-turn conversations, deterministic user turns,
and three things Core-Bench cannot see:

  repeat_rate     replies identical to an earlier one in the SAME conversation
  distinct_ratio  unique replies across the whole suite
  top1_share      how much of all output is one phrase

Deliberately NOT an accuracy metric. It measures whether the model varies, not
whether it is right, and it is meant to be read alongside Core-Bench: a model
can win here by babbling, and win Core-Bench by repeating. Only a model that
does well on both is actually holding a conversation.

Trivial acknowledgements are exempt from the repeat count -- "yeah" twice in a
conversation is human, "that sounds rough" four times is not.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

VARIETY_VERSION = "varietybench-v1.0.0"

# Scripted user sides. Each stays on one topic so a varied model has to find
# several ways to stay engaged, rather than being rescued by topic changes.
CONVERSATIONS: list[tuple[str, tuple[str, ...]]] = [
    ("good_day", ("today was lovely", "same as it always is", "yeah",
                  "anyway", "mm", "it was nice though")),
    ("bad_day", ("work was awful", "my boss shouted at me", "yeah",
                 "it's been like that a while", "mm", "i'm done with it")),
    ("neutral_news", ("i got a new bike", "it's blue", "yeah",
                      "been riding it a lot", "mm", "it's good fun")),
    ("tired", ("i'm knackered", "barely slept", "yeah", "need a nap",
               "mm", "long week")),
    ("hobby", ("i started running", "did 5k this morning", "yeah",
               "trying to keep it up", "mm", "we'll see")),
    ("social", ("i saw an old friend", "we got coffee", "yeah",
                "hadn't seen them in years", "mm", "it was good")),
    ("bored", ("nothing going on today", "just sat around", "yeah",
               "might watch something", "mm", "dunno really")),
    ("plans", ("not sure about the weekend", "might go away", "yeah",
               "haven't booked anything", "mm", "we'll see how it goes")),
]

TRIVIAL = {"yeah", "mm", "right", "ok", "okay", "mhm", "ha", "haha", "sure", "yep"}
_NORM = re.compile(r"[^a-z0-9' ]+")


def normalise(t: str) -> str:
    return re.sub(r"\s+", " ", _NORM.sub(" ", t.lower())).strip()


def checksum() -> str:
    h = hashlib.sha256(VARIETY_VERSION.encode())
    for name, turns in CONVERSATIONS:
        h.update(name.encode())
        h.update("|".join(turns).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def score(transcripts: dict[str, list[str]]) -> dict:
    """transcripts: conversation name -> the model's replies, in order."""
    repeats = turns = 0
    all_replies: list[str] = []
    per_conv = {}
    for name, replies in transcripts.items():
        seen: set[str] = set()
        r_here = 0
        for r in replies:
            n = normalise(r)
            all_replies.append(n)
            turns += 1
            if n in seen and n not in TRIVIAL:
                r_here += 1
            seen.add(n)
        repeats += r_here
        per_conv[name] = r_here / max(1, len(replies))

    from collections import Counter
    c = Counter(all_replies)
    return {
        "checksum": checksum(),
        "repeat_rate": repeats / max(1, turns),
        "distinct_ratio": len(c) / max(1, len(all_replies)),
        "top1_share": c.most_common(1)[0][1] / max(1, len(all_replies)) if c else 0.0,
        "per_conversation": per_conv,
    }


def freeze(path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cs = checksum()
    p.write_text(json.dumps({"version": VARIETY_VERSION, "checksum": cs,
                             "conversations": len(CONVERSATIONS)}, indent=2))
    return cs


def verify_frozen(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run freeze() before comparing runs")
    if json.loads(p.read_text())["checksum"] != checksum():
        raise RuntimeError("VarietyBench drifted; earlier scores are not comparable")
