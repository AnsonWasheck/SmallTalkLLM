"""StateBench-v1: does a short-horizon conversational state survive several turns?

Core-Bench answers "given this user turn, is the reply appropriate?". That is a
single-turn question, and a model that ignores history entirely can score well on
it. StateBench answers a different question:

    Given a conversation whose state was established two turns ago, and a probe
    turn that is deliberately uninformative on its own, does the model still
    behave as though it knows the state?

The load-bearing construct is the COUNTERFACTUAL PAIR. Two trajectories are
identical in their probe turn and differ in exactly one state variable:

    A: "today was great"  ... "yeah"  -> should continue positively
    B: "today was awful"  ... "yeah"  -> should continue sympathetically

The probe tokens are byte-identical. A model that conditions only on the last
user turn MUST emit the same reply for both, and therefore scores 0.0 on
`divergence` no matter how good its reflexes are. That single number is the
instrument this project was missing: it cannot be satisfied by better
last-turn classification.

Everything here is deterministic -- no RNG -- and frozen by checksum, on the same
terms as Core-Bench. Held-out probes never appear in any generator.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

STATE_VERSION = "statebench-v1.0.0"

_NORM = re.compile(r"[^a-z0-9' ]+")


def normalise(text: str) -> str:
    t = _NORM.sub(" ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


# Replies that are appropriate ONLY when the established valence is positive /
# negative. Deliberately generous: we are measuring whether the model tracks
# state, not whether it picks our favourite phrasing.
POSITIVE_OK = [
    "that's great", "that's great!", "nice", "oh nice", "good stuff", "glad to hear",
    "that's good", "nice one", "lovely", "happy for you", "that's brilliant",
    "sounds good", "can't complain then", "long may it last", "yeah nice",
    "that's lovely", "amazing", "congrats", "good to hear", "yeah exactly",
]
NEGATIVE_OK = [
    "that sounds rough", "that's rough", "sorry to hear that", "oh no, i'm sorry",
    "that sucks", "that's rubbish", "ugh, that's rough", "sorry about that",
    "hopefully tomorrow's better", "that's hard", "sounds exhausting",
    "hope it gets better", "i'm sorry", "oh no", "that's a shame", "rough one",
]
# Neutral continuations are acceptable under EITHER valence: they commit to
# nothing. Counted separately, because a model that only ever plays it safe is
# not tracking state either -- it is declining to.
NEUTRAL_OK = [
    "yeah", "mm", "right", "i see", "fair enough", "makes sense", "yeah?",
    "go on", "how come", "what happened", "oh really", "mm yeah",
]

# The generic high-prior replies both checkpoints collapse into under
# uncertainty. Measured at 25.6% of all assistant turns in the v0.2 corpus.
ATTRACTORS = [
    "no worries", "yeah exactly", "what do you mean?", "same, nothing going on here",
    "hey", "not bad, you?", "i'm good, how about you?", "oh nice, how's that going?",
    "no idea honestly", "see you", "haha",
]


@dataclass(frozen=True)
class Trajectory:
    """One side of a counterfactual pair.

    `turns` is the scripted history (alternating user/assistant, starting user).
    `probe` is the final user turn -- uninformative alone, by construction.
    """
    pair_id: str
    valence: str                       # positive | negative
    topic: str
    turns: tuple[str, ...]
    probe: str

    @property
    def id(self) -> str:
        return f"{self.pair_id}::{self.valence}::{normalise(self.probe)}"


def _pair(pair_id: str, topic: str, pos_open: str, neg_open: str,
          mid_user: str, probe: str,
          pos_assist: str = "oh yeah?", neg_assist: str = "oh no") -> list[Trajectory]:
    """Build a matched pair differing only in valence.

    The assistant turns in the scripted history are neutral and identical across
    both sides wherever possible, so the ONLY signal distinguishing them is the
    user's own stated valence. If the model needs the assistant's prior reply to
    infer valence, it is leaning on its own output rather than on the user.
    """
    return [
        Trajectory(pair_id, "positive", topic,
                   (pos_open, pos_assist, mid_user, "mm"), probe),
        Trajectory(pair_id, "negative", topic,
                   (neg_open, neg_assist, mid_user, "mm"), probe),
    ]


# 24 counterfactual pairs across 8 topics. Probes are backchannels: "yeah",
# "i guess", "mm" -- these carry no valence whatsoever, which is the point.
PAIRS: list[list[Trajectory]] = [
    _pair("day-01", "day", "today was great", "today was awful",
          "just how it went really", "yeah"),
    _pair("day-02", "day", "i had a really good day", "i had a really bad day",
          "same as usual otherwise", "i guess"),
    _pair("work-01", "work", "work went really well today", "work was a nightmare today",
          "it's been like that all week", "yeah"),
    _pair("work-02", "work", "i finally finished the project", "i completely messed up the project",
          "been working on it for months", "mm"),
    _pair("sleep-01", "sleep", "i slept really well last night", "i barely slept last night",
          "went to bed about eleven", "yeah"),
    _pair("health-01", "health", "i'm feeling much better now", "i'm feeling really rough now",
          "been like this a few days", "mm"),
    _pair("social-01", "social", "i had a lovely time with them", "i had a horrible time with them",
          "we were there about an hour", "yeah"),
    _pair("news-01", "news", "they offered me the position", "they passed on me",
          "they told me this morning", "yeah"),
    _pair("news-02", "news", "the results came back clear", "the results came back bad",
          "i've been waiting weeks", "mm"),
    _pair("home-01", "home", "the flat's finally sorted", "the flat's a complete mess",
          "been dealing with it all month", "yeah"),
    _pair("money-01", "money", "i got a raise", "i got a pay cut",
          "found out on friday", "i guess"),
    _pair("study-01", "study", "i scraped through the exam", "i bombed the exam",
          "it was the third one this term", "yeah"),
]

# Held-out probe surfaces used ONLY here. Any generator emitting these would be
# leaking the test; the leakage gate checks for them.
HELD_OUT_PROBES = ["yeah", "i guess", "mm"]


def trajectories() -> list[Trajectory]:
    return [t for pair in PAIRS for t in pair]


def pairs() -> list[tuple[Trajectory, Trajectory]]:
    return [(p[0], p[1]) for p in PAIRS]


def classify(reply: str) -> str:
    """positive | negative | neutral | attractor | other."""
    n = normalise(reply)
    if n in {normalise(a) for a in ATTRACTORS}:
        return "attractor"
    if n in {normalise(a) for a in POSITIVE_OK}:
        return "positive"
    if n in {normalise(a) for a in NEGATIVE_OK}:
        return "negative"
    if n in {normalise(a) for a in NEUTRAL_OK}:
        return "neutral"
    return "other"


def is_correct(traj: Trajectory, reply: str) -> bool:
    """Committing to the RIGHT valence. Neutral is not correct -- it is a decline."""
    return classify(reply) == traj.valence


def checksum() -> str:
    h = hashlib.sha256(STATE_VERSION.encode())
    for t in trajectories():
        h.update(t.id.encode())
        h.update("|".join(t.turns).encode())
        h.update(b"\0")
    for bank in (POSITIVE_OK, NEGATIVE_OK, NEUTRAL_OK, ATTRACTORS):
        h.update(json.dumps(sorted(normalise(x) for x in bank)).encode())
    return h.hexdigest()[:16]


def freeze(path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cs = checksum()
    p.write_text(json.dumps({
        "version": STATE_VERSION, "checksum": cs,
        "pairs": len(PAIRS), "trajectories": len(trajectories()),
        "ids": [t.id for t in trajectories()],
    }, indent=2))
    return cs


def verify_frozen(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run freeze() before comparing runs")
    frozen = json.loads(p.read_text())["checksum"]
    if frozen != checksum():
        raise RuntimeError(
            f"StateBench drifted: frozen={frozen} current={checksum()}. "
            "Editing trajectories invalidates every previously reported score."
        )
