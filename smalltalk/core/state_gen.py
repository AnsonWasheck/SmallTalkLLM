"""v0.3-Core-State: trajectories where the answer DEPENDS on conversational state.

Why this replaces the v0.2 context mechanism
--------------------------------------------
v0.2 built multi-turn examples by prepending randomly sampled, unrelated real
dialogue to a probe. Measured: in 0% of those examples did the correct answer
depend on anything before the last user turn. The loss-minimising solution is
therefore to ignore history entirely, and the model duly learned that. StateBench
confirmed it: on 12 counterfactual pairs whose probe turns are byte-identical and
whose established valence is opposite, the r004 checkpoint produced the SAME
reply for both sides in 11 of 12 cases, and scored 0.0% directional.

Here, every trajectory is emitted as a COUNTERFACTUAL PAIR: two conversations
identical in structure and identical at the probe turn, differing in exactly one
state variable. The probe is a bare backchannel ("yeah", "mm", "i guess") that
carries no valence of its own. To fit both sides, the model has no option but to
condition on the earlier turns -- ignoring them costs loss on one side of every
pair, which is the gradient signal that was previously absent.

Three trajectory shapes:

  sustained   valence established, held for the whole conversation
  corrected   valence established, then explicitly reversed mid-conversation
  drifted     topic changes but valence persists (tests which variable is tracked)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from ..data.schema import Conversation, Turn

STATE_GEN_VERSION = "state-gen-v0.3.0"

# ---------------------------------------------------------------------------
# Openers: what establishes the state. Broad paraphrase coverage per (topic,
# valence) so the model must learn the semantic class, not the surface strings.
# ---------------------------------------------------------------------------
OPENERS: dict[str, dict[str, list[str]]] = {
    "day": {
        "positive": ["today was lovely", "had a really nice day", "today went well",
                     "it's been a good one", "today's been great honestly",
                     "nice day today", "today turned out well", "had a good one today"],
        "negative": ["today was miserable", "had a really bad day", "today went badly",
                     "it's been a rough one", "today's been grim honestly",
                     "horrible day today", "today turned out badly", "had a bad one today"],
    },
    "work": {
        "positive": ["work went smoothly", "work's been good lately", "had a good shift",
                     "work was actually fine", "things went well at work",
                     "my shift flew by", "work's been easy this week"],
        "negative": ["work was chaos", "work's been awful lately", "had a horrible shift",
                     "work was a disaster", "things went badly at work",
                     "my shift dragged forever", "work's been brutal this week"],
    },
    "sleep": {
        "positive": ["slept like a log", "i actually slept well", "got a full night finally",
                     "had a proper sleep", "slept right through"],
        "negative": ["hardly slept a wink", "i slept terribly", "was up half the night",
                     "couldn't get to sleep", "kept waking up all night"],
    },
    "health": {
        "positive": ["i'm feeling loads better", "finally shaken it off",
                     "back on my feet now", "feeling much healthier"],
        "negative": ["i'm feeling awful", "still can't shake it", "come down with something",
                     "feeling really run down"],
    },
    "social": {
        "positive": ["it was lovely seeing them", "we had a great time",
                     "really enjoyed catching up", "they were on good form"],
        "negative": ["it was awkward seeing them", "we had a miserable time",
                     "the catch up was painful", "they were in a foul mood"],
    },
    "news": {
        "positive": ["i heard back and it's good", "the news was good",
                     "it went through in the end", "they accepted it"],
        "negative": ["i heard back and it's bad", "the news was bad",
                     "it fell through in the end", "they turned it down"],
    },
    "home": {
        "positive": ["the place is finally tidy", "got the house sorted",
                     "the flat's coming together"],
        "negative": ["the place is a tip", "the house is falling apart",
                     "the flat's a disaster"],
    },
    "money": {
        "positive": ["money's been alright lately", "finally in the black",
                     "the budget's working out"],
        "negative": ["money's been tight lately", "i'm in the red again",
                     "the budget's blown"],
    },
    "study": {
        "positive": ["the course is going well", "i'm on top of the coursework",
                     "revision's actually going fine"],
        "negative": ["the course is going badly", "i'm drowning in coursework",
                     "revision's going nowhere"],
    },
    "travel": {
        "positive": ["the trip was brilliant", "the journey was smooth",
                     "the holiday was perfect"],
        "negative": ["the trip was a nightmare", "the journey was hellish",
                     "the holiday was ruined"],
    },
}

# Neutral middles: carry the conversation without carrying valence. These are
# what make the probe genuinely uninformative -- the state has to be held across
# them rather than re-derived from the immediately preceding turn.
MIDDLES = [
    "it's been like that a while", "same as it always is", "that's just how it went",
    "been going on a few days now", "not much i can do about it",
    "it is what it is really", "anyway, that's the situation",
    "been thinking about it all day", "it's the same every time",
    "we'll see how it goes", "that's about the size of it",
    "nothing much has changed", "it's been going on since monday",
]

# Assistant turns inside the scripted middle. Neutral on purpose: if the model
# could read valence off its OWN prior reply it would not need the user's turn.
NEUTRAL_ACKS = ["mm", "right", "i see", "yeah?", "go on", "mhm", "okay", "fair enough"]

# The probe: bare backchannels with no valence content whatsoever.
PROBES = ["yeah", "mm", "i guess", "yeah i know", "suppose so", "mm yeah", "right"]

# Valence-committed continuations. Narrow (the design rule), but conditioned on
# STATE rather than on the last turn -- which is the whole point of v0.3.
CONTINUE = {
    "positive": ["that's great", "nice one", "glad to hear it", "long may it last",
                 "can't complain then", "good stuff"],
    "negative": ["that sounds rough", "hopefully tomorrow's better", "that's rubbish",
                 "sorry to hear that", "that sucks", "hope it picks up"],
}

# Openers used by StateBench. The generator must never emit these or the test
# leaks. Checked at module import and again by the corpus builder.
from .statebench import PAIRS as _BENCH_PAIRS  # noqa: E402

BENCH_SURFACES: frozenset[str] = frozenset(
    re.sub(r"[^a-z0-9' ]+", " ", t.turns[0].lower()).strip()
    for pair in _BENCH_PAIRS for t in pair
)

_FLIP = {"positive": "negative", "negative": "positive"}

CORRECTIONS = {
    "positive": ["actually no, it was awful", "wait, i'm making it sound better than it was",
                 "honestly though it was bad", "scratch that, it was rough"],
    "negative": ["actually it wasn't that bad", "wait, it did get better",
                 "honestly though it turned out fine", "scratch that, it was alright"],
}


@dataclass
class StateConfig:
    n: int = 40000
    seed: int = 11
    shapes: dict = field(default_factory=lambda: {
        "sustained": 0.55, "corrected": 0.25, "drifted": 0.20})
    min_probes: int = 1
    max_probes: int = 3
    emit_length_token: bool = True


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", t.lower())).strip()


def _target(valence: str, r: random.Random, length_token: bool) -> str:
    t = r.choice(CONTINUE[valence])
    return f"<|len_short|> {t}" if length_token else t


def _build(shape: str, topic: str, valence: str, r: random.Random,
           cfg: StateConfig) -> tuple[list[Turn], str]:
    """Return (messages, final_valence). Deterministic given `r`."""
    msgs: list[Turn] = []
    opener = r.choice(OPENERS[topic][valence])
    msgs.append(Turn("user", opener))
    msgs.append(Turn("assistant", r.choice(NEUTRAL_ACKS)))

    current = valence
    n_mid = r.randint(1, 2)
    for _ in range(n_mid):
        msgs.append(Turn("user", r.choice(MIDDLES)))
        msgs.append(Turn("assistant", r.choice(NEUTRAL_ACKS)))

    if shape == "corrected":
        msgs.append(Turn("user", r.choice(CORRECTIONS[current])))
        msgs.append(Turn("assistant", r.choice(NEUTRAL_ACKS)))
        current = _FLIP[current]
        msgs.append(Turn("user", r.choice(MIDDLES)))
        msgs.append(Turn("assistant", r.choice(NEUTRAL_ACKS)))
    elif shape == "drifted":
        # Topic moves, valence persists. Tests WHICH variable is being tracked:
        # a model keying on topic words rather than on state will fail here.
        other = r.choice([t for t in OPENERS if t != topic])
        msgs.append(Turn("user", r.choice(OPENERS[other][current])))
        msgs.append(Turn("assistant", r.choice(NEUTRAL_ACKS)))

    # One or more probes. Each is a bare backchannel; each requires the state.
    for _ in range(r.randint(cfg.min_probes, cfg.max_probes)):
        msgs.append(Turn("user", r.choice(PROBES)))
        msgs.append(Turn("assistant", _target(current, r, cfg.emit_length_token)))
        if r.random() < 0.5:
            msgs.append(Turn("user", r.choice(MIDDLES)))
            msgs.append(Turn("assistant", r.choice(NEUTRAL_ACKS)))

    return msgs, current


def generate(cfg: StateConfig | None = None) -> Iterator[Conversation]:
    """Emit counterfactual PAIRS. Both sides share a family so they never split."""
    cfg = cfg or StateConfig()
    shapes = list(cfg.shapes)
    weights = [cfg.shapes[s] for s in shapes]
    topics = list(OPENERS)

    emitted = 0
    pair_idx = 0
    while emitted < cfg.n:
        # The seed is keyed on the PAIR, not on the emitted-conversation counter.
        # Seeding per conversation advanced the RNG between the two sides, so the
        # probe turns and neutral filler diverged and the pair stopped being
        # minimal -- which would have let the model distinguish the sides by
        # something other than the state variable under test.
        shape = random.Random(cfg.seed + pair_idx).choices(shapes, weights=weights)[0]
        topic = random.Random(cfg.seed + pair_idx * 7).choice(topics)
        fam = f"state:{topic}:{shape}:{pair_idx:05d}"

        built = []
        for valence in ("positive", "negative"):
            r = random.Random(cfg.seed * 31 + pair_idx)
            built.append((valence,) + _build(shape, topic, valence, r, cfg))

        pair_idx += 1
        if any(_norm(m[1][0].content) in BENCH_SURFACES for m in built):
            continue

        for valence, msgs, final in built:
            yield Conversation(
                id=f"state-{pair_idx - 1:06d}-{valence[:3]}",
                messages=msgs,
                source="state_gen",
                meta={"family": fam, "skill": f"state_{shape}", "topic": topic,
                      "valence": valence, "final_valence": final, "shape": shape,
                      "state_gen_version": STATE_GEN_VERSION},
            )
            emitted += 1
