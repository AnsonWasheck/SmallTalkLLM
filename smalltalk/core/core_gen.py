"""Generate the Core curriculum: broad input surface, narrow output behaviour.

Two deliberate asymmetries:

  * INPUT is noised aggressively (case, punctuation, fillers, typos, a leading
    real turn for context) so the model must map *meaning* to response.
  * OUTPUT is not noised at all. Every example for an intent emits the same
    canonical target, prefixed by its length-policy token. This is the opposite
    of combi_gen/splice_gen and is intentional: at 6.7M we want P(y|x) peaked.

Frequency follows real conversational usage, not uniform coverage: greetings and
how-are-you are oversampled because they are what a user actually types first.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from ..data.schema import Conversation, Turn
from .intents import CORE_VERSION, INTENTS, Intent, normalise

# Every surface Core-Bench will probe with, including its fixed variants.
HELD_OUT_SURFACES: frozenset[str] = frozenset(
    normalise(v(p)) for i in INTENTS for p in i.held_out
    for v in (lambda s: s, lambda s: s.rstrip("?!."), lambda s: s.upper(),
              lambda s: "so " + s, lambda s: s + " lol")
)

# how often each intent appears, relative. Trivial openers dominate.
WEIGHTS = {
    "greeting": 10.0, "greeting_how_are_you": 9.0, "how_are_you": 8.0,
    "thanks": 5.0, "goodbye": 5.0, "apology": 3.0,
    "good_news": 4.0, "bad_news": 4.0, "bored": 3.0, "tired": 3.0,
    "confused": 3.0, "agreement": 3.0, "disagreement": 2.0,
    "user_vents": 3.0, "user_jokes": 3.0, "user_asks_opinion": 2.0,
    "greeting_plus_name": 2.0, "check_in": 2.0, "compliment": 2.0,
    "small_plan": 3.0,
    # weighted like a common intent: declining is a skill, not an edge case
    "out_of_scope": 6.0,
    # The single most common thing a real person says. Measured: 99.3% of real
    # user turns match no other intent. Weighted at the top of the table because
    # under-weighting it is what taught the model to force-fit every ordinary
    # statement into the nearest reflex ("i adopted a cat" -> "oh no, i'm sorry").
    "topic_statement": 12.0,
}

# Intents that only occur mid-conversation in real use.
CONTEXT_HEAVY = {"goodbye", "check_in", "thanks", "agreement", "disagreement",
                 "confused", "user_asks_opinion", "compliment"}

_FILLERS = ["", "", "", "", "so ", "ok so ", "hey ", "right ", "anyway ", "um "]
_TAILS = ["", "", "", "", " lol", " haha", " though", " tbh", " :)", "!"]


def _noise(text: str, r: random.Random) -> str:
    """Surface-level input variation. Never applied to targets."""
    t = text
    if r.random() < 0.30:
        t = r.choice(_FILLERS) + t
    if r.random() < 0.25:
        t = t + r.choice(_TAILS)
    if r.random() < 0.15:                       # drop terminal punctuation
        t = t.rstrip("?!.")
    if r.random() < 0.10:                       # shout
        t = t.upper()
    elif r.random() < 0.12:                     # capitalise properly
        t = t[:1].upper() + t[1:]
    if r.random() < 0.06 and len(t) > 5:        # single-char typo
        i = r.randrange(1, len(t) - 1)
        t = t[:i] + t[i] + t[i:]
    return t


@dataclass
class CoreConfig:
    n: int = 60000
    seed: int = 7
    emit_length_token: bool = True
    # Raised from 0.35 after the context axis was first measured. 65% of the
    # curriculum was turn-1 only, and the benchmark showed reflexes degrading
    # sharply once a preamble exists -- goodbye fell 76.2% -> 35.7%, the single
    # largest effect in the run. A reflex trained only as an opener is not a
    # reliable reflex; real conversations reach "i'm heading out" at turn nine.
    context_frac: float = 0.65      # fraction that get 1-3 preceding real turns
    max_context_turns: int = 3
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))


def _context_pool(convs: Sequence[Conversation]) -> list[tuple[str, str]]:
    """Real (user, assistant) pairs, minus any that collide with a bench surface.

    Real dialogue genuinely contains "take care" and "what's new", so without this
    filter roughly 1 in 2000 generated examples would hand the model a Core-Bench
    prompt with someone else's answer attached -- leakage, and mislabelled leakage
    at that. Measured: 27/60000 before the filter.
    """
    from ..data.splice_gen import _load_real_turn_pool
    if not convs:
        return []
    return [(u, a) for u, a in _load_real_turn_pool(convs)
            if normalise(u) not in HELD_OUT_SURFACES]


def target_for(intent: Intent, r: random.Random, length_token: bool) -> str:
    t = r.choice(intent.targets)
    return f"<|len_{intent.length}|> {t}" if length_token else t


def generate(cfg: CoreConfig | None = None,
             real_convs: Sequence[Conversation] = ()) -> Iterator[Conversation]:
    cfg = cfg or CoreConfig()
    r = random.Random(cfg.seed)
    pool = _context_pool(real_convs)

    names = [i.name for i in INTENTS]
    weights = [cfg.weights.get(n, 1.0) for n in names]
    by_name = {i.name: i for i in INTENTS}

    for k in range(cfg.n):
        intent = by_name[r.choices(names, weights=weights)[0]]
        msgs: list[Turn] = []

        # Optional preceding real exchange. The intent turn must still dominate
        # the decision, so context is short and never contains another Core cue.
        # Some intents are inherently mid-conversation: nobody opens a chat with
        # "i'm heading out". Training those as turn-1 examples teaches a reflex
        # that fires in a position it will never occupy in use.
        frac = 0.9 if intent.name in CONTEXT_HEAVY else cfg.context_frac
        if pool and r.random() < frac:
            for _ in range(r.randint(1, cfg.max_context_turns)):
                u, a = r.choice(pool)
                msgs.append(Turn("user", u))
                msgs.append(Turn("assistant", a))

        # Noise can accidentally reconstruct a held-out prompt ("so " + "so bored"),
        # which would silently turn Core-Bench into a lookup test. Reject and retry.
        for _ in range(8):
            probe = _noise(r.choice(intent.train), r)
            if normalise(probe) not in HELD_OUT_SURFACES:
                break
        else:
            probe = r.choice(intent.train)
        msgs.append(Turn("user", probe))
        msgs.append(Turn("assistant", target_for(intent, r, cfg.emit_length_token)))

        yield Conversation(
            id=f"core-{k:06d}",
            messages=msgs,
            source="core_gen",
            meta={"family": f"core:{intent.name}", "skill": intent.name,
                  "length": intent.length, "tier": intent.tier,
                  "core_version": CORE_VERSION},
        )
