"""Frame generator: teach the SYNTAX of elaboration, not a table of topics.

The measured problem
--------------------
Every reply in every corpus this project has built is a closed stock phrase.
ElaborationBench v2 scores all three checkpoints at 0.0% elaboration: not one
reply anywhere reuses a content word the user just said. The model has no
sentence frame with a slot for a noun, which is why biasing generation toward the
referent produces "dog dog dog dog dog?" rather than "what's the dog called?" --
there is nowhere in its learned syntax for the noun to go.

The design
----------
A CLOSED set of frames, an OPEN set of nouns.

  frames  ~6 conversational moves, each a sentence shape with a referent slot
  nouns   hundreds per slot, drawn from open banks and split train/held-out

The frame is small enough to learn. The noun is impossible to memorise: a model
cannot store the right follow-up for nine hundred nouns in 6.7M parameters, so
the only learnable rule is "reuse the thing they said in this shape". That is
the same argument that broke slot-memorisation in v0.1, applied to syntax
instead of facts.

Held-out nouns never appear in training, so generalisation to unseen subjects is
directly measurable rather than assumed. ElaborationBench prompts are blocked
outright.

What this deliberately does NOT do
----------------------------------
It does not teach topic knowledge. "what's the dog called?" requires no
understanding of dogs -- only the recognition that an object was acquired and
that objects have names. Topic knowledge does not fit in this model and is not
the goal; reusing the referent is what makes a reply feel understood.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterator

from ..data.schema import Conversation, Turn

FRAME_GEN_VERSION = "frame-gen-v0.4.0"

# --- open noun banks --------------------------------------------------------
OBJECTS = """bike car laptop phone kettle sofa desk chair bed lamp fridge oven
mirror rug curtain shelf mattress printer camera speaker keyboard monitor guitar
piano violin drum radio watch ring necklace jacket coat boots trainers backpack
suitcase tent kayak surfboard skateboard scooter mower shed fence gate greenhouse
aquarium telescope typewriter blender toaster microwave dishwasher heater fan
projector console controller headset router doorbell""".split()

PETS = """dog cat puppy kitten rabbit hamster parrot budgie goldfish terrapin
gecko snake ferret guinea tortoise pony donkey chicken duck""".split()

JOBS = """nurse teacher plumber electrician chef baker driver mechanic engineer
accountant designer librarian gardener joiner painter roofer welder florist
optician dentist vet pharmacist paramedic firefighter surveyor architect
translator editor photographer barber tailor butcher farmer courier""".split()

PLACES = """italy spain portugal norway iceland morocco japan peru canada wales
scotland cornwall brighton glasgow dublin lisbon prague vienna krakow seville
kyoto osaka montreal boston austin denver perth adelaide galway inverness""".split()

ACTIVITIES = """running swimming cycling climbing baking pottery knitting
gardening painting drawing woodwork sewing yoga boxing rowing fishing hiking
birdwatching photography chess bouldering kayaking archery fencing skating""".split()

RELATIONS = """sister brother cousin nephew niece aunt uncle mum dad flatmate
neighbour colleague friend partner grandad grandma godmother""".split()

BANKS = {"object": OBJECTS, "pet": PETS, "job": JOBS, "place": PLACES,
         "activity": ACTIVITIES, "relation": RELATIONS}


@dataclass(frozen=True)
class Frame:
    """One conversational move: how the user raises it, how the reply reuses it."""
    name: str
    slot: str
    user: tuple[str, ...]
    reply: tuple[str, ...]        # every one of these MUST contain {n}

    # NOTE: user and reply templates are cross-producted, so EVERY reply must
    # make sense after EVERY user form in the same frame. An earlier version
    # paired "my partner called earlier" with "how long is your partner
    # staying?", which is exactly the kind of near-miss that teaches a model to
    # produce fluent nonsense.


FRAMES: list[Frame] = [
    Frame("acquired", "object",
          ("i got a new {n}", "just picked up a {n}", "i bought a {n} yesterday",
           "we got a {n} at the weekend", "i've finally got a {n}"),
          ("what kind of {n}?", "how's the {n}?", "what made you go for a {n}?",
           "is the {n} any good?", "how much was the {n}?",
           "what's the {n} like?")),
    Frame("pet", "pet",
          ("i got a {n}", "we've got a new {n}", "my {n} arrived last week",
           "we adopted a {n}"),
          ("what's the {n} called?", "how old is the {n}?",
           "where'd you get the {n}?", "is the {n} settling in?",
           "what colour's the {n}?")),
    Frame("role", "job",
          ("i'm a {n}", "i work as a {n}", "i've been a {n} for a while",
           "i trained as a {n}"),
          ("how long have you been a {n}?", "do you like being a {n}?",
           "what made you become a {n}?", "is being a {n} hard work?",
           "where do you work as a {n}?")),
    Frame("trip", "place",
          # All past-tense: "i've never been to {n}" was removed because it
          # cannot take "how was {n}?".
          ("we went to {n}", "just got back from {n}", "i was in {n} last week",
           "we spent a few days in {n}"),
          ("how was {n}?", "whereabouts in {n}?", "how long were you in {n}?",
           "would you go back to {n}?", "what's {n} like?")),
    Frame("pursuit", "activity",
          ("i've taken up {n}", "i started {n} recently", "i've been doing {n}",
           "i'm getting into {n}"),
          ("how long have you been {n}?", "how's the {n} going?",
           "what got you into {n}?", "is {n} hard to pick up?",
           "how often do you do {n}?")),
    Frame("person", "relation",
          # All imply the person is around, so the staying/visiting replies fit.
          ("my {n} is visiting", "my {n} is staying with us",
           "my {n} came round today", "i've got my {n} over"),
          ("how long is your {n} staying?", "how's your {n} doing?",
           "do you see your {n} often?", "is your {n} nearby?",
           "what's your {n} up to?")),
]

# Blocked outright: any subject ElaborationBench probes with. Held-out nouns are
# split separately; this is the harder guarantee that the test is never trained.
from .elaboration import SCENARIOS as _BENCH_SCENARIOS  # noqa: E402

BENCH_WORDS: frozenset[str] = frozenset(
    w for s in _BENCH_SCENARIOS for w in re.findall(r"[a-z']+", s.prompt.lower()))


def noun_split(seed: int = 5, held_out: float = 0.25
               ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split every bank into train / held-out nouns.

    Held-out nouns are the generalisation test: if the model only echoes nouns it
    was trained on, it memorised a list rather than learning the frame.
    """
    train, held = {}, {}
    for slot, bank in BANKS.items():
        pool = sorted(w for w in bank if w not in BENCH_WORDS)
        r = random.Random(seed + len(slot))
        r.shuffle(pool)
        cut = max(1, int(len(pool) * held_out))
        held[slot], train[slot] = pool[:cut], pool[cut:]
    return train, held


@dataclass
class FrameConfig:
    n: int = 30000
    seed: int = 5
    held_out: float = 0.25
    follow_turns: int = 2          # turns of ordinary chat after the elaboration
    use_held_out: bool = False     # True only for building a probe set


ACKS = ("mm", "right", "i see", "yeah?", "oh nice", "fair enough")
CLOSERS = ("that's good", "nice one", "sounds alright", "fair enough", "mm, nice")


def generate(cfg: FrameConfig | None = None) -> Iterator[Conversation]:
    cfg = cfg or FrameConfig()
    train, held = noun_split(cfg.seed, cfg.held_out)
    nouns = held if cfg.use_held_out else train
    r = random.Random(cfg.seed)

    for i in range(cfg.n):
        f = r.choice(FRAMES)
        pool = nouns[f.slot]
        n = r.choice(pool)
        msgs: list[Turn] = []
        msgs.append(Turn("user", r.choice(f.user).format(n=n)))
        # The elaboration: the reply reuses the referent. Sampled without reuse
        # inside a conversation so the model does not learn one frozen follow-up.
        used: set[str] = set()
        options = [t for t in f.reply if t not in used]
        target = r.choice(options)
        used.add(target)
        msgs.append(Turn("assistant", f"<|len_short|> {target.format(n=n)}"))

        for _ in range(r.randint(1, cfg.follow_turns)):
            msgs.append(Turn("user", r.choice(
                ("yeah", "it's alright", "pretty good so far", "not bad",
                 "early days", "we'll see", "mm"))))
            opts = [t for t in f.reply if t not in used]
            if opts and r.random() < 0.45:
                t = r.choice(opts)
                used.add(t)
                msgs.append(Turn("assistant", f"<|len_short|> {t.format(n=n)}"))
            else:
                msgs.append(Turn("assistant", r.choice(ACKS + CLOSERS)))

        yield Conversation(
            id=f"frame-{i:06d}",
            messages=msgs,
            source="frame_gen",
            meta={"family": f"frame:{f.name}:{n}", "skill": f"frame_{f.name}",
                  "frame": f.name, "slot": f.slot, "noun": n,
                  "frame_gen_version": FRAME_GEN_VERSION},
        )
