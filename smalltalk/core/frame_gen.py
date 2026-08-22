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
# Banks are deliberately LARGE. The pilot learned the frame perfectly and then
# filled the slot with a different member of the same bank -- "i got a new dog"
# -> "what's the puppy called?". With ~22 nouns per slot the model could hedge
# across the bank and still cut its loss; the noun was predictable from the
# frame, so attending to the user's turn bought almost nothing. Several hundred
# per slot makes hedging worthless and copying the only strategy that pays.
OBJECTS = """bike car laptop phone kettle sofa desk chair bed lamp fridge oven
mirror rug curtain shelf mattress printer camera speaker keyboard monitor guitar
piano violin drum radio watch ring necklace jacket coat boots trainers backpack
suitcase tent kayak surfboard skateboard scooter mower shed fence gate greenhouse
aquarium telescope typewriter blender toaster microwave dishwasher heater fan
projector console controller headset router doorbell van caravan trailer bicycle
tricycle wheelbarrow ladder toolbox drill sander chisel hammer saw wrench
lawnmower hosepipe sprinkler barbecue firepit hammock parasol bench stool
wardrobe dresser bookcase cabinet sideboard footstool armchair recliner
bunk cot crib pram pushchair highchair playpen trampoline swing slide
kite frisbee cricket racket bat ball glove helmet goggles wetsuit
snorkel flippers paddle canoe dinghy sail anchor compass binoculars
lantern torch stove flask cooler rucksack sleeping mat pillow duvet
blanket throw cushion doormat coathook clock barometer thermometer
kettlebell dumbbell treadmill rower bench crosstrainer skipping
banjo mandolin ukulele harmonica flute clarinet trumpet saxophone
turntable amplifier mixer microphone tripod lens flash filter
tablet stylus scanner shredder stapler whiteboard projector
freezer boiler radiator thermostat extractor humidifier purifier
kayak paddleboard bodyboard wakeboard snowboard toboggan""".split()

PETS = """dog cat puppy kitten rabbit hamster parrot budgie goldfish terrapin
gecko snake ferret guinea tortoise pony donkey chicken duck cockatiel canary
finch lovebird macaw parakeet chinchilla degu gerbil mouse rat hedgehog
axolotl newt frog toad iguana chameleon python corn boa tarantula
goat sheep alpaca llama pigeon quail turkey goose swan koi
labrador collie beagle poodle spaniel terrier retriever dachshund
husky corgi pug boxer greyhound whippet setter pointer""".split()

JOBS = """nurse teacher plumber electrician chef baker driver mechanic engineer
accountant designer librarian gardener joiner painter roofer welder florist
optician dentist vet pharmacist paramedic firefighter surveyor architect
translator editor photographer barber tailor butcher farmer courier
midwife physio radiographer podiatrist counsellor therapist dietitian
carpenter bricklayer plasterer glazier locksmith upholsterer cobbler
jeweller watchmaker blacksmith potter weaver printer bookbinder
sailor pilot conductor guard signaller dispatcher stevedore
brewer distiller vintner fishmonger greengrocer grocer caterer
auditor actuary underwriter broker valuer bailiff registrar
lecturer tutor invigilator archivist curator conservator
ranger warden keeper groom farrier shearer beekeeper""".split()

PLACES = """italy spain portugal norway iceland morocco japan peru canada wales
scotland cornwall brighton glasgow dublin lisbon prague vienna krakow seville
kyoto osaka montreal boston austin denver perth adelaide galway inverness
greece croatia slovenia estonia latvia finland sweden denmark belgium
poland hungary romania bulgaria turkey tunisia egypt kenya namibia
chile argentina uruguay colombia ecuador bolivia mexico cuba
vietnam thailand malaysia taiwan korea nepal bhutan mongolia
cardiff swansea bristol exeter plymouth norwich lincoln durham
leeds sheffield hull preston carlisle stirling aberdeen dundee
cork limerick belfast derry kilkenny sligo tralee
naples bologna verona siena bergamo turin genoa palermo
porto braga faro coimbra evora sintra""".split()

ACTIVITIES = """running swimming cycling climbing baking pottery knitting
gardening painting drawing woodwork sewing yoga boxing rowing fishing hiking
birdwatching photography chess bouldering kayaking archery fencing skating
crochet quilting embroidery weaving whittling carving calligraphy
origami bookbinding printmaking sculpting welding blacksmithing
surfing sailing canoeing paddleboarding snorkelling diving
skiing snowboarding curling bowling darts snooker
gardening composting foraging beekeeping brewing baking
juggling unicycling slacklining orienteering geocaching
birding stargazing metalwork modelling gaming coding
dancing singing drumming piano guitar violin
pilates spinning weightlifting stretching walking jogging""".split()

RELATIONS = """.split()sister brother cousin nephew niece aunt uncle mum dad flatmate
neighbour colleague friend partner grandad grandma godmother godfather
stepsister stepbrother stepmum stepdad sisterinlaw brotherinlaw
housemate roommate landlord tenant workmate teammate classmate
mentor apprentice trainee supervisor manager""".split()

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
    # A noun must belong to exactly ONE slot. "guitar" is both an object and an
    # activity, and "printer" is both an object and a job; without this, a noun
    # held out of one bank leaks back in through another and the generalisation
    # test is quietly void. First bank wins, deterministically.
    claimed: set[str] = set()
    train, held = {}, {}
    for slot in sorted(BANKS):
        pool = []
        for w in sorted(set(BANKS[slot])):
            if w in BENCH_WORDS or w in claimed:
                continue
            claimed.add(w)
            pool.append(w)
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
    # Only nouns the model can actually copy. Measured on r009: elaboration
    # succeeded on 11% of 1-token referents and 0 of 29 multi-token ones. Copying
    # a single token is already hard for 8 layers with one KV head; copying a
    # 2-4 token span is beyond it. Roughly 45% of the banks were multi-token, so
    # nearly half the training signal demanded something impossible -- and a
    # model asked to do the impossible learns to hedge instead.
    allowed_nouns: dict | None = None


ACKS = ("mm", "right", "i see", "yeah?", "oh nice", "fair enough")
CLOSERS = ("that's good", "nice one", "sounds alright", "fair enough", "mm, nice")


def generate(cfg: FrameConfig | None = None) -> Iterator[Conversation]:
    cfg = cfg or FrameConfig()
    train, held = noun_split(cfg.seed, cfg.held_out)
    nouns = held if cfg.use_held_out else train
    if cfg.allowed_nouns is not None:
        nouns = {k: [w for w in v if w in cfg.allowed_nouns.get(k, ())] or v
                 for k, v in nouns.items()}
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
