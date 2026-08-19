"""TinyStories-style combinatorial conversation generator.

Three lessons from TinyStories (Eldan & Li 2023) and Phi-1, applied to small talk:

1. NARROW THE DOMAIN AND THE VOCABULARY.
   A 6.7M model cannot afford a long lexical tail. Everything here is built from a
   constrained bank of everyday conversational English.

2. COMBINATORIAL DIVERSITY INJECTION.
   TinyStories seeded each story with random content words + a structural feature,
   producing near-unlimited non-repeating data. `social_gold` has an 89% utterance
   repeat rate, which is why it shapes register but cannot teach language. Here
   every dialogue samples (topic x event x emotion x relation x acts x features),
   so the surface form almost never repeats.

3. CONSISTENCY OF REGISTER.
   Mixing DailyDialog's formal written English with texting register produced
   garbled blends. Everything generated here is one voice: lowercase-leaning,
   contracted, 3-25 word replies, friend-not-assistant.

HARD CONSTRAINT -- VALENCE CONDITIONING.
   The model's worst observed failure was answering "my grandad passed away" with
   "ugh, that's annoying". Response pools are keyed to event valence, and grief /
   bad news can NEVER draw a cheerful or dismissive reply. This is enforced
   structurally, not statistically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

from .schema import Conversation, Turn

# ---------------------------------------------------------------------------
# Constrained content banks. Everyday words only.
# ---------------------------------------------------------------------------
NAMES = ["sam", "alex", "jess", "ben", "maya", "tom", "ellie", "raj", "kim", "noah",
         "lucy", "dan", "priya", "chris", "amy", "leo", "zoe", "will", "nina", "omar",
         "cara", "finn", "hana", "jack", "mia", "theo", "ruby", "isaac", "lena", "max"]
PETS = ["dog", "cat", "puppy", "kitten", "rabbit", "hamster"]
PET_NAMES = ["biscuit", "luna", "pepper", "mango", "olive", "smokey", "bandit", "poppy",
             "waffle", "clover", "dash", "ziggy", "nala", "pip", "coco"]
PLACES = ["work", "school", "the gym", "the shop", "town", "the park", "home",
          "the office", "class", "the station", "my flat", "the cafe"]
FOODS = ["pizza", "pasta", "curry", "noodles", "toast", "soup", "tacos", "rice",
         "a sandwich", "leftovers", "cereal", "eggs"]
DRINKS = ["coffee", "tea", "water", "juice", "a smoothie"]
HOBBIES = ["running", "drawing", "guitar", "baking", "reading", "gaming", "cycling",
           "swimming", "knitting", "photography", "gardening", "piano"]
SHOWS = ["a documentary", "some comedy", "a thriller", "an old film", "a cooking show",
         "a cartoon", "a drama"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "tomorrow", "tonight", "the weekend"]
TIMES = ["6am", "7am", "9", "half ten", "noon", "2pm", "4", "6", "8pm", "late"]
JOBS = ["a nurse", "a teacher", "a barista", "a driver", "a designer", "a cleaner",
        "a chef", "a builder", "a vet nurse", "a shop assistant"]

# --- events keyed by valence ------------------------------------------------
POSITIVE_EVENTS = [
    "i got the job", "i passed my test", "i finished the thing finally",
    "i got a raise", "my sister had her baby", "i ran further than ever",
    "we won our match", "i got tickets", "the interview went well",
    "i finally slept properly", "i made bread and it worked",
    "i sold my old bike", "my plant flowered", "i got a puppy",
]
MILD_NEGATIVE_EVENTS = [
    "work was brutal", "i barely slept", "my bus never turned up",
    "i burned dinner", "my laptop died", "i lost my keys",
    "plans got cancelled", "i forgot a meeting", "it rained on my walk",
    "i failed my test", "my phone cracked", "i got soaked",
    "the shop was shut", "i missed my train",
]
HEAVY_NEGATIVE_EVENTS = [
    "my grandad passed away", "my dog died", "my nan is in hospital",
    "we lost the baby", "my mum's really ill", "my friend passed away",
    "i lost my job today", "we're breaking up", "my cat had to be put down",
]
BORED_EVENTS = [
    "im so bored", "bored out of my mind", "nothing to do", "so bored rn",
    "there's nothing on", "cant be bothered with anything",
    "kinda tired today", "honestly kinda tired today", "im knackered",
    "no energy today", "feeling flat today", "low energy today",
    "just drained honestly", "cant get going today",
]
NEUTRAL_EVENTS = [
    "i went for a walk", "just cooking", "watching something",
    "sorting laundry", "waiting for a delivery", "on the bus",
    "tidying up", "nothing much really",
]

# --- reply pools, strictly valence-keyed ------------------------------------
ACK_POSITIVE = ["oh nice!", "yesss.", "amazing.", "that's brilliant.", "get in.",
                "love that.", "no way!", "that's great.", "ayy congrats!", "solid."]
ACK_MILD_NEG = ["oof.", "ugh.", "that's rubbish.", "aw man.", "yikes.", "that sucks.",
                "no fun.", "brutal.", "annoying.", "that's rough."]
ACK_HEAVY_NEG = ["oh no.", "im so sorry.", "that's awful.", "god, im sorry.",
                 "that's really hard.", "im so sorry, truly.", "oh, that's heavy.",
                 "im here.", "that's a lot to carry."]
ACK_NEUTRAL = ["yeah?", "oh right.", "mm.", "gotcha.", "fair.", "nice one.",
               "sounds alright.", "cool."]

FOLLOW_POSITIVE = ["how'd it feel?", "tell me everything.", "when did you find out?",
                   "are you buzzing?", "what happens now?", "you must be pleased."]
FOLLOW_MILD_NEG = ["what happened?", "how come?", "long day?", "you alright?",
                   "is it sorted now?", "ugh, all day?"]
FOLLOW_HEAVY_NEG = ["how are you holding up?", "were you close?", "do you want to talk about it?",
                    "is there anyone with you?", "how's the family doing?",
                    "you don't have to talk if you'd rather not."]
FOLLOW_NEUTRAL = ["anything good?", "how's that going?", "what are you making?",
                  "nice, whereabouts?", "keeping busy then?"]

REACT_POSITIVE = ["you earned that.", "so happy for you.", "that's a proper win.",
                  "about time honestly.", "big day then."]
REACT_MILD_NEG = ["some days are just like that.", "hope tomorrow's kinder.",
                  "at least it's over.", "you deserve a sit down.", "that tracks honestly."]
REACT_HEAVY_NEG = ["take whatever time you need.", "im really sorry.",
                   "that's not something you rush.", "im thinking of you.",
                   "no right way to feel about that."]
REACT_NEUTRAL = ["sounds peaceful.", "that's a decent evening.", "nice and easy then.",
                 "can't fault that."]

CLOSERS_USER = ["ok im off", "gotta go", "talk later", "night", "right, bed",
                "catch you later", "im heading out"]
CLOSERS_ASST = ["later!", "night, rest up.", "take care.", "see ya.", "talk soon.",
                "have a good one.", "look after yourself."]
CLOSERS_ASST_HEAVY = ["take care of yourself.", "im around if you need me.",
                      "sending love.", "look after yourself, ok?"]

BACKCHANNEL = ["yeah", "mm", "true", "right", "fair", "haha", "oh", "same",
               "for real", "makes sense", "i getcha"]
UNKNOWN_REPLIES = ["honestly no idea.", "no clue, sorry.", "beats me lol.",
                   "never heard of it.", "not a scooby.", "dunno, what is it?"]
OBSCURE_QS = ["what year was the {} invented", "who designed the first {}",
              "how does a {} actually work", "what's the tallest {} in the world"]
OBSCURE_NOUNS = ["telegraph", "zeppelin", "submarine", "lighthouse", "windmill",
                 "typewriter", "aqueduct", "kiln", "loom", "sextant"]

VALENCE_POOLS = {
    "pos": (ACK_POSITIVE, FOLLOW_POSITIVE, REACT_POSITIVE),
    "mild": (ACK_MILD_NEG, FOLLOW_MILD_NEG, REACT_MILD_NEG),
    "heavy": (ACK_HEAVY_NEG, FOLLOW_HEAVY_NEG, REACT_HEAVY_NEG),
    "neu": (ACK_NEUTRAL, FOLLOW_NEUTRAL, REACT_NEUTRAL),
}
EVENTS_BY_VALENCE = {
    "pos": POSITIVE_EVENTS, "mild": MILD_NEGATIVE_EVENTS,
    "heavy": HEAVY_NEGATIVE_EVENTS, "neu": NEUTRAL_EVENTS,
    "bored": BORED_EVENTS,
}
OPENERS_USER = ["hey", "hi", "yo", "hey you", "heyy", "morning", "evening",
                "hey, you about?", "hiya", "you there?"]
OPENERS_ASST = ["hey", "hey, what's up?", "yo", "hiya", "hey! how's things?",
                "hey, you good?", "oh hey", "hey, what's new?"]



# ---------------------------------------------------------------------------
# Diversity expansion (round 3).
# Measured problem: exact-utterance repeat fraction was 0.97 -- WORSE than
# social_gold's 0.89 -- so the model recited "you deserve a sit down" 4 times in
# 8 samples. Fix: compose replies from independent slots (opener x core x tail)
# instead of drawing whole strings, which multiplies surface forms instead of
# adding them. Also adds boredom handling, which was entirely missing and caused
# "im so bored" -> "yeah?" degeneration 5 times in 8 samples.
# ---------------------------------------------------------------------------
OPEN_FILLER = ["", "", "", "ah, ", "oh, ", "hm, ", "right, ", "ok so ", "honestly ",
               "aw, ", "man, ", "ooh, ", "well, ", "wait, "]
TAIL_FILLER = ["", "", "", "", " honestly.", " tbh.", " though.", " lol.", " haha.",
               " ngl.", " for real.", " i reckon.", " or nah?", " right?"]

BORED_USER = ["im so bored", "nothing to do", "bored out of my mind",
              "there's nothing on", "cant be bothered with anything",
              "sat here doing nothing", "so bored rn", "nothing going on here"]
BORED_REPLIES = [
    "what've you got the energy for?", "want distraction or company?",
    "same energy here honestly", "what's the last thing you enjoyed?",
    "go stand outside for two minutes, trust me",
    "pick one: snack, walk, or nap", "boredom's just a nudge to move honestly",
    "wanna hear something daft?", "what would past you do right now?",
    "got anything half started you could poke at?",
    "even a shower counts as an event today", "tell me the most boring detail of your day",
    "i'd say snack, but i say that about everything",
    "put one song on and see what happens",
]
TOPIC_OFFERS = [
    "seen anything decent lately?", "what are you eating today?",
    "any plans this week?", "how's the sleep been?",
    "what's the weather doing there?", "listening to anything good?",
]


SOLEMN_TAILS = ["", "", "", " honestly.", " truly.", " really."]


def _vary(r: random.Random, core: str, valence: str = "neu") -> str:
    """Multiply surface forms: opener x core x tail (valence-gated)."""
    openers = OPEN_FILLER if valence not in ("heavy", "gentle") else \
        ["", "", "", "oh, ", "ah, ", "hm, "]
    out = _pick(r, openers) + core
    if r.random() < 0.30 and not core.endswith("?"):
        tails = SOLEMN_TAILS if valence in ("heavy", "gentle") else TAIL_FILLER
        out = out.rstrip(".") + _pick(r, tails)
    if r.random() < 0.12:
        out = out.replace(".", "", 1)
    return out.strip()


def partition_pool(pool: list, split: str) -> list:
    """Deterministically carve a reply pool into disjoint train/val/test thirds.

    Family tagging alone does NOT give a clean split for a template generator:
    families shared these global pools, so 62.9% of validation utterances still
    appeared verbatim in training. Splitting the POOLS is what actually makes
    held-out data held out. (For teacher-generated data this is unnecessary --
    novel language is disjoint by construction.)
    """
    if split == "all":
        return list(pool)
    idx = {"train": 0, "val": 1, "test": 2}[split]
    out = [x for i, x in enumerate(sorted(pool, key=str)) if i % 3 == idx]
    return out or list(pool)


@dataclass
class GenConfig:
    n: int = 20000
    seed: int = 0
    split: str = "all"   # train | val | test | all -- partitions the reply pools
    min_pairs: int = 3
    max_pairs: int = 9
    # Feature mix -- these are the structural "features" TinyStories injected.
    p_memory_probe: float = 0.22
    p_memory_update: float = 0.12
    p_topic_switch: float = 0.18
    p_unknown_fact: float = 0.10
    p_boundary: float = 0.08
    p_closer: float = 0.55
    valence_mix: dict = field(default_factory=lambda: {
        "pos": 0.24, "mild": 0.26, "heavy": 0.11, "neu": 0.19, "bored": 0.20})


ACK_BORED = ["oof.", "mm.", "yeah, flat one.", "ah, one of those.", "same honestly.",
             "that kind of day huh.", "low battery day.", "ugh, that mood."]
REACT_BORED = ["those days are heavy in a quiet way.", "no shame in a slow one.",
               "sometimes the day just doesn't start.", "rest counts as doing something."]
VALENCE_POOLS["bored"] = (ACK_BORED, BORED_REPLIES, REACT_BORED)


def _pick(r: random.Random, pool): return r.choice(pool)


def _fact(r: random.Random) -> tuple[str, str, str]:
    """Return (statement, question, answer) for a memory probe."""
    kind = r.choice(["pet", "name", "job", "day", "place", "hobby"])
    if kind == "pet":
        pet, nm = r.choice(PETS), r.choice(PET_NAMES)
        return (f"i got a {pet}, {nm}", f"what's my {pet} called again?", nm)
    if kind == "name":
        nm = r.choice(NAMES)
        return (f"im {nm} by the way", "what's my name again?", nm)
    if kind == "job":
        job = r.choice(JOBS)
        return (f"im {job}", "what do i do for work again?", job.split()[-1])
    if kind == "day":
        d = r.choice(DAYS)
        return (f"im going {d}", "what day was i going again?", d)
    if kind == "place":
        p = r.choice(PLACES)
        return (f"im off to {p}", "where did i say i was going?", p.split()[-1])
    h = r.choice(HOBBIES)
    return (f"i started {h}", "what did i say i started?", h)


# Once a conversation carries grief/bad news, EVERY later assistant turn must stay
# in a supportive register. Neutral small-talk pools contain "nice one" / "solid",
# which produced "my dog died" -> "nice one." -- caught by the valence audit.
GENTLE_NEUTRAL_ACK = ["yeah.", "mm.", "of course.", "right.", "i hear you.",
                      "makes sense.", "understandable.", "yeah, i get that."]
GENTLE_NEUTRAL_FOLLOW = ["how are you doing with it?", "anything i can do?",
                         "do you want to talk more, or not right now?",
                         "have you eaten today?", "is anyone with you?"]
GENTLE_NEUTRAL_REACT = ["no rush on any of it.", "that's completely fair.",
                        "take it at your own pace.", "im around either way."]
GENTLE_POOLS = (GENTLE_NEUTRAL_ACK, GENTLE_NEUTRAL_FOLLOW, GENTLE_NEUTRAL_REACT)


def _effective_valence(conv_valence: str, turn_valence: str) -> str:
    """Grief locks the conversation: never fall back to cheerful/neutral pools."""
    if conv_valence == "heavy":
        return "gentle" if turn_valence in ("neu", "pos") else "heavy"
    return turn_valence


def _asst(r: random.Random, valence: str, kind: str, conv_valence: str | None = None) -> str:
    valence = _effective_valence(conv_valence or valence, valence)
    ack, follow, react = GENTLE_POOLS if valence == "gentle" else VALENCE_POOLS[valence]
    if kind == "ack":
        return _vary(r, _pick(r, ack), valence)
    if kind == "follow":
        return _vary(r, _pick(r, follow), valence)
    if kind == "react":
        return _vary(r, _pick(r, react), valence)
    tail = _pick(r, follow) if r.random() < 0.55 else _pick(r, react)
    return _vary(r, f"{_pick(r, ack)} {tail}", valence)


def generate(cfg: GenConfig | None = None) -> Iterator[Conversation]:
    cfg = cfg or GenConfig()
    r = random.Random(cfg.seed)
    if cfg.split != "all":
        sp = cfg.split
        for _name, _pool in list(globals().items()):
            if isinstance(_pool, list) and _pool and isinstance(_pool[0], str) and _name.isupper():
                globals()[_name] = partition_pool(_pool, sp)
        for _k, _v in list(VALENCE_POOLS.items()):
            VALENCE_POOLS[_k] = tuple(partition_pool(list(x), sp) for x in _v)
    valences = list(cfg.valence_mix)
    weights = [cfg.valence_mix[v] for v in valences]

    for i in range(cfg.n):
        valence = r.choices(valences, weights=weights)[0]
        msgs: list[Turn] = []
        features: list[str] = []

        msgs.append(Turn("user", _pick(r, OPENERS_USER)))
        msgs.append(Turn("assistant", _pick(r, OPENERS_ASST)))

        # optional early fact for a later memory probe
        fact = None
        if r.random() < cfg.p_memory_probe:
            stmt, q, ans = _fact(r)
            fact = (q, ans)
            msgs.append(Turn("user", stmt))
            msgs.append(Turn("assistant", _asst(r, "neu", "ack", valence)))
            features.append("memory")

        event = _pick(r, EVENTS_BY_VALENCE[valence])
        msgs.append(Turn("user", event))
        msgs.append(Turn("assistant", _asst(r, valence, "combo", valence)))
        # heavy grief always gets a second supportive beat, never a pivot
        if valence == "heavy":
            msgs.append(Turn("user", _pick(r, ["yeah", "it's a lot", "thanks",
                                               "still processing it", "yeah it's hard"])))
            msgs.append(Turn("assistant", _asst(r, "heavy", "react", valence)))

        if r.random() < cfg.p_boundary and valence in ("mild", "heavy"):
            msgs.append(Turn("user", _pick(r, [
                "can we talk about something else",
                "don't wanna get into it",
                "not in the mood to unpack that",
                "let's skip that one",
                "i'll spare you the details",
                "rather leave that one alone",
                "can we park that for now"])))
            msgs.append(Turn("assistant", _pick(r, [
                "course, no worries.", "yeah of course, we can skip it.",
                "totally fine. what else is going on?",
                "sure. tell me something else then."])))
            features.append("boundary")

        n_pairs = r.randint(cfg.min_pairs, cfg.max_pairs)
        for _ in range(n_pairs):
            roll = r.random()
            if roll < cfg.p_topic_switch:
                topic = r.choice([f"anyway, i had {_pick(r, FOODS)} earlier",
                                  f"random but i've been doing {_pick(r, HOBBIES)}",
                                  f"oh, i watched {_pick(r, SHOWS)} last night",
                                  f"also i need to go to {_pick(r, PLACES)}"])
                msgs.append(Turn("user", topic))
                msgs.append(Turn("assistant", _asst(r, "neu", "combo", valence)))
                features.append("topic_switch")
            elif roll < cfg.p_topic_switch + cfg.p_unknown_fact and valence != "heavy":
                # No trivia detours in a grief conversation -- people don't ask
                # about zeppelins right after saying their dog died.
                q = _pick(r, OBSCURE_QS).format(_pick(r, OBSCURE_NOUNS))
                msgs.append(Turn("user", q))
                reply = _pick(r, UNKNOWN_REPLIES)
                if r.random() < 0.5:
                    reply += " " + _pick(r, ["what made you think of that?",
                                             "where'd you see it?", "why d'you ask?"])
                msgs.append(Turn("assistant", reply))
                features.append("unknown_fact")
            else:
                msgs.append(Turn("user", _pick(r, BACKCHANNEL)))
                kind = r.choices(["combo", "follow", "react", "ack"],
                                 weights=[0.5, 0.2, 0.2, 0.1])[0]
                msgs.append(Turn("assistant", _asst(r, valence if r.random() < 0.4 else "neu",
                                                    kind, valence)))

        if fact and r.random() < 0.85:
            q, ans = fact
            msgs.append(Turn("user", q))
            msgs.append(Turn("assistant", r.choice([ans, f"{ans}, right?", f"{ans} :)"])))

        if r.random() < cfg.p_closer:
            msgs.append(Turn("user", _pick(r, CLOSERS_USER)))
            closers = CLOSERS_ASST_HEAVY if valence == "heavy" else CLOSERS_ASST
            msgs.append(Turn("assistant", _pick(r, closers)))

        yield Conversation(
            id=f"combi-{i:06d}", messages=msgs, source="combi_gen",
            meta={"valence": valence, "features": features,
                  # family = generator grammar. Train/val/test split BY FAMILY so
                  # validation never shares a grammar with training (the flaw that
                  # made val_loss=0.205 meaningless).
                  "family": "combi:" + valence + ":" + ("+".join(sorted(set(features))) or "plain")},
        )
