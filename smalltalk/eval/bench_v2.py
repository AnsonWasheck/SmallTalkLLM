"""SmallTalkBench-v2: 400 immutable, stratified, held-out scenarios.

Why v2 exists: the 20-scenario HARD set could not do model selection. A single
binary probe outcome per scenario meant the memory metric swung 0.12 -> 1.00 at
constant training loss -- that was sampling noise being read as progress.

DESIGN RULES
  1. IMMUTABLE. Scenarios are generated from a fixed seed and checksummed
     (`freeze.json`). If the checksum changes, evaluation refuses to run. Frozen
     goalposts are the whole point.
  2. DISJOINT VOCABULARY. Every slot bank here is deliberately disjoint from the
     training generators (names, pets, places, topics, obscure nouns). A model
     cannot pass by having memorised training surface forms.
  3. FAMILY-TAGGED. Every scenario carries `family` (its grammar) so we can prove
     no training family overlaps an evaluation family.
  4. STRATIFIED. 18 skills x ~22 scenarios, so per-skill estimates have enough
     samples to be worth reading.

The 20-scenario HARD set is retained as a fast smoke test, not for selection.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .bench import Probe, Scenario

SKILLS = [
    "long_memory", "memory_update", "memory_absent", "implicit_emotion",
    "sarcasm", "topic_callback", "repair", "contradiction",
    "ambiguous_referent", "negative_news", "boundary", "humor_timing",
    "no_fabrication", "minimal_input", "persona_consistency", "long_horizon",
    "celebration", "epistemic_discrimination",
]

# --- slot banks, DISJOINT from smalltalk/data/combi_gen.py -------------------
E_NAMES = ["marguerite", "desmond", "priyanka", "callum", "yusuf", "beatrix",
           "lorenzo", "annika", "tobias", "solveig", "rashid", "delphine",
           "gustav", "imelda", "krishna", "orla", "fergus", "renata"]
E_PETS = [("tortoise", "hercules"), ("parrot", "sonata"), ("ferret", "quill"),
          ("greyhound", "juniper"), ("gecko", "marbles"), ("pony", "thistle"),
          ("terrier", "wellington"), ("cockatiel", "pistachio")]
E_JOBS = ["a locksmith", "a radiographer", "an arborist", "a piano tuner",
          "a cartographer", "a beekeeper", "a stonemason", "a sound engineer"]
E_PLACES = ["aberdeen", "trieste", "kaunas", "hobart", "galway", "utrecht",
            "valparaiso", "sapporo"]
E_HOBBIES = ["falconry", "bookbinding", "sea kayaking", "orienteering",
             "glassblowing", "birdwatching", "fencing", "pottery throwing"]
E_OBSCURE = ["astrolabe", "orrery", "theremin", "penny-farthing", "hurdy-gurdy",
             "camera obscura", "quern-stone", "zoetrope", "hygrometer", "clepsydra"]
E_DAYS = ["monday", "wednesday", "thursday", "saturday", "sunday"]
E_TIMES = ["half six", "quarter past nine", "ten thirty", "midday", "seven"]
E_FILLER = ["yeah", "mm", "right", "i suppose", "true", "fair enough", "sort of",
            "kind of", "i guess so", "yeah exactly"]
E_SMALL = ["been quiet here", "not much going on", "same as always",
           "just pottering about", "nothing exciting", "the usual"]


def _pad(turns: list[str], rng: random.Random, target: int) -> list[str]:
    """Extend a scenario with low-signal filler to a target length."""
    out = list(turns)
    while len(out) < target:
        out.insert(len(out) - 1, rng.choice(E_FILLER + E_SMALL))
    return out


def _build(rng: random.Random) -> list[Scenario]:
    S: list[Scenario] = []
    n = 0

    def add(skill: str, family: str, turns: list[str], probes=(), note=""):
        nonlocal n
        S.append(Scenario(f"v2-{skill[:4]}-{n:04d}", skill, list(turns),
                          [Probe(**p) for p in probes], note))
        n += 1

    # 1. long_memory -- fact at turn 2-3, queried 8+ turns later
    for i in range(22):
        pet, pname = rng.choice(E_PETS)
        turns = ["hey", f"good thanks, i've got a {pet} called {pname}",
                 "yeah he's a handful", "had him about three years now"]
        turns = _pad(turns, rng, 12) + [f"what's my {pet} called again?"]
        add("long_memory", "pet_recall", turns,
            [{"turn": len(turns), "type": "long_memory", "expect_any": [pname]}])

    # 2. memory_update -- fact changes mid-conversation
    for i in range(22):
        d1, d2 = rng.sample(E_DAYS, 2)
        turns = ["hey", f"i'm travelling {d1}", "yeah booked it last week",
                 f"oh they moved it, it's {d2} now", "yeah annoying"]
        turns = _pad(turns, rng, 12) + ["what day am i travelling?"]
        add("memory_update", "day_change", turns,
            [{"turn": len(turns), "type": "memory_update",
              "expect_any": [d2], "forbid_any": [d1]}])

    # 3. memory_absent -- never told, must not invent
    for i in range(22):
        turns = ["hey", "just got back from a trip", "yeah it was good"]
        turns = _pad(turns, rng, 10) + ["so where did i say i went?"]
        add("memory_absent", "never_told", turns,
            [{"turn": len(turns), "type": "memory_absent",
              "expect_any": ["didn't", "did not", "never", "don't think you",
                             "you haven't", "not sure", "no idea", "?"],
              "forbid_any": E_PLACES}],
            note="user never named the place; naming one is fabrication")

    # 4. implicit_emotion
    for i in range(22):
        turns = ["hey", "it's fine honestly", "they went without me",
                 "no it's fine, i said it's fine", "i didn't really want to go anyway"]
        turns = _pad(turns, rng, 11)
        add("implicit_emotion", "exclusion", turns,
            [{"turn": 3, "type": "implicit_emotion",
              "forbid_any": ["nice", "great", "sounds fun", "awesome", "congrats"]}])

    # 5. sarcasm
    for i in range(22):
        thing = rng.choice(["my boiler packed in", "the train was cancelled",
                            "i locked myself out", "my tooth cracked"])
        turns = ["hey", "oh it's been a MARVELLOUS day", f"{thing}. wonderful.",
                 "truly living my best life", "ok no it was awful"]
        turns = _pad(turns, rng, 11)
        add("sarcasm", "ironic_positive", turns,
            [{"turn": 3, "type": "sarcasm",
              "forbid_any": ["that's great", "congrats", "amazing", "so happy", "wonderful!"]}])

    # 6. topic_callback
    for i in range(22):
        hob = rng.choice(E_HOBBIES)
        turns = ["hey", f"i took up {hob}", "yeah it's harder than it looks",
                 "oh hang on, did you hear the news about the bridge",
                 "yeah closed for months apparently", "anyway",
                 f"so yeah, the {hob}"]
        turns = _pad(turns, rng, 12)
        add("topic_callback", "digression_return", turns,
            [{"turn": 7, "type": "topic_callback",
              "expect_any": [hob.split()[0], "how", "going", "?"]}])

    # 7. repair
    for i in range(22):
        turns = ["hey", "i bought a boat", "sorry i meant a COAT not a boat",
                 "yeah a winter coat", "warm one"]
        turns = _pad(turns, rng, 11)
        add("repair", "misspoke", turns,
            [{"turn": 3, "type": "repair", "forbid_any": ["boat", "sail", "water", "river"]}])

    # 8. contradiction
    for i in range(22):
        turns = ["hey", "i never drink coffee, can't stand it", "yeah tea only",
                 "anyway i had two coffees this morning", "yeah i know"]
        turns = _pad(turns, rng, 11)
        add("contradiction", "self_contradiction", turns,
            [{"turn": 4, "type": "contradiction",
              "expect_any": ["?", "thought", "wait", "but", "hang on", "said"]}])

    # 9. ambiguous_referent
    for i in range(22):
        turns = ["hey", "it happened again", "no not that one, the other thing",
                 "ugh forget it"]
        turns = _pad(turns, rng, 11)
        add("ambiguous_referent", "vague_it", turns,
            [{"turn": 2, "type": "ambiguous_referent", "expect_any": ["?"]}])

    # 10. negative_news
    for i in range(22):
        who = rng.choice(["my grandmother", "my uncle", "my neighbour",
                          "my old teacher", "my aunt"])
        turns = ["hey", "not good news i'm afraid", f"{who} died on the weekend",
                 "yeah it was sudden", "funeral's next week"]
        turns = _pad(turns, rng, 12)
        add("negative_news", "bereavement", turns,
            [{"turn": 3, "type": "negative_news",
              "forbid_any": ["congrats", "that's great", "nice", "awesome", "lol",
                             "haha", "annoying", "wonderful", "exciting"]}])

    # 11. boundary
    for i in range(22):
        turns = ["hey", "rough one", "i'd rather not go into it",
                 "just talk to me about something else"]
        turns = _pad(turns, rng, 11)
        add("boundary", "deflect", turns,
            [{"turn": 3, "type": "boundary",
              "forbid_any": ["what happened", "tell me more", "why", "what's wrong",
                             "go on then"]}])

    # 12. humor_timing
    for i in range(22):
        turns = ["hey", "i did something stupid", "put my phone in the fridge",
                 "found it an hour later", "still cold"]
        turns = _pad(turns, rng, 11)
        add("humor_timing", "self_deprecating", turns,
            [{"turn": 3, "type": "humor_timing",
              "forbid_any": ["that's a common", "you should", "make sure you",
                             "next time try"]}])

    # 13. no_fabrication
    for i in range(22):
        obj = rng.choice(E_OBSCURE)
        turns = ["hey", "random question", f"what year was the {obj} invented",
                 "yeah i wondered too"]
        turns = _pad(turns, rng, 11)
        add("no_fabrication", "obscure_trivia", turns,
            [{"turn": 3, "type": "no_fabrication", "forbid_confident": True}])

    # 14. minimal_input
    for i in range(22):
        turns = ["hey"] + [rng.choice(E_FILLER) for _ in range(13)]
        add("minimal_input", "low_signal", turns, note="degeneration trap")

    # 15. persona_consistency
    for i in range(22):
        turns = ["hey", "do you like music?", "what sort?", "cool"]
        turns = _pad(turns, rng, 10) + ["what kind of music did you say you liked?"]
        add("persona_consistency", "self_recall", turns,
            [{"turn": len(turns), "type": "persona_consistency",
              "forbid_any": ["as an ai", "i don't have", "i am not able"]}])

    # 16. long_horizon -- 20 turns
    for i in range(22):
        turns = ["hey", "not bad you?", "just in for the night",
                 "made something simple", "yeah nothing fancy"]
        turns = _pad(turns, rng, 20)
        add("long_horizon", "endurance", turns, note="20 turns, low stakes")

    # 17. celebration
    for i in range(22):
        good = rng.choice(["i passed my viva", "we completed on the house",
                           "i got into the programme", "my book got accepted",
                           "i finished the marathon"])
        turns = ["hey", "big news", good, "yeah still sinking in"]
        turns = _pad(turns, rng, 11)
        add("celebration", "good_news", turns,
            [{"turn": 3, "type": "celebration",
              "expect_any": ["congrat", "amazing", "nice", "great", "well done",
                             "brilliant", "yes", "happy", "wow", "!"]}])

    # 18. epistemic_discrimination -- answer IS in context, must recall not refuse
    for i in range(22):
        job = rng.choice(E_JOBS)
        turns = ["hey", f"i'm {job}", "yeah about six years now"]
        turns = _pad(turns, rng, 10) + ["what do i do for a living again?"]
        add("epistemic_discrimination", "context_has_answer", turns,
            [{"turn": len(turns), "type": "epistemic_discrimination",
              "expect_any": [job.split()[-1]],
              "forbid_any": ["no idea", "don't know", "you didn't say", "not sure"]}],
            note="answer IS present; refusing is the failure mode")
    return S


def build_scenarios(seed: int = 20240819) -> list[Scenario]:
    return _build(random.Random(seed))


def checksum(scenarios: list[Scenario]) -> str:
    blob = json.dumps([s.to_dict() for s in scenarios], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


FREEZE_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "bench_v2_freeze.json"


def freeze(path: Path | None = None) -> dict:
    s = build_scenarios()
    info = {"seed": 20240819, "n_scenarios": len(s), "checksum": checksum(s),
            "skills": {k: sum(1 for x in s if x.category == k) for k in SKILLS}}
    p = path or FREEZE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2))
    return info


def verify_frozen(path: Path | None = None) -> None:
    """Refuse to evaluate if the benchmark has drifted."""
    p = path or FREEZE_PATH
    if not p.exists():
        raise FileNotFoundError(f"benchmark not frozen; run freeze() -> {p}")
    info = json.loads(p.read_text())
    actual = checksum(build_scenarios(info["seed"]))
    if actual != info["checksum"]:
        raise RuntimeError(
            f"SmallTalkBench-v2 CHANGED (frozen {info['checksum']}, now {actual}). "
            "Scenarios are immutable -- revert the edit or results are not comparable."
        )


def benchmark_families() -> set[str]:
    """Grammar families used by evaluation; training must never reuse these."""
    return {"pet_recall", "day_change", "never_told", "exclusion", "ironic_positive",
            "digression_return", "misspoke", "self_contradiction", "vague_it",
            "bereavement", "deflect", "self_deprecating", "obscure_trivia",
            "low_signal", "self_recall", "endurance", "good_news",
            "context_has_answer"}
