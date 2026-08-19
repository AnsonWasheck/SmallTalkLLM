"""Deterministic latent scenario planner.

The single most important design decision in this pipeline: **structure is sampled
deterministically in code; only surface language comes from Qwen.**

Asking a 9B model for "random conversations" collapses toward its favourite patterns —
that is precisely how we got an 89%-repeat corpus from the last synthetic source, and how
the template generator got to 97%. Here the combinatorial space is enumerated by us
(relationship x setting x mood x topic x discourse plan x entity set x epistemic
condition x ...), and Qwen's only job is to *realise* one sampled point naturally.

The student never sees this metadata — it is stripped before training and kept only for
filtering, analysis, family splitting and curriculum construction.

Entity banks here are deliberately LARGE and open-ended. The previous failure was
`surface pattern -> memorised slot value`: with ~15 pet names in the corpus the model
learned to emit a pet name from that closed set rather than read context. Names,
occupations, pets, places and objects are combinatorially generated so the student cannot
memorise the value set.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

PLANNER_VERSION = "planner-v1.0.0"

# ---------------------------------------------------------------------------
# Open-ended entity generation (NOT small fixed banks)
# ---------------------------------------------------------------------------
_FIRST_A = ["mar", "jen", "dav", "sam", "ros", "tom", "pri", "cal", "yus", "bea", "lor",
            "ann", "tob", "sol", "rash", "del", "gus", "imel", "krish", "orl", "ferg",
            "ren", "nad", "oli", "hann", "isa", "jam", "kier", "lia", "mic", "nor",
            "pau", "quin", "rob", "sim", "tan", "urs", "vic", "wen", "xav", "yol", "zar"]
_FIRST_B = ["a", "ie", "o", "an", "en", "us", "ia", "el", "ah", "y", "is", "on", "ric",
            "ella", "ony", "ina", "ard", "een", "ita", "ley", "son", "ette"]
_PET_A = ["bis", "lun", "pep", "man", "oli", "smo", "ban", "pop", "waf", "clo", "dash",
          "zig", "nal", "pip", "coc", "mur", "tof", "juni", "her", "quil", "mar", "this",
          "well", "pist", "mo", "nug", "pick", "bean", "sock", "mango", "tur", "wil"]
_PET_B = ["cuit", "a", "per", "go", "ve", "key", "dit", "py", "fle", "ver", "", "gy",
          "la", "", "oa", "phy", "fee", "per", "cules", "l", "bles", "tle", "ington"]

_PET_KINDS = ["dog", "cat", "kitten", "puppy", "rabbit", "hamster", "parrot", "budgie",
              "tortoise", "gecko", "ferret", "guinea pig", "cockatiel", "beagle",
              "greyhound", "terrier", "spaniel", "tabby", "goldfish", "corn snake"]
_JOBS = ["nurse", "teacher", "barista", "electrician", "graphic designer", "bus driver",
         "chef", "plumber", "vet nurse", "shop assistant", "software tester",
         "physiotherapist", "librarian", "hairdresser", "paramedic", "accountant",
         "warehouse picker", "care worker", "bar manager", "dental hygienist",
         "locksmith", "radiographer", "arborist", "piano tuner", "beekeeper",
         "stonemason", "sound engineer", "florist", "tattoo artist", "midwife",
         "scaffolder", "translator", "postie", "lab technician", "estate agent",
         "train conductor", "baker", "welder", "surveyor", "optician"]
_HOBBIES = ["climbing", "pottery", "baking sourdough", "birdwatching", "sea swimming",
            "learning bass", "cross-stitch", "running", "board games", "gardening",
            "cycling", "photography", "knitting", "fencing", "salsa classes",
            "restoring furniture", "model trains", "kayaking", "chess", "roller derby",
            "bookbinding", "foraging", "darts", "beekeeping", "geocaching", "origami",
            "amateur radio", "urban sketching", "bouldering", "wild camping"]
_PLACES = ["leeds", "bristol", "glasgow", "cork", "hull", "dundee", "swansea", "norwich",
           "plymouth", "derry", "wrexham", "inverness", "carlisle", "stoke", "luton",
           "aberdeen", "galway", "utrecht", "trieste", "kaunas", "hobart", "sapporo",
           "valparaiso", "tromso", "porto", "ghent", "olomouc", "bergen"]
_TOPICS = [
    "work", "a shift that overran", "a coworker", "a job application", "school",
    "an exam", "a group project", "food", "cooking a disaster", "a takeaway",
    "the weekly shop", "sleep", "insomnia", "a nap", "a new album", "a gig",
    "learning an instrument", "a game they're stuck on", "a show they finished",
    "a film that disappointed", "the gym", "a run", "a walk", "a friend's news",
    "a friend moving away", "a sibling", "a parent", "a family visit", "the weather",
    "endless rain", "the first warm day", "weekend plans", "cancelled plans",
    "boredom", "a delivery that never came", "a broken appliance", "a dentist trip",
    "a haircut", "a neighbour", "commuting", "a train delay", "a new flat",
    "moving house", "money being tight", "a small win at work", "a hobby they picked up",
    "a pet doing something stupid", "a wedding invite", "a birthday", "a hangover",
    "a minor injury", "a doctor's appointment", "houseplants", "a leaky tap",
    "a parcel misdelivered", "a bad night out", "a good night in", "a podcast",
]

_RELATIONSHIPS = ["close friend of years", "newish friend", "old school friend",
                  "sibling", "cousin", "flatmate", "coworker they like",
                  "partner", "friend they see rarely", "friend from a hobby group"]
_SETTINGS = ["late-night texting", "midday message exchange", "commute texting",
             "weekend afternoon", "early morning before work", "evening wind-down",
             "texting while procrastinating", "quick check-in between things"]
_MOODS = ["neutral", "tired", "cheerful", "flat", "distracted", "anxious", "content",
          "irritable", "wistful", "giddy", "drained", "restless", "relieved", "bored"]
_REGISTERS = ["lowercase texter, minimal punctuation", "punctuation-normal, warm",
              "dry and clipped", "chatty and rambly", "emoji-light, playful",
              "short bursts, several messages in a row", "considered, complete sentences"]
_GOALS = ["just chatting, no goal", "wants to vent a bit", "sharing news",
          "killing time", "checking in on the other person", "asking a small favour",
          "working out plans", "processing something out loud"]

# Skill mix from the brief (must sum to ~1.0)
SKILL_MIX = {
    "mundane": 0.30,
    "coherence_callback": 0.15,
    "epistemic": 0.12,
    "memory_state": 0.12,
    "emotion": 0.10,
    "humor_sarcasm": 0.06,
    "repair_ambiguity": 0.05,
    "disagreement": 0.05,
    "low_energy": 0.05,
}

EPISTEMIC_CONDITIONS = [
    "known_in_context",      # answer is present earlier -> answer it
    "never_stated",          # never mentioned -> say you don't know
    "corrected",             # value changed -> use the newest
    "conflicting",           # two claims -> resolve by chronology
    "uncertain",             # partial info -> hedge appropriately
    "ambiguous_referent",    # unclear "it"/"she" -> ask
    "two_similar_entities",  # keep two similar things distinct
]

DISCOURSE_MOVES = [
    "opening", "disclosure", "acknowledgement", "clarification", "detail",
    "related anecdote", "side topic", "new entity introduced", "callback to earlier topic",
    "correction", "later reference to an earlier fact", "mild disagreement",
    "teasing", "topic drift", "winding down", "close",
]

LENGTH_BUCKETS = [("short", 4, 7), ("medium", 8, 14), ("long", 15, 30)]
# Reference distance buckets measured in STUDENT tokenizer tokens.
DISTANCE_BUCKETS = ["0-128", "129-256", "257-512", "513-768", "769-1024"]


def _name(r: random.Random) -> str:
    return (r.choice(_FIRST_A) + r.choice(_FIRST_B)).capitalize()


def _petname(r: random.Random) -> str:
    return (r.choice(_PET_A) + r.choice(_PET_B)).capitalize()


@dataclass
class LatentSpec:
    """Hidden structure for one dialogue. Never shown to the student."""

    id: str
    family: str
    split: str
    skill: str
    relationship: str
    setting: str
    user_mood: str
    assistant_register: str
    topic: str
    conversation_goal: str
    discourse_plan: list[str]
    entities: dict[str, Any]
    known_facts: list[str]
    unknown_facts: list[str]
    state_mutations: list[str]
    contradictions: list[str]
    callbacks: list[str]
    epistemic_condition: str
    sarcasm: bool
    ambiguity: bool
    target_turns: int
    length_profile: str
    reference_distance: str | None
    planner_version: str = PLANNER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _family_for(skill: str, r: random.Random) -> str:
    """Family = the grammar/scenario template. Assigned BEFORE surface realisation."""
    return f"{skill}:{r.randrange(400):03d}"


def sample_spec(r: random.Random, idx: int, split_of_family) -> LatentSpec:
    skills = list(SKILL_MIX)
    weights = [SKILL_MIX[s] for s in skills]
    skill = r.choices(skills, weights=weights)[0]
    family = _family_for(skill, r)
    split = split_of_family(family)

    length_profile, lo, hi = r.choice(LENGTH_BUCKETS)
    # Long-context curriculum: memory/coherence skills skew long.
    if skill in ("memory_state", "coherence_callback") and r.random() < 0.55:
        length_profile, lo, hi = "long", 15, 30
    turns = r.randint(lo, hi)

    # Entities: generated, not drawn from a tiny closed bank.
    ents: dict[str, Any] = {}
    n_people = r.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    ents["people"] = [{"name": _name(r), "relation": r.choice(
        ["sister", "brother", "flatmate", "coworker", "friend", "cousin", "neighbour",
         "partner", "mum", "dad", "manager"])} for _ in range(n_people)]
    if r.random() < 0.35:
        ents["pet"] = {"name": _petname(r), "kind": r.choice(_PET_KINDS)}
    if r.random() < 0.30:
        ents["job"] = r.choice(_JOBS)
    if r.random() < 0.30:
        ents["hobby"] = r.choice(_HOBBIES)
    if r.random() < 0.25:
        ents["place"] = r.choice(_PLACES)

    known: list[str] = []
    unknown: list[str] = []
    mutations: list[str] = []
    contradictions: list[str] = []
    callbacks: list[str] = []
    epistemic = "not_applicable"
    ref_distance = None

    def distance_for(turns_: int) -> str:
        """Reference distance must be physically achievable in `turns_` messages.

        Bug this fixes: sampling the bucket independently produced specs like
        "about 4 messages" + "recalled detail ~513-768 tokens earlier", which is
        impossible and forces the writer to either pad or ignore the constraint.
        """
        if turns_ <= 7:
            return "0-128"
        if turns_ <= 11:
            return r.choice(["0-128", "129-256"])
        if turns_ <= 16:
            return r.choice(["129-256", "257-512"])
        if turns_ <= 22:
            return r.choice(["257-512", "513-768"])
        return r.choice(["513-768", "769-1024"])

    if skill == "epistemic":
        epistemic = r.choice(EPISTEMIC_CONDITIONS)
        ref_distance = distance_for(turns)
        subj = ents["pet"]["name"] if "pet" in ents else ents["people"][0]["name"]
        if epistemic == "known_in_context":
            known.append(f"the user states early on a specific detail about {subj}, "
                         f"and later asks the assistant to recall it")
        elif epistemic == "never_stated":
            unknown.append("the user asks about a detail they have NEVER mentioned; "
                           "the assistant must say it doesn't know rather than invent one")
        elif epistemic == "corrected":
            mutations.append(f"a fact about {subj} is stated, then explicitly corrected "
                             f"later; a still-later question must get the NEW value")
        elif epistemic == "conflicting":
            contradictions.append("the user says two incompatible things; the assistant "
                                  "resolves by the most recent statement, lightly noting it")
        elif epistemic == "uncertain":
            known.append("only partial information is available; the assistant hedges "
                         "instead of committing")
        elif epistemic == "ambiguous_referent":
            unknown.append("a pronoun or 'it' has no clear antecedent; the assistant asks "
                           "which one rather than guessing")
        else:
            known.append("two similar entities are discussed; the assistant keeps them "
                         "distinct and does not conflate them")

    if skill == "memory_state":
        ref_distance = distance_for(turns)
        subj = ents["people"][0]["name"]
        known.append(f"a concrete fact about {subj} is established early")
        if r.random() < 0.6:
            mutations.append(f"that fact about {subj} changes partway through")
        if r.random() < 0.4 and len(ents["people"]) > 1:
            known.append(f"a second person, {ents['people'][1]['name']}, is also "
                         f"discussed and must not be confused with {subj}")
        callbacks.append("late in the conversation the user asks about the earlier fact")

    if skill == "coherence_callback":
        callbacks.append("the conversation drifts to a second topic and then genuinely "
                         "returns to the first, referring back to a specific detail")

    plan = ["opening"]
    pool = [m for m in DISCOURSE_MOVES if m not in ("opening", "close")]
    r.shuffle(pool)
    plan += pool[: max(3, min(len(pool), turns // 2))]
    if callbacks and "callback to earlier topic" not in plan:
        plan.append("callback to earlier topic")
    if mutations and "correction" not in plan:
        plan.insert(len(plan) // 2, "correction")
    if r.random() < 0.6:
        plan.append("close")

    return LatentSpec(
        id=f"q-{idx:07d}",
        family=family,
        split=split,
        skill=skill,
        relationship=r.choice(_RELATIONSHIPS),
        setting=r.choice(_SETTINGS),
        user_mood=r.choice(_MOODS),
        assistant_register=r.choice(_REGISTERS),
        topic=r.choice(_TOPICS),
        conversation_goal=r.choice(_GOALS),
        discourse_plan=plan,
        entities=ents,
        known_facts=known,
        unknown_facts=unknown,
        state_mutations=mutations,
        contradictions=contradictions,
        callbacks=callbacks,
        epistemic_condition=epistemic,
        sarcasm=(skill == "humor_sarcasm" and r.random() < 0.6),
        ambiguity=(skill == "repair_ambiguity"),
        target_turns=turns,
        length_profile=length_profile,
        reference_distance=ref_distance,
    )


def family_splitter(val_frac: float = 0.12, test_frac: float = 0.12):
    """Hash-based family -> split. Deterministic, and disjoint by construction."""
    def split_of(family: str) -> str:
        h = int(hashlib.blake2b(family.encode(), digest_size=8).hexdigest(), 16)
        x = (h % 10_000) / 10_000.0
        if x < val_frac:
            return "validation"
        if x < val_frac + test_frac:
            return "test"
        return "train"
    return split_of


def generate_specs(n: int, seed: int = 0) -> Iterator[LatentSpec]:
    r = random.Random(seed)
    split_of = family_splitter()
    for i in range(n):
        yield sample_spec(r, i, split_of)


def write_specs(path, n: int, seed: int = 0) -> int:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for spec in generate_specs(n, seed):
            f.write(json.dumps(spec.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
