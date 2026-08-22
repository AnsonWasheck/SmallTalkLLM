"""ElaborationBench: does the model expand on what it understands, and stay quiet
about what it does not?

Why the existing benchmarks miss this
-------------------------------------
Core-Bench scores one turn against an accept set. StateBench scores valence
tracking. VarietyBench scores repetition. None of them notice that all three
current checkpoints answer "i got a new dog", "i'm a nurse", "i've been learning
guitar" and "we went to italy" with the SAME reply -- "oh nice, how's that
going?". That is a topic-agnostic hedge that sounds engaged and demonstrates no
understanding whatsoever, and every existing metric scores it as a success.

What this measures
------------------
Two halves, and the second one matters more.

  KNOWN topics    -- subjects the curriculum covers. A good reply says something
                     tied to THIS subject.
  UNKNOWN topics  -- subjects deliberately outside it. A good reply falls back to
                     a generic reaction and asserts nothing specific.

The headline failure is FALSE ELABORATION: confidently saying something specific
about a subject the model does not know. A model that hedges too often is merely
dull; a model that fabricates specificity is broken, and at 6.7M parameters the
honest behaviour is to recognise a handful of subjects well and decline on the
rest.

The definitional choice that drives everything (v2)
---------------------------------------------------
ELABORATION MEANS INCORPORATING THE USER'S REFERENT.

v1 counted any subject-relevant word as elaboration, which let a family of
content-free follow-ups -- "what kind?", "how long?", "whereabouts?" -- pass as
understanding. They are not: they are simply a larger hedge vocabulary, and a
model cycling through eight hedges understands no more than one cycling through
a single hedge.

A reply elaborates when it reuses a content word the user actually said:

    "i got a new dog"          -> "what's the dog called?"
    "my laptop screen cracked" -> "is the laptop still usable?"

This is unfoolable by phrase-book expansion, it generalises to any noun
including subjects never trained on, and it is genuinely harder than hedging --
the model must attend to and reproduce the referent instead of emitting stock
text. It is also exactly what makes a reply feel understood.

Consequence for the unknown half: unknown subjects should now be ELABORATED ON
too, referentially. The sin is not specificity, it is INVENTED specificity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ELABORATION_VERSION = "elaborationbench-v2.0.0"

_WORD = re.compile(r"[a-z']+")


def words(t: str) -> set[str]:
    """Tokens plus crude singulars, so "dogs" matches a "dog" relevance set.

    Not a stemmer: a real one would collapse distinctions this benchmark relies
    on. Stripping a trailing s is enough for the plural/singular mismatches that
    actually occur here.
    """
    ws = set(_WORD.findall(t.lower()))
    return ws | {w[:-1] for w in ws if len(w) > 3 and w.endswith("s")}


# Generic valence reactions: correct as a fallback, never evidence of understanding.
GENERIC = {
    "that's great", "that's great!", "nice one", "good stuff", "that's brilliant",
    "ah lovely", "that's the way", "happy for you", "glad to hear it",
    "that sounds rough", "that sucks", "that's rubbish", "sorry to hear that",
    "oh no", "oh no, i'm sorry", "that's a pain", "rough one", "that's hard going",
    "sounds exhausting", "hope it picks up", "hopefully tomorrow's better",
    "yeah exactly", "i see", "mm", "right", "fair enough", "okay", "mhm", "go on",
    "no worries", "long may it last", "can't complain then", "not great is it",
    "ugh, that's grim", "that's lovely",
}

# Universal follow-ups: applied to any subject, so they demonstrate nothing.
HEDGE_PATTERNS = [
    r"how'?s (that|it) going",
    r"what happened",          # anchoring this to the start missed
                               # "oh no, what happened?", which is the same hedge
    r"^tell me (more|about (it|that))",
    r"^how (come|so)$",
    r"^(oh )?really\??$",
    r"^what do you mean",
    r"^go on$",
]


@dataclass(frozen=True)
class Scenario:
    sid: str
    prompt: str
    known: bool
    relevant: frozenset[str]     # content a subject-aware reply would use
    forbidden: frozenset[str]    # subjects it must NOT invent (unknown half only)

    @property
    def id(self) -> str:
        return f"{self.sid}::{self.prompt}"


def _k(sid, prompt, relevant) -> Scenario:
    return Scenario(sid, prompt, True, frozenset(relevant.split()), frozenset())


def _u(sid, prompt, forbidden) -> Scenario:
    return Scenario(sid, prompt, False, frozenset(), frozenset(forbidden.split()))


# --- KNOWN: subjects the curriculum covers ----------------------------------
KNOWN: list[Scenario] = [
    _k("work-1", "i'm a nurse", "nurse nursing hospital shifts ward patients work job"),
    _k("work-2", "i started a new job", "job work start role team office first"),
    _k("work-3", "work has been really busy", "work busy job shifts hours week"),
    _k("pet-1", "i got a new dog", "dog puppy pup breed name walk walks"),
    _k("pet-2", "my cat keeps waking me up", "cat kitten night sleep bed hungry"),
    _k("study-1", "i've got exams next week", "exam exams revision study revising school subject"),
    _k("study-2", "i signed up for an evening course", "course class learning study evening subject"),
    _k("travel-1", "we went to italy", "italy trip holiday travel food went abroad where"),
    _k("travel-2", "i'm off to scotland next month", "scotland trip travel going visit where"),
    _k("health-1", "i've been feeling run down", "rest sleep unwell better doctor tired ill"),
    _k("sleep-1", "i barely slept last night", "sleep slept bed tired night awake"),
    _k("home-1", "we're redecorating the kitchen", "kitchen paint house home room decorating colour"),
    _k("hobby-1", "i've been learning guitar", "guitar play playing music song chords learning"),
    _k("hobby-2", "i've taken up jogging", "jog jogging run miles pace morning far"),
    _k("food-1", "i made bread from scratch", "bread bake baking loaf oven made dough"),
    _k("family-1", "my sister is visiting", "sister visit staying long she her family"),
    _k("money-1", "the bills came in higher than usual", "bills money pay budget rent higher"),
    _k("social-1", "i saw an old friend yesterday", "friend saw catch years met she he they"),
]

# --- UNKNOWN: subjects deliberately outside the curriculum -------------------
# `forbidden` lists subjects it must not invent. A safe reply reacts to the
# valence and asserts nothing; asking "what happened?" is fine, inventing a dog
# is not.
_FORBID = "dog cat guitar nurse italy exam kitchen baking running sister job"
UNKNOWN: list[Scenario] = [
    _u("car-1", "my car broke down on the motorway", _FORBID),
    _u("legal-1", "i've got jury duty next week", _FORBID),
    _u("dental-1", "i had a root canal this morning", _FORBID),
    _u("admin-1", "my passport application got rejected", _FORBID),
    _u("plumb-1", "the boiler packed in again", _FORBID),
    _u("wedding-1", "my cousin's wedding is on saturday", _FORBID),
    _u("tech-1", "my laptop screen cracked", _FORBID),
    _u("insure-1", "the insurance claim went through", _FORBID),
    _u("neighbour-1", "the neighbours were arguing all night", _FORBID),
    _u("parking-1", "i got a parking ticket", _FORBID),
    _u("delivery-1", "the delivery never turned up", _FORBID),
    _u("bank-1", "the bank froze my account", _FORBID),
]

SCENARIOS: list[Scenario] = KNOWN + UNKNOWN


# Words carried by almost any reply; matching on these would score "i" or "you"
# as referent reuse and make the metric meaningless.
STOP = {
    "i", "you", "it", "that", "this", "the", "a", "an", "is", "was", "are", "were",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "so", "my", "your",
    "me", "we", "they", "he", "she", "them", "him", "her", "do", "did", "does",
    "have", "has", "had", "be", "been", "got", "get", "no", "not", "yeah", "yes",
    "oh", "ah", "well", "how", "what", "when", "where", "who", "why", "s", "t",
    "up", "out", "all", "just", "really", "very", "with", "about", "like", "if",
    "there", "here", "again", "still", "too", "then", "now", "one", "new", "much",
}


def referents(prompt: str) -> set[str]:
    """Content words the user actually said. The harness can extract these
    exactly; the model should not have to learn to."""
    return {w for w in _WORD.findall(prompt.lower())
            if w not in STOP and len(w) > 2}


def classify(reply: str, sc: Scenario) -> str:
    """ELABORATION | HEDGE | GENERIC | FALSE_ELABORATION | OTHER.

    Order matters. GENERIC and HEDGE are checked first because a stock phrase is
    what the model would have said regardless of the input; if it happens to
    contain a referent by coincidence, that is not evidence of understanding.
    """
    r = reply.strip().lower().rstrip(".!")
    if not r:
        return "OTHER"
    if r in GENERIC:
        return "GENERIC"
    for pat in HEDGE_PATTERNS:
        if re.search(pat, r):
            return "HEDGE"
    w = words(r)
    # Invented specificity: naming a subject the user never mentioned.
    if not sc.known and (w & sc.forbidden) and not (w & referents(sc.prompt)):
        return "FALSE_ELABORATION"
    if w & referents(sc.prompt):
        return "ELABORATION"
    return "OTHER"


def score(replies: dict[str, str]) -> dict:
    from collections import Counter
    known = [s for s in SCENARIOS if s.known]
    unknown = [s for s in SCENARIOS if not s.known]
    kc = Counter(classify(replies.get(s.id, ""), s) for s in known)
    uc = Counter(classify(replies.get(s.id, ""), s) for s in unknown)
    allc = kc + uc

    # Elaboration is now scored on BOTH halves: referencing what the user said is
    # correct everywhere. The unknown half is therefore a GENERALISATION test --
    # can it engage with a subject it was never trained on -- rather than a
    # restraint test.
    elaboration = allc["ELABORATION"] / len(SCENARIOS)
    elab_known = kc["ELABORATION"] / len(known)
    elab_unknown = uc["ELABORATION"] / len(unknown)
    hedge = allc["HEDGE"] / len(SCENARIOS)
    false_elab = uc["FALSE_ELABORATION"] / len(unknown)

    # Inventing a subject is penalised twice as hard as hedging is rewarded:
    # a model that fabricates is worse company than one that is merely dull.
    composite = elaboration - 2.0 * false_elab - 0.25 * hedge

    return {
        "checksum": checksum(),
        "elaboration_rate": elaboration,
        "elaboration_known": elab_known,
        "elaboration_unknown": elab_unknown,
        "false_elaboration_rate": false_elab,
        "hedge_rate": hedge,
        "composite": composite,
        "known_breakdown": dict(kc),
        "unknown_breakdown": dict(uc),
    }


def checksum() -> str:
    h = hashlib.sha256(ELABORATION_VERSION.encode())
    for s in SCENARIOS:
        h.update(s.id.encode())
        h.update(",".join(sorted(s.relevant)).encode())
        h.update(",".join(sorted(s.forbidden)).encode())
        h.update(b"\0")
    h.update(json.dumps(sorted(GENERIC)).encode())
    h.update(json.dumps(HEDGE_PATTERNS).encode())
    return h.hexdigest()[:16]


def freeze(path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cs = checksum()
    p.write_text(json.dumps({"version": ELABORATION_VERSION, "checksum": cs,
                             "known": len(KNOWN), "unknown": len(UNKNOWN)}, indent=2))
    return cs


def verify_frozen(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run freeze() before comparing runs")
    if json.loads(p.read_text())["checksum"] != checksum():
        raise RuntimeError("ElaborationBench drifted; earlier scores are not comparable")
