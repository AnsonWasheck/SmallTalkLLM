"""SmallTalkBench-HARD: a deliberately punishing, FROZEN, held-out benchmark.

This is meant to be out of reach for a 6.7M model at the start. It exists to
produce a *ranked list of failures* to drive corpus work, not to be passed.

RULES (enforced by tools/check_leakage.py -- do not relax them):
  1. FROZEN. Once committed, scenarios are never edited, reworded or removed.
     Moving the goalposts would make overnight "progress" meaningless.
  2. HELD OUT. No generated training data may reuse these strings. Every
     iteration runs a leakage check against n-grams from this file.
  3. Difficulty is intentional. A low score is the expected starting state.

What makes it hard (each is a named capability the corpus must teach):
  long_memory        recall a detail from 10+ turns earlier
  memory_update      a fact CHANGES mid-conversation; use the new one
  implicit_emotion   feeling is implied, never named
  sarcasm            surface text contradicts intent
  topic_callback     return to an earlier topic after a digression
  repair             user corrects a misunderstanding
  contradiction      user says something inconsistent with earlier
  ambiguous_referent "it"/"that" with no clear antecedent
  negative_news      bad news that must not be met with cheer
  boundary           user deflects; model must not push
  humor_timing       a joke that needs a beat, not an explanation
  no_fabrication     obscure factual question, must decline naturally
  minimal_input      near-zero signal for many consecutive turns
  persona_consistency the model's own earlier claims must stay stable
  long_horizon       20 turns without collapse
"""

from __future__ import annotations

from .bench import Probe, Scenario

HARD_CATEGORIES = (
    "long_memory", "memory_update", "implicit_emotion", "sarcasm",
    "topic_callback", "repair", "contradiction", "ambiguous_referent",
    "negative_news", "boundary", "humor_timing", "no_fabrication",
    "minimal_input", "persona_consistency", "long_horizon",
)


def _s(sid, cat, turns, probes=(), note="") -> Scenario:
    return Scenario(sid, cat, list(turns), [Probe(**p) for p in probes], note)


HARD_SCENARIOS: list[Scenario] = [
    _s("h-mem-01", "long_memory", [
        "hey", "im good, just got back from my sister's", "her name's priya, she just moved to leeds",
        "yeah the drive was long", "like four hours", "traffic was awful",
        "anyway how are you", "that's good", "i had a sandwich at a services stop",
        "it was mediocre honestly", "so yeah, long day", "im gonna crash soon",
        "oh wait, what was my sister's name again?",
    ], probes=[{"turn": 13, "type": "long_memory", "expect_any": ["priya"],
                "note": "recall from turn 3, 10 turns later"}]),

    _s("h-mem-02", "memory_update", [
        "hey", "im meeting jamie on friday", "yeah we're getting dinner",
        "actually he just messaged, we moved it to saturday",
        "yeah saturday now", "should be fun", "we're going to that ramen place",
        "ive been craving it", "hope it's good", "so what day am i seeing jamie?",
    ], probes=[{"turn": 10, "type": "memory_update", "expect_any": ["saturday"],
                "forbid_any": ["friday"], "note": "must use the UPDATED day"}]),

    _s("h-emo-01", "implicit_emotion", [
        "hey", "i guess it's fine", "no it's whatever, honestly",
        "he just didn't show up. that's all", "no i waited like an hour",
        "i mean it's not a big deal", "yeah", "i dunno why im even annoyed",
        "maybe im overreacting", "anyway",
    ], probes=[{"turn": 5, "type": "implicit_emotion",
                "forbid_any": ["congrats", "that's great", "nice", "awesome", "lol"],
                "note": "user is hurt but never says so"}]),

    _s("h-sarc-01", "sarcasm", [
        "hey", "oh it was a FANTASTIC day", "yeah my car broke down. amazing.",
        "and then it rained. love that for me", "truly living the dream",
        "no but seriously it was rough", "yeah", "im just tired",
        "gonna order food and give up", "night",
    ], probes=[{"turn": 3, "type": "sarcasm",
                "forbid_any": ["that's great", "awesome!", "congrats", "so happy"],
                "note": "surface positive, actual negative"}]),

    _s("h-call-01", "topic_callback", [
        "hey", "i started that pottery class", "yeah it's messy but fun",
        "oh hold on, did you see the storm last night", "our power went out for an hour",
        "yeah it was wild", "anyway", "so about the class, i made a bowl",
        "it's lopsided but i love it", "im gonna keep going",
    ], probes=[{"turn": 8, "type": "topic_callback", "expect_any": ["bowl", "pottery", "class", "made"],
                "note": "must re-engage pottery after the storm digression"}]),

    _s("h-rep-01", "repair", [
        "hey", "i got a new bike", "no sorry i meant a new BIKE, not a car",
        "yeah bicycle", "haha yeah", "it's second hand but nice",
        "im gonna ride it to work", "about 20 minutes",
        "yeah beats the bus", "gonna try tomorrow",
    ], probes=[{"turn": 4, "type": "repair", "forbid_any": ["car", "drive", "engine"],
                "note": "must drop the wrong reading"}]),

    _s("h-con-01", "contradiction", [
        "hey", "i hate coffee, never touch it", "yeah tea person",
        "anyway i had three coffees today", "lol", "yeah i know",
        "i was desperate", "deadlines", "im so wired now", "cant sleep",
    ], probes=[{"turn": 4, "type": "contradiction", "expect_any": ["?", "thought", "hate", "wait", "but"],
                "note": "should notice the inconsistency, gently"}]),

    _s("h-amb-01", "ambiguous_referent", [
        "hey", "so it happened again", "yeah that thing",
        "no not that, the other one", "ugh nevermind",
        "it's not important", "anyway how was your day", "that's nice",
        "yeah", "ok im off",
    ], probes=[{"turn": 2, "type": "ambiguous_referent", "expect_any": ["?"],
                "note": "must ask, not pretend to understand"}]),

    _s("h-neg-01", "negative_news", [
        "hey", "not great honestly", "my grandad passed away on tuesday",
        "yeah it was expected but still", "we're doing the funeral friday",
        "im ok, just numb", "yeah", "thanks", "ill be alright",
        "talk later",
    ], probes=[{"turn": 3, "type": "negative_news",
                "forbid_any": ["congrats", "that's great", "nice", "awesome", "lol", "haha",
                               "sorry to hear that, however", "as an ai"],
                "note": "hardest possible valence test"}]),

    _s("h-bound-01", "boundary", [
        "hey", "rough week", "i'd rather not talk about it",
        "yeah", "just distract me", "tell me something else",
        "yeah anything", "haha ok", "that's better", "thanks",
    ], probes=[{"turn": 3, "type": "boundary",
                "forbid_any": ["what happened", "tell me more", "why", "what's wrong"],
                "note": "must respect the deflection"}]),

    _s("h-fab-01", "no_fabrication", [
        "hey", "quick q", "what year was the zeppelin LZ-129 refitted",
        "yeah i thought so too", "no idea either honestly",
        "i saw it in a documentary", "it was pretty good",
        "about airships mostly", "yeah niche i know", "anyway later",
    ], probes=[{"turn": 3, "type": "no_fabrication", "forbid_confident": True,
                "note": "must decline, not invent a year"}]),

    _s("h-min-01", "minimal_input", [
        "hey", "yeah", "mm", "idk", "sure", "ok", "yeah", "hm", "maybe", "k",
        "yeah", "ok", "mm", "sure", "yeah",
    ], note="15 near-empty turns; degeneration trap"),

    _s("h-pers-01", "persona_consistency", [
        "hey", "do you like music?", "what kind?", "cool",
        "do you play anything?", "nice", "so what kind of music did you say you liked?",
        "right", "yeah that tracks", "cool",
    ], probes=[{"turn": 7, "type": "persona_consistency", "forbid_any": ["as an ai", "i don't have"],
                "note": "must not contradict its own turn-2 answer"}]),

    _s("h-long-01", "long_horizon", [
        "hey", "not bad, you?", "just chilling", "watched a film last night",
        "some sci fi thing", "it was ok", "bit slow honestly",
        "yeah the ending was rushed", "i'd give it a 6",
        "what've you been up to", "nice", "sounds relaxing",
        "im thinking of going for a walk", "yeah it's clear out",
        "maybe the park", "there's a lake there", "ducks and everything",
        "haha yeah", "ok im heading out", "later!",
    ], note="20 turns, low stakes; pure endurance"),

    _s("h-hum-01", "humor_timing", [
        "hey", "i just did something stupid", "i put salt in my coffee",
        "instead of sugar", "yeah i drank half of it before i noticed",
        "it was awful", "im wide awake now though", "silver linings",
        "yeah", "anyway",
    ], probes=[{"turn": 5, "type": "humor_timing",
                "forbid_any": ["that's a common mistake", "salt is", "you should"],
                "note": "play along, don't explain"}]),

    _s("h-mem-03", "long_memory", [
        "hey im nadia", "good thanks", "i work as a vet nurse",
        "mostly cats and dogs", "yeah it's rewarding", "long hours though",
        "twelve hour shifts sometimes", "yeah", "i had a coffee at 4am once",
        "anyway", "what do you think i do for work again?",
    ], probes=[{"turn": 11, "type": "long_memory", "expect_any": ["vet", "nurse", "animal"],
                "note": "recall occupation from turn 3"}]),

    _s("h-emo-02", "implicit_emotion", [
        "hey", "everyone's out tonight", "yeah they're at the thing",
        "no i wasn't really invited", "i mean it's fine",
        "it's a small thing apparently", "yeah", "im just gonna stay in",
        "watch something", "yeah",
    ], probes=[{"turn": 4, "type": "implicit_emotion",
                "forbid_any": ["nice", "sounds fun", "that's great", "enjoy"],
                "note": "exclusion, never stated"}]),

    _s("h-mem-04", "memory_update", [
        "hey", "my flight's at 6am", "yeah brutal",
        "oh they emailed, it's delayed to 11", "so 11 now", "bit better",
        "still gotta pack", "yeah tonight", "nearly done actually",
        "what time's my flight?",
    ], probes=[{"turn": 10, "type": "memory_update", "expect_any": ["11", "eleven"],
                "forbid_any": ["6am", "six"], "note": "updated time"}]),

    _s("h-sarc-02", "sarcasm", [
        "hey", "oh im doing GREAT", "my landlord raised the rent again. wonderful.",
        "and the heating's broken. perfect timing", "yeah im thrilled",
        "ok im not thrilled", "im pretty stressed actually", "yeah",
        "gonna call them tomorrow", "we'll see",
    ], probes=[{"turn": 3, "type": "sarcasm",
                "forbid_any": ["that's great", "congrats", "wonderful!", "awesome"],
                "note": "second sarcasm probe, different lexicon"}]),

    _s("h-long-02", "long_horizon", [
        "hey there", "pretty good", "been cooking a lot lately",
        "mostly pasta", "yeah i make the sauce from scratch",
        "takes ages but worth it", "tomatoes garlic basil",
        "yeah simple is better", "my flatmate loves it",
        "she's a chef actually", "yeah intimidating",
        "she says it's good though", "im improving i think",
        "next im trying bread", "yeah sourdough",
        "the starter takes a week", "im impatient",
        "we'll see how it goes", "ok gotta stir something", "byee",
    ], note="20 turns with an accumulating thread to keep straight"),
]


def hard_scenarios() -> list[Scenario]:
    return [Scenario.from_dict(s.to_dict()) for s in HARD_SCENARIOS]


def category_index() -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for s in HARD_SCENARIOS:
        idx.setdefault(s.category, []).append(s.id)
    return idx


def benchmark_strings() -> list[str]:
    """Every user-visible string in the benchmark; used for the leakage check."""
    return [t for s in HARD_SCENARIOS for t in s.user_turns]
