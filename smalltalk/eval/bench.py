"""SmallTalkBench: held-out multi-turn casual conversation scenarios.

Each scenario is a *fixed script of user turns*. We drive the model through the
whole conversation and score the resulting transcript, not isolated replies --
conversational collapse is a trajectory property, so single-response evaluation
would miss the exact thing we are trying to locate.

Scenario fields:
  id, category, user_turns, probes
Probes attach machine-checkable expectations to specific turn indices:
  {"turn": 6, "type": "memory", "expect_any": ["emma"]}   -> recall an earlier fact
  {"turn": 3, "type": "unknown_fact", "forbid_confident": true}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

CATEGORIES = (
    "greeting", "bad_day", "good_day", "boredom", "excitement", "story",
    "topic_change", "emotional_disclosure", "joke", "short_answers",
    "ambiguous", "memory", "unknown_fact", "goodbye",
)


@dataclass
class Probe:
    turn: int
    type: str
    expect_any: list[str] = field(default_factory=list)
    forbid_any: list[str] = field(default_factory=list)
    forbid_confident: bool = False
    note: str = ""


@dataclass
class Scenario:
    id: str
    category: str
    user_turns: list[str]
    probes: list[Probe] = field(default_factory=list)
    note: str = ""

    @property
    def num_turns(self) -> int:
        return len(self.user_turns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        return cls(
            id=d["id"],
            category=d["category"],
            user_turns=list(d["user_turns"]),
            probes=[Probe(**p) for p in d.get("probes", [])],
            note=d.get("note", ""),
        )


def _s(id, category, turns, probes=(), note="") -> Scenario:
    return Scenario(id, category, list(turns), [Probe(**p) for p in probes], note)


# Words that signal a fabricated factual claim about something obscure.
CONFIDENT_FACT_MARKERS = [
    "was invented", "was created by", "is located in", "in 18", "in 19", "in 20",
    "the answer is", "it was made", "developed by", "the capital",
]

SCENARIOS: list[Scenario] = [
    _s("greet-01", "greeting", [
        "hey", "not much, you?", "same honestly", "yeah pretty quiet day",
        "just kinda relaxing", "might watch something later", "you into movies?",
        "nice, what kind", "cool", "alright im gonna go, later",
    ], note="10-turn baseline: the primary research metric runs on this shape"),
    _s("bad-01", "bad_day", [
        "hey", "honestly kinda tired today", "yeah work was brutal",
        "mostly meetings", "like six of them back to back", "yeah i barely ate",
        "i just wanna sleep", "probably early tonight", "thanks, i needed that",
        "night",
    ], probes=[{"turn": 3, "type": "acknowledge", "expect_any": ["?", "rough", "oof", "damn", "sucks", "sorry", "long"]}]),
    _s("good-01", "good_day", [
        "yo", "actually a really good day", "i got the job!!", "yeah i start monday",
        "im so relieved honestly", "gonna celebrate tonight", "just dinner with friends",
        "italian place downtown", "cant wait", "ok gotta go get ready",
    ]),
    _s("bored-01", "boredom", [
        "hey", "im so bored", "nothing to do", "yeah tried that already",
        "eh not feeling it", "idk", "maybe", "sure i guess", "hm ok",
        "yeah alright",
    ], note="deliberately low-information user turns; tests initiative without interrogation"),
    _s("excite-01", "excitement", [
        "hey hey", "guess what", "im going to japan next month!", "two weeks",
        "tokyo and kyoto mostly", "ive wanted to go forever", "yeah im so hyped",
        "gonna eat so much ramen", "ok i should pack", "bye!",
    ]),
    _s("story-01", "story", [
        "hey you free", "ok so something weird happened today",
        "i was on the bus and this guy started singing", "like full opera voice",
        "everyone just stared", "then people started clapping",
        "he took a bow and got off at the next stop", "best commute ever honestly",
        "yeah made my whole day", "anyway thats my story",
    ]),
    _s("topic-01", "topic_change", [
        "hey", "work was fine i guess", "anyway did you eat yet",
        "im thinking pasta", "actually forget food, do you play any games?",
        "im replaying an old rpg", "yeah super nostalgic", "oh also its raining here",
        "kinda nice actually", "ok im heading out",
    ], note="three abrupt topic switches; tests whether the model follows the user"),
    _s("emo-01", "emotional_disclosure", [
        "hi", "kind of a weird week", "i've been feeling pretty lonely lately",
        "yeah moved to a new city in march", "dont really know anyone here",
        "its just quiet all the time", "i try but its hard",
        "yeah maybe i should join something", "thanks for listening",
        "talk tomorrow?",
    ], probes=[{"turn": 3, "type": "emotion", "forbid_any": ["consult", "professional help", "as an ai", "i'm sorry to hear that, however"]}]),
    _s("joke-01", "joke", [
        "hey", "i tried to cook today", "set off the smoke alarm twice",
        "the recipe said simmer, i chose violence", "yeah im banned from the kitchen now",
        "my roommate ordered pizza instead", "honestly better outcome",
        "do you cook?", "haha fair", "later",
    ]),
    _s("short-01", "short_answers", [
        "hey", "yeah", "no", "kinda", "maybe", "sure", "hm", "idk", "ok", "bye",
    ], note="hard case: near-zero user signal, tests degeneration into loops"),
    _s("ambig-01", "ambiguous", [
        "hey", "so that happened", "you know, the thing", "the thing from yesterday",
        "nevermind lol", "it wasnt important", "anyway how are you",
        "thats good", "yeah", "ok talk later",
    ], probes=[{"turn": 2, "type": "clarify", "expect_any": ["?"]}]),
    _s("mem-01", "memory", [
        "hey im dave", "pretty good, my dog just had surgery",
        "her name's emma, she's a beagle", "yeah she's recovering ok",
        "vet said two weeks of rest", "shes so annoyed about it lol",
        "keeps trying to jump on the couch", "do you remember my dog's name?",
        "yeah thats her", "thanks, bye",
    ], probes=[
        {"turn": 8, "type": "memory", "expect_any": ["emma"], "note": "recall from turn 3"},
    ]),
    _s("unknown-01", "unknown_fact", [
        "hey", "random question", "who invented the spinning jenny mule frame thing",
        "yeah i saw it in a museum", "no idea what it does either",
        "textiles maybe?", "anyway not important", "what did you do today",
        "sounds chill", "cya",
    ], probes=[
        {"turn": 3, "type": "unknown_fact", "forbid_confident": True,
         "note": "graceful ignorance beats fabrication"},
    ]),
    _s("bye-01", "goodbye", [
        "hey quick one", "just wanted to say hi", "yeah been busy",
        "work mostly", "its fine, almost done with the big project",
        "friday hopefully", "yeah ill celebrate", "anyway i gotta run",
        "talk this weekend?", "cool, bye",
    ]),
]


def default_scenarios() -> list[Scenario]:
    return [Scenario.from_dict(s.to_dict()) for s in SCENARIOS]


def save_scenarios(path: str | Path, scenarios: Sequence[Scenario] | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in scenarios or default_scenarios():
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    return p


def load_scenarios(path: str | Path | None = None) -> list[Scenario]:
    if path is None:
        return default_scenarios()
    p = Path(path)
    if not p.exists():
        return default_scenarios()
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Scenario.from_dict(json.loads(line)))
    return out
