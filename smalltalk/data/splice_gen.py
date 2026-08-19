"""Splice generator: real human language + generated memory/epistemic structure.

Why this exists
---------------
Three corpora have now been measured and each fails on one axis:

  * DailyDialog / EmpatheticDialogues -- real human language, genuine entropy, but
    almost no cross-turn fact recall. Trains language, not reference.
  * social_gold -- correct texting register, but 89% utterance-repeat.
  * combi_gen (templates) -- correct structure, but 97% repeat and a ~15-value slot
    bank, which is precisely what taught the student `surface -> memorised slot`
    (measured: 100% of memory answers contained a training entity, 0% the user's).

This module composes the *language* of the real corpora with the *structure* of the
latent planner, using open-ended generated entities so no closed value bank exists to
memorise. Real turns supply the connective tissue and the token distance; the injected
turns supply the skill being taught.

Every emitted conversation carries `meta.family` and `meta.skill` so it flows through
the existing family-split and leakage infrastructure unchanged.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from ..eval.metrics import words
from .schema import Conversation, Turn

SPLICE_VERSION = "splice-v1.0.0"

# --- how the assistant states / recalls, phrased many ways -------------------
_ACK_FACT = [
    "oh nice", "ah ok", "gotcha", "oh cool", "right", "noted", "ah nice one",
    "oh lovely", "makes sense", "ha, good name", "oh that's sweet", "nice",
]
_RECALL_LEAD = [
    "{v}", "{v}, right?", "{v} i think", "it was {v}", "{v} wasn't it",
    "{v} — you said earlier", "pretty sure you said {v}", "{v}, yeah?",
]
_DONT_KNOW = [
    "you never told me", "don't think you said", "you haven't mentioned that",
    "no idea, you didn't say", "hmm you never said", "i don't think you told me that",
    "you've not said actually", "dunno, you didn't mention it",
]
_UPDATED = [
    "{v} now", "{v}, since it changed", "{v} — you changed it", "{v} these days",
    "it's {v} now isn't it", "{v}, after the change",
]
_ASK_WHICH = [
    "which one?", "wait which?", "hang on, which one do you mean?",
    "sorry, which one?", "which of them?", "you mean which one?",
]

_QUESTION_FORMS = {
    "pet_name": ["what's my {kind} called again?", "remind me what i called my {kind}?",
                 "what did i say my {kind}'s name was?", "my {kind}'s name — what was it?"],
    "person_name": ["what's my {rel}'s name again?", "remind me my {rel}'s name?",
                    "what did i say my {rel} was called?"],
    "job": ["what do i do for work again?", "remind me what my job is?",
            "what did i say i did for a living?"],
    "hobby": ["what was that thing i took up?", "what did i say i'd started?",
              "remind me what hobby i picked up?"],
    "place": ["where did i say i was going?", "remind me where i'm off to?",
              "where was i headed again?"],
}

_STATE_FORMS = {
    "pet_name": ["i got a {kind}, called {v}", "we named the {kind} {v}",
                 "my {kind}'s called {v}", "got a {kind} — {v}"],
    "person_name": ["my {rel}'s called {v}", "my {rel} {v} is around",
                    "{v}, my {rel}, is visiting"],
    "job": ["i'm a {v}", "i work as a {v}", "i've been a {v} for a while"],
    "hobby": ["i've taken up {v}", "i started {v} recently", "i've got into {v}"],
    "place": ["i'm off to {v}", "heading to {v}", "going to {v} soon"],
}

_CORRECTION_FORMS = [
    "actually it's {v} not {old}", "sorry, {v} — i said {old} by mistake",
    "wait no, {v}. i got that wrong", "correction: {v}, not {old}",
    "hang on i meant {v}", "ignore that, it's {v}",
]


@dataclass
class SpliceConfig:
    n: int = 12000
    seed: int = 0
    # skill mix for the spliced portion (structure-bearing only)
    mix: dict = field(default_factory=lambda: {
        "long_memory": 0.24,
        "memory_absent": 0.18,
        "memory_update": 0.18,
        "epistemic_known": 0.14,
        "ambiguous_referent": 0.10,
        "two_entities": 0.10,
        "persona_stable": 0.06,
    })
    min_gap_turns: int = 2
    max_gap_turns: int = 14
    # token distance buckets, measured with the STUDENT tokenizer downstream
    long_gap_frac: float = 0.35


def _clean_real_turn(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t


def _load_real_turn_pool(convs: Sequence[Conversation], min_words=2, max_words=28
                         ) -> list[tuple[str, str]]:
    """(user_turn, assistant_turn) adjacency pairs harvested from real dialogue."""
    pool: list[tuple[str, str]] = []
    for c in convs:
        ms = c.messages
        for i in range(len(ms) - 1):
            if ms[i].role == "user" and ms[i + 1].role == "assistant":
                u, a = _clean_real_turn(ms[i].content), _clean_real_turn(ms[i + 1].content)
                if min_words <= len(words(u)) <= max_words and \
                   min_words <= len(words(a)) <= max_words:
                    pool.append((u, a))
    return pool


def generate(real_convs: Sequence[Conversation], cfg: SpliceConfig | None = None
             ) -> Iterator[Conversation]:
    cfg = cfg or SpliceConfig()
    r = random.Random(cfg.seed)
    pool = _load_real_turn_pool(real_convs)
    if len(pool) < 200:
        raise ValueError(f"real turn pool too small ({len(pool)}); need DailyDialog/ED")

    from ..qwen.planner import (_HOBBIES, _JOBS, _PET_KINDS, _PLACES,
                                _name, _petname)

    skills = list(cfg.mix)
    weights = [cfg.mix[s] for s in skills]

    for i in range(cfg.n):
        skill = r.choices(skills, weights=weights)[0]
        fam = f"splice:{skill}:{r.randrange(300):03d}"
        msgs: list[Turn] = []

        # --- pick the fact type and its open-ended value --------------------
        ftype = r.choice(list(_STATE_FORMS))
        rel = r.choice(["sister", "brother", "flatmate", "mate", "cousin", "neighbour",
                        "partner", "coworker", "mum", "dad"])
        kind = r.choice(_PET_KINDS)
        if ftype == "pet_name":
            val, alt = _petname(r), _petname(r)
        elif ftype == "person_name":
            val, alt = _name(r), _name(r)
        elif ftype == "job":
            val, alt = r.choice(_JOBS), r.choice(_JOBS)
        elif ftype == "hobby":
            val, alt = r.choice(_HOBBIES), r.choice(_HOBBIES)
        else:
            val, alt = r.choice(_PLACES), r.choice(_PLACES)
        while alt == val:
            alt = _petname(r) if ftype == "pet_name" else _name(r)

        def fmt(t: str, v: str) -> str:
            return t.format(v=v, kind=kind, rel=rel, old=val)

        # --- opening real exchange -----------------------------------------
        u0, a0 = r.choice(pool)
        msgs.append(Turn("user", u0))
        msgs.append(Turn("assistant", a0))

        stated = skill != "memory_absent"
        if stated:
            msgs.append(Turn("user", fmt(r.choice(_STATE_FORMS[ftype]), val)))
            msgs.append(Turn("assistant", r.choice(_ACK_FACT)))

        second_val = None
        if skill == "two_entities":
            second_val = alt
            rel2 = r.choice(["mate", "cousin", "coworker", "neighbour"])
            msgs.append(Turn("user", f"my {rel2}'s is called {alt} confusingly"))
            msgs.append(Turn("assistant", r.choice(["ha, two of them", "confusing",
                                                    "noted, two names to keep straight"])))

        # --- REAL conversation as the gap (this is the entropy source) ------
        gap = r.randint(cfg.min_gap_turns, cfg.max_gap_turns)
        if r.random() < cfg.long_gap_frac:
            gap = r.randint(cfg.max_gap_turns, cfg.max_gap_turns + 12)
        for _ in range(gap):
            u, a = r.choice(pool)
            msgs.append(Turn("user", u))
            msgs.append(Turn("assistant", a))

        # --- optional correction --------------------------------------------
        answer_val = val
        if skill == "memory_update" and stated:
            new = alt
            msgs.append(Turn("user", fmt(r.choice(_CORRECTION_FORMS), new)))
            msgs.append(Turn("assistant", r.choice(["ah ok, got it", "noted", "right, {} then".format(new)])))
            answer_val = new
            # more real conversation after the correction, so recall is not adjacent
            for _ in range(r.randint(1, 6)):
                u, a = r.choice(pool)
                msgs.append(Turn("user", u))
                msgs.append(Turn("assistant", a))

        # --- the probe -------------------------------------------------------
        if skill == "ambiguous_referent":
            msgs.append(Turn("user", r.choice(["did it go ok in the end?",
                                               "was it any good?",
                                               "how did that one turn out?"])))
            msgs.append(Turn("assistant", r.choice(_ASK_WHICH)))
        elif skill == "persona_stable":
            msgs.append(Turn("user", "you still there?"))
            msgs.append(Turn("assistant", r.choice(["yeah still here", "yep, listening",
                                                    "here, go on"])))
        else:
            q = r.choice(_QUESTION_FORMS[ftype]).format(kind=kind, rel=rel)
            msgs.append(Turn("user", q))
            if skill == "memory_absent":
                ans = r.choice(_DONT_KNOW)
            elif skill == "memory_update":
                ans = r.choice(_UPDATED).format(v=answer_val)
            else:
                ans = r.choice(_RECALL_LEAD).format(v=answer_val)
            msgs.append(Turn("assistant", ans))

        yield Conversation(
            id=f"splice-{i:06d}",
            messages=msgs,
            source="splice_gen",
            meta={"family": fam, "skill": skill, "fact_type": ftype,
                  "answer": answer_val if skill != "memory_absent" else None,
                  "gap_turns": gap, "splice_version": SPLICE_VERSION},
        )
