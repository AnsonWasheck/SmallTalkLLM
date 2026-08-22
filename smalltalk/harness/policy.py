"""Conversational policy: a small constrained decision made BEFORE generation.

The Cactus-Needle idea, adapted from tool routing to conversational policy: do
not ask a 6.7M model to reason about meaning, length, question restraint and
phrasing simultaneously in one open-ended forward pass. Make one tiny discrete
decision first, with a bounded candidate set, then let the model do the thing it
is actually good at -- producing short natural language.

HOW THE DECISION IS MADE, and why it adds zero parameters
---------------------------------------------------------
Each policy owns a handful of EXEMPLAR replies. To infer the policy we score
log P(exemplar | conversation) under the unchanged model and take the policy
whose exemplars the model finds most probable. This reads the model's own
implicit belief about what should come next, using nothing but forward passes.

The exemplars are never emitted. They are a measuring instrument, not a response
bank -- the visible reply is always generated. Returning an exemplar directly
would be templating, which this experiment explicitly forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    ACKNOWLEDGE = "ACKNOWLEDGE"
    RECIPROCATE = "RECIPROCATE"
    CELEBRATE = "CELEBRATE"
    EMPATHIZE = "EMPATHIZE"
    CLARIFY = "CLARIFY"
    PLAY_ALONG = "PLAY_ALONG"
    CLOSE = "CLOSE"
    CONTINUE = "CONTINUE"
    NEUTRAL_REACTION = "NEUTRAL_REACTION"


class QuestionPolicy(str, Enum):
    NO_QUESTION = "NO_QUESTION"
    ALLOWED = "QUESTION_ALLOWED"
    PREFERRED = "QUESTION_PREFERRED"


class Length(str, Enum):
    REACTION = "REACTION"
    VERY_SHORT = "VERY_SHORT"
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"


# Token ceilings per length class, measured in tokenizer tokens rather than
# words: the byte-level BPE does not align with word counts.
LENGTH_TOKENS: dict[Length, int] = {
    Length.REACTION: 4,
    Length.VERY_SHORT: 8,
    Length.SHORT: 16,
    Length.MEDIUM: 32,
}


@dataclass(frozen=True)
class Policy:
    pid: str                       # compact id, e.g. "P07"
    intent: str
    action: Action
    question: QuestionPolicy
    length: Length
    exemplars: tuple[str, ...]     # scoring probes ONLY -- never emitted

    def __str__(self) -> str:
        return (f"{self.pid}:{self.intent}|{self.action.value}|"
                f"{self.question.value}|{self.length.value}")


def _p(pid, intent, action, question, length, *exemplars) -> Policy:
    return Policy(pid, intent, action, question, length, tuple(exemplars))


# The ontology. Kept small on purpose: every additional policy is another way
# for the classifier to be wrong, and combinatorial explosion buys nothing.
POLICIES: list[Policy] = [
    _p("P01", "GREETING", Action.RECIPROCATE, QuestionPolicy.ALLOWED, Length.VERY_SHORT,
       "hey", "hi", "hey there"),
    _p("P02", "GREETING_HOW_ARE_YOU", Action.RECIPROCATE, QuestionPolicy.PREFERRED, Length.SHORT,
       "i'm good, how about you?", "good thanks, you?"),
    _p("P03", "HOW_ARE_YOU", Action.RECIPROCATE, QuestionPolicy.PREFERRED, Length.SHORT,
       "not bad, you?", "pretty good, you?"),
    _p("P04", "THANKS", Action.ACKNOWLEDGE, QuestionPolicy.NO_QUESTION, Length.REACTION,
       "no worries", "any time", "of course"),
    _p("P05", "APOLOGY", Action.ACKNOWLEDGE, QuestionPolicy.NO_QUESTION, Length.VERY_SHORT,
       "it's alright", "no harm done"),
    _p("P06", "GOODBYE", Action.CLOSE, QuestionPolicy.NO_QUESTION, Length.VERY_SHORT,
       "see you", "see ya", "night"),
    _p("P07", "GOOD_NEWS", Action.CELEBRATE, QuestionPolicy.ALLOWED, Length.SHORT,
       "that's great", "nice one", "congrats"),
    _p("P08", "BAD_NEWS", Action.EMPATHIZE, QuestionPolicy.ALLOWED, Length.SHORT,
       "that sounds rough", "sorry to hear that", "oh no"),
    _p("P09", "TIRED", Action.EMPATHIZE, QuestionPolicy.ALLOWED, Length.VERY_SHORT,
       "long day?", "rough one?"),
    _p("P10", "BORED", Action.RECIPROCATE, QuestionPolicy.ALLOWED, Length.SHORT,
       "same here", "yeah it's a slow one"),
    _p("P11", "VENTING", Action.EMPATHIZE, QuestionPolicy.NO_QUESTION, Length.SHORT,
       "that sounds hard", "ugh, that's rough"),
    _p("P12", "CONFUSION", Action.CLARIFY, QuestionPolicy.PREFERRED, Length.VERY_SHORT,
       "what do you mean?", "sorry, which bit?"),
    _p("P13", "AGREEMENT", Action.ACKNOWLEDGE, QuestionPolicy.NO_QUESTION, Length.REACTION,
       "yeah exactly", "right?"),
    _p("P14", "DISAGREEMENT", Action.CLARIFY, QuestionPolicy.ALLOWED, Length.SHORT,
       "really? i don't see it", "hm, i'm not sure"),
    _p("P15", "JOKE", Action.PLAY_ALONG, QuestionPolicy.NO_QUESTION, Length.REACTION,
       "haha", "ha, good one"),
    _p("P16", "TOPIC_STATEMENT", Action.CONTINUE, QuestionPolicy.PREFERRED, Length.SHORT,
       "oh nice, how's that going?", "oh really? how come"),
    _p("P17", "AMBIGUOUS", Action.CLARIFY, QuestionPolicy.PREFERRED, Length.VERY_SHORT,
       "how do you mean?", "in what way?"),
    _p("P18", "ACKNOWLEDGEMENT", Action.NEUTRAL_REACTION, QuestionPolicy.NO_QUESTION,
       Length.REACTION, "mm", "right", "fair enough"),
    _p("P19", "UNKNOWN", Action.ACKNOWLEDGE, QuestionPolicy.NO_QUESTION, Length.VERY_SHORT,
       "no idea honestly", "couldn't tell you"),
]

BY_ID = {p.pid: p for p in POLICIES}
BY_INTENT = {p.intent: p for p in POLICIES}

# Where the policy distribution is flat, these are the safe places to land: they
# commit to little and are appropriate under most states. "today was weird"
# should get "weird how?", not confident sympathy for a day that may have been
# fine.
CONSERVATIVE = BY_ID["P17"]        # AMBIGUOUS -> clarify
CONSERVATIVE_CLOSING = BY_ID["P06"]


def consistency(candidate: str, policy: Policy) -> float:
    """How well a MODEL-GENERATED candidate matches a policy, in [0, 1].

    Token-level F1 against the policy's exemplars. This selects among things the
    model already produced; it never emits exemplar text, and a candidate that
    shares no words with any exemplar is not rejected -- it merely loses a
    tiebreak to one that does.

    Caveat worth stating plainly: this biases selection toward exemplar-like
    phrasing, which is the closest this harness comes to templating. It is
    included because the alternative -- constraints on length and question marks
    alone -- was measured to discriminate almost nothing, leaving the reranker
    unable to exploit the headroom the candidate set demonstrably contains.
    """
    import re
    def toks(t: str) -> set:
        return set(re.findall(r"[a-z']+", t.lower()))
    c = toks(candidate)
    if not c:
        return 0.0
    best = 0.0
    for ex in policy.exemplars:
        e = toks(ex)
        if not e:
            continue
        inter = len(c & e)
        if inter:
            prec, rec = inter / len(c), inter / len(e)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def high_confidence_shortcut(f) -> Policy | None:
    """Deterministic overrides, used ONLY where the surface cue is unambiguous.

    These are conversational invariants, not benchmark answers: a message that is
    literally "thanks" is a thanks, and a farewell should not be answered with a
    question in any conversation. Anything less certain is left to the model.
    """
    text = f.text.strip().lower()
    if f.has_goodbye and f.n_words <= 6:
        return BY_ID["P06"]
    if f.has_thanks and f.n_words <= 5 and not f.has_question_mark:
        return BY_ID["P04"]
    if f.has_greeting and f.n_words <= 3 and not f.has_question_mark:
        return BY_ID["P01"]
    return None
