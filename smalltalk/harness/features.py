"""Deterministic feature extraction.

Conservative by design: features DESCRIBE the message, they do not rewrite it.
The user's casing, punctuation, typos, slang and emoji all reach the model
untouched, because those are exactly the surface cues a conversational model
should be reading. Nothing here is treated as semantic truth -- these are cheap
signals for the policy layer and for validation, not a replacement for the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GREETING = re.compile(r"\b(hi|hey|hello|yo|hiya|heya|howdy|morning|evening|sup)\b", re.I)
GOODBYE = re.compile(r"\b(bye|goodbye|see ya|see you|later|night|goodnight|gtg|"
                     r"heading off|i'm off|take care|catch you)\b", re.I)
THANKS = re.compile(r"\b(thanks|thank you|thx|ty|cheers|appreciate)\b", re.I)
APOLOGY = re.compile(r"\b(sorry|apolog|my bad|my fault)\b", re.I)
CORRECTION = re.compile(r"\b(actually|no wait|i meant|not quite|scratch that|"
                        r"correction|wrong)\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|didn't|don't|can't|won't|isn't|wasn't)\b", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
REPEATED_PUNCT = re.compile(r"([!?.])\1{1,}")


@dataclass
class Features:
    text: str
    n_chars: int = 0
    n_words: int = 0
    has_question_mark: bool = False
    has_greeting: bool = False
    has_goodbye: bool = False
    has_thanks: bool = False
    has_apology: bool = False
    has_correction: bool = False
    has_negation: bool = False
    has_emoji: bool = False
    all_caps: bool = False
    repeated_punct: bool = False
    prev_assistant_was_question: bool = False
    consecutive_assistant_questions: int = 0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "text"}


def extract(text: str, *, prev_assistant_was_question: bool = False,
            consecutive_assistant_questions: int = 0) -> Features:
    stripped = text.strip()
    words = stripped.split()
    letters = [c for c in stripped if c.isalpha()]
    return Features(
        text=text,
        n_chars=len(stripped),
        n_words=len(words),
        has_question_mark="?" in stripped,
        has_greeting=bool(GREETING.search(stripped)),
        has_goodbye=bool(GOODBYE.search(stripped)),
        has_thanks=bool(THANKS.search(stripped)),
        has_apology=bool(APOLOGY.search(stripped)),
        has_correction=bool(CORRECTION.search(stripped)),
        has_negation=bool(NEGATION.search(stripped)),
        has_emoji=bool(EMOJI.search(stripped)),
        # "all caps" needs enough letters to be a deliberate choice rather than
        # an artefact of a two-letter reply like "OK".
        all_caps=len(letters) >= 4 and all(c.isupper() for c in letters),
        repeated_punct=bool(REPEATED_PUNCT.search(stripped)),
        prev_assistant_was_question=prev_assistant_was_question,
        consecutive_assistant_questions=consecutive_assistant_questions,
    )
