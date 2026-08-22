"""Repetition detection.

Deterministic, and deliberately tolerant: "yeah" and "mm" recurring in casual
conversation is correct behaviour, not a defect. What we suppress is the
degenerate case -- the same substantive reply twice in a row, or a stuck opening.
"""

from __future__ import annotations

import re

TRIVIAL = {"yeah", "mm", "right", "ok", "okay", "haha", "ha", "mhm", "sure", "yep",
           "no worries", "see you", "hey", "hi"}


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z']+", t.lower())


def is_exact_repeat(reply: str, recent: list[str]) -> bool:
    r = reply.strip().lower()
    if r in TRIVIAL:                    # short acknowledgements may recur freely
        return False
    return r in {x.strip().lower() for x in recent}


def shares_long_ngram(reply: str, recent: list[str], n: int = 4) -> bool:
    w = _words(reply)
    if len(w) < n:
        return False
    grams = {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    for prev in recent:
        pw = _words(prev)
        pg = {tuple(pw[i:i + n]) for i in range(len(pw) - n + 1)}
        if grams & pg:
            return True
    return False


def internal_loop(reply: str, n: int = 3) -> bool:
    """A phrase repeated inside one reply -- the classic small-model failure."""
    w = _words(reply)
    if len(w) < n * 2:
        return False
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(grams) != len(set(grams))
