"""Confidence measures over a policy distribution.

Three cheap statistics, reported together because they disagree in informative
ways: a distribution can have a high top-1 and still be unsafe if the runner-up
is nearly tied, and a low top-1 over a flat 19-way distribution is a different
situation from a low top-1 over two strong contenders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Confidence:
    top1: float
    margin: float          # top1 - top2
    entropy: float         # nats, normalised by ln(n)
    status: str            # HIGH | MEDIUM | LOW

    def as_dict(self) -> dict:
        return {"top1": round(self.top1, 4), "margin": round(self.margin, 4),
                "entropy": round(self.entropy, 4), "status": self.status}


def score(probs: dict[str, float], *, min_top1: float, min_margin: float) -> Confidence:
    if not probs:
        return Confidence(0.0, 0.0, 1.0, "LOW")
    ordered = sorted(probs.values(), reverse=True)
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    margin = top1 - top2
    ent = -sum(p * math.log(p) for p in probs.values() if p > 0)
    ent /= math.log(len(probs)) if len(probs) > 1 else 1.0

    if top1 >= min_top1 and margin >= min_margin:
        status = "HIGH"
    elif top1 >= min_top1 * 0.75 and margin >= min_margin * 0.5:
        # AND, not OR: a distribution can have a respectable top-1 and still be
        # unsafe when the runner-up is nearly tied, which is exactly the
        # "today was weird" case the confidence gate exists to catch.
        status = "MEDIUM"
    else:
        status = "LOW"
    return Confidence(top1, margin, ent, status)
