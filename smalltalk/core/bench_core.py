"""SmallTalk Core-Bench: deterministic reliability test on conversational primitives.

Construction: every intent's HELD-OUT paraphrases (never trained on) crossed with
a fixed set of surface variants. Scored at temperature 0, so the number is a
property of the model's argmax, not of a lucky sample. Re-running gives the same
answer to the token.

This is a *reliability* instrument, deliberately not a *quality* instrument. It
asks one question: given an ordinary conversational input, does the model choose
an appropriate short response? Long-memory, sarcasm and pragmatics are measured
by SmallTalkBench-v2 and are explicitly out of scope here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .intents import CORE_VERSION, INTENTS, Intent, normalise

# Fixed conversational preambles. Until v0.2.2 every scenario was evaluated on a
# freshly reset engine, so the benchmark was 100% single-turn and structurally
# incapable of measuring whether a reflex survives context -- which is exactly the
# axis a manual side-by-side test found to separate two checkpoints the score
# considered close. A reflex that only fires on turn one is not reliable.
#
# The preambles are deliberately neutral and carry no Core cue of their own, so a
# correct model answers the probe identically with or without them. Divergence is
# the measurement.
CONTEXTS: list[tuple[str, ...]] = [
    (),
    ("so what have you been up to", "not much, you?"),
    ("did you see the weather today", "yeah, grim isn't it"),
]

# Fixed surface transforms. Deterministic -- no RNG anywhere in this module.
VARIANTS = [
    lambda s: s,
    lambda s: s.rstrip("?!."),
    lambda s: s[:1].upper() + s[1:],
    lambda s: s.upper(),
    lambda s: "so " + s,
    lambda s: s + " lol",
]

# Reliability bars. Tier 1 states are the ones a user hits in the first message.
TIER_TARGETS = {1: 0.99, 2: 0.95, 3: 0.90}


@dataclass(frozen=True)
class CoreScenario:
    intent: str
    tier: int
    prompt: str
    context: tuple[str, ...] = ()      # alternating user/assistant turns

    @property
    def id(self) -> str:
        ctx = "|".join(normalise(c) for c in self.context)
        return f"{self.intent}::{normalise(self.prompt)}::{ctx}"


def build_scenarios() -> list[CoreScenario]:
    """Held-out paraphrases x surface transforms x conversational context.

    Surface transforms are applied only in the bare-context case: crossing all
    three axes would triple the runtime to re-measure the same surface robustness
    under context, when the question context asks is simply "does the right answer
    survive a preceding exchange".
    """
    out: list[CoreScenario] = []
    seen: set[str] = set()
    for it in INTENTS:
        for p in it.held_out:
            for v in VARIANTS:
                s = CoreScenario(it.name, it.tier, v(p), ())
                if s.id not in seen:
                    seen.add(s.id)
                    out.append(s)
            for ctx in CONTEXTS[1:]:
                s = CoreScenario(it.name, it.tier, p, ctx)
                if s.id not in seen:
                    seen.add(s.id)
                    out.append(s)
    return out


def checksum() -> str:
    """Identifies the exact test. A changed checksum invalidates comparisons."""
    h = hashlib.sha256(CORE_VERSION.encode())
    for s in build_scenarios():
        h.update(s.id.encode())
        h.update(b"\0")
    for it in INTENTS:
        h.update(f"{it.name}|{sorted(it.accepted())}".encode())
    return h.hexdigest()[:16]


def freeze(path: str | Path) -> str:
    """Write the checksum + full scenario list so drift is detectable later."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cs = checksum()
    p.write_text(json.dumps({
        "version": CORE_VERSION,
        "checksum": cs,
        "n_scenarios": len(build_scenarios()),
        "scenarios": [s.id for s in build_scenarios()],
    }, indent=2))
    return cs


def verify_frozen(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run freeze() before comparing runs")
    frozen = json.loads(p.read_text())["checksum"]
    now = checksum()
    if frozen != now:
        raise RuntimeError(
            f"Core-Bench drifted: frozen={frozen} current={now}. "
            "Editing intents invalidates every previously reported Core score."
        )


def score(intent_name: str, reply: str) -> bool:
    it = next(i for i in INTENTS if i.name == intent_name)
    return _strip_len(reply) and it.is_correct(_strip_len(reply))


def _strip_len(reply: str) -> str:
    r = reply.strip()
    for tok in ("<|len_reaction|>", "<|len_vshort|>", "<|len_short|>", "<|len_medium|>"):
        if r.startswith(tok):
            return r[len(tok):].strip()
    return r


def intent_of(s: CoreScenario) -> Intent:
    return next(i for i in INTENTS if i.name == s.intent)
