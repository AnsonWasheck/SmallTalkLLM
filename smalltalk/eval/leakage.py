"""Guard against training on every frozen held-out benchmark.

The overnight loop generates corpus data aimed at fixing benchmark failures. That
is legitimate *capability* work, but it becomes cheating the moment benchmark
strings themselves end up in training data. This module is the tripwire: it runs
before every training round and refuses corpora that overlap the benchmark.

We check 6-gram overlap rather than exact strings, because paraphrase-level copying
("what was my sister's name again?" -> "what was my sisters name again") would
otherwise slip through. Any hit is a rejection in v0.2; a corpus generation run
must be fixed, not averaged into an acceptable leakage rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..data.schema import Conversation
from .metrics import ngrams, words

NGRAM_N = 6
MAX_OVERLAP_RATE = 0.0


@dataclass
class LeakageReport:
    checked: int
    flagged: list[tuple[str, str]]
    overlap_rate: float
    ngrams_in_bench: int

    @property
    def clean(self) -> bool:
        return self.overlap_rate <= MAX_OVERLAP_RATE

    def summary(self) -> str:
        status = "CLEAN" if self.clean else "LEAKAGE DETECTED"
        s = (f"[leakage] {status}: {len(self.flagged)}/{self.checked} conversations "
             f"overlap the benchmark (rate {self.overlap_rate:.4f}, "
             f"threshold {MAX_OVERLAP_RATE})")
        for cid, snippet in self.flagged[:5]:
            s += f"\n           {cid}: {snippet!r}"
        return s


def benchmark_ngrams(n: int = NGRAM_N) -> set[tuple[str, ...]]:
    from .bench import default_scenarios
    from .bench_v2 import build_scenarios, verify_frozen
    from .hard_bench import hard_scenarios

    # Fail closed if the public frozen benchmark has drifted.
    verify_frozen()

    strings: list[str] = []
    for sc in list(hard_scenarios()) + list(default_scenarios()) + list(build_scenarios()):
        strings.extend(sc.user_turns)
        for p in sc.probes:
            strings.extend(p.expect_any)
    out: set[tuple[str, ...]] = set()
    for s in strings:
        out.update(ngrams(words(s), n))
    return out


def check_conversations(
    conversations: Iterable[Conversation], n: int = NGRAM_N
) -> LeakageReport:
    bench = benchmark_ngrams(n)
    flagged: list[tuple[str, str]] = []
    checked = 0
    for conv in conversations:
        checked += 1
        text = " ".join(m.content for m in conv.messages)
        hits = [g for g in ngrams(words(text), n) if g in bench]
        if hits:
            flagged.append((conv.id, " ".join(hits[0])))
    rate = len(flagged) / max(checked, 1)
    return LeakageReport(checked, flagged, rate, len(bench))


def filter_leaked(
    conversations: Sequence[Conversation], n: int = NGRAM_N
) -> tuple[list[Conversation], LeakageReport]:
    """Drop any conversation overlapping the benchmark. Fail closed."""
    bench = benchmark_ngrams(n)
    kept, flagged = [], []
    for conv in conversations:
        text = " ".join(m.content for m in conv.messages)
        hits = [g for g in ngrams(words(text), n) if g in bench]
        if hits:
            flagged.append((conv.id, " ".join(hits[0])))
        else:
            kept.append(conv)
    rate = len(flagged) / max(len(conversations), 1)
    return kept, LeakageReport(len(conversations), flagged, rate, len(bench))
