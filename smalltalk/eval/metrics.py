"""Automatic conversational metrics + the primary research metric.

Nothing here needs an external model. These metrics are cheap proxies for
"obviously broken", which is what the primary research metric is defined on:

  PRIMARY: can the model complete a 10-turn casual conversation with zero
           obviously broken, repetitive or contextually nonsensical responses?

`broken_turns` implements "obviously broken" as an explicit, auditable rule set
so the capability cliff is measured the same way for every model size.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

WORD = re.compile(r"[a-z']+")

AI_STYLE_MARKERS = [
    "as an ai", "as a language model", "i'm an ai", "it's important to",
    "i hope this helps", "let me know if", "feel free to", "here are some",
    "in conclusion", "i'd be happy to", "certainly!", "great question",
    "i don't have personal", "step 1", "furthermore", "in summary",
]
CONFIDENT_FACT_MARKERS = [
    "was invented by", "was created by", "is located in", "the capital of",
    "the answer is", "developed by", "founded in", "according to",
]
GRACEFUL_IGNORANCE = [
    "no idea", "not sure", "dunno", "don't know", "no clue", "beats me",
    "never heard", "couldn't tell you", "?",
]


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(seq: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(texts: Sequence[str], n: int) -> float:
    total, uniq = 0, set()
    for t in texts:
        gs = ngrams(words(t), n)
        total += len(gs)
        uniq.update(gs)
    return len(uniq) / total if total else 0.0


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    """Fraction of n-grams inside one response that are duplicates."""
    gs = ngrams(words(text), n)
    if not gs:
        return 0.0
    counts = Counter(gs)
    return sum(c - 1 for c in counts.values()) / len(gs)


def type_token_ratio(texts: Sequence[str]) -> float:
    all_words = [w for t in texts for w in words(t)]
    return len(set(all_words)) / len(all_words) if all_words else 0.0


def self_bleu_like(texts: Sequence[str], n: int = 2) -> float:
    """Mean pairwise n-gram overlap between responses. High = repetitive persona."""
    sets = [set(ngrams(words(t), n)) for t in texts]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    scores = []
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            scores.append(len(a & b) / len(a | b))
    return sum(scores) / len(scores)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def context_copy_ratio(response: str, user_text: str) -> float:
    """Fraction of the response's words lifted verbatim from the user's turn."""
    rw = words(response)
    if not rw:
        return 0.0
    uw = set(words(user_text))
    return sum(w in uw for w in rw) / len(rw)


def question_ratio(texts: Sequence[str]) -> float:
    return sum("?" in t for t in texts) / len(texts) if texts else 0.0


def loop_detected(responses: Sequence[str], threshold: float = 0.8) -> bool:
    """Two consecutive near-identical replies, or one reply repeated 3+ times."""
    norm = [" ".join(words(r)) for r in responses]
    for a, b in zip(norm, norm[1:]):
        if a and a == b:
            return True
        if a and b and jaccard(a, b) >= threshold:
            return True
    counts = Counter(n for n in norm if n)
    return any(c >= 3 for c in counts.values())


def looks_ungrammatical(text: str) -> bool:
    """Crude, size-agnostic English well-formedness heuristics (RQ1).

    Deliberately conservative: it flags the failure modes tiny models actually
    exhibit (word salad, immediate token repetition, no vowels) rather than
    trying to be a real grammar checker.
    """
    w = words(text)
    if not w:
        return True
    if any(a == b for a, b in zip(w, w[1:])) and len(w) > 2:
        return True                                   # "i i think think"
    if sum(not re.search(r"[aeiouy]", x) for x in w) / len(w) > 0.35:
        return True                                   # consonant soup
    if len(w) > 4 and len(set(w)) / len(w) < 0.45:
        return True
    if re.search(r"(.)\1{4,}", text):
        return True                                   # "hhhhhh"
    return False


@dataclass
class TurnEval:
    index: int
    user: str
    response: str
    n_words: int
    rep_ngram: float
    copy_ratio: float
    is_question: bool
    empty: bool
    ungrammatical: bool
    ai_style: bool
    broken_reasons: list[str] = field(default_factory=list)

    @property
    def broken(self) -> bool:
        return bool(self.broken_reasons)


@dataclass
class ConversationEval:
    scenario_id: str
    category: str
    turns: list[TurnEval]
    metrics: dict[str, float]
    probe_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def broken_turns(self) -> int:
        return sum(t.broken for t in self.turns)

    @property
    def completed_clean(self) -> bool:
        return self.broken_turns == 0 and not self.metrics.get("loop", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "broken_turns": self.broken_turns,
            "completed_clean": self.completed_clean,
            "metrics": self.metrics,
            "probe_results": self.probe_results,
            "turns": [asdict(t) for t in self.turns],
        }


@dataclass
class BrokenThresholds:
    """The operational definition of 'obviously broken'. Tunable, documented."""

    max_words: int = 60
    min_words: int = 1
    max_rep_ngram: float = 0.34
    max_copy_ratio: float = 0.85
    min_copy_len: int = 4


def evaluate_transcript(
    messages: Sequence[dict[str, str]],
    scenario_id: str = "adhoc",
    category: str = "unknown",
    probes: Sequence[Any] = (),
    thresholds: BrokenThresholds | None = None,
) -> ConversationEval:
    th = thresholds or BrokenThresholds()
    turn_evals: list[TurnEval] = []
    responses: list[str] = []

    pairs = [
        (i, messages[i]["content"], messages[i + 1]["content"])
        for i in range(len(messages) - 1)
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant"
    ]

    for turn_no, (_, user_text, resp) in enumerate(pairs, start=1):
        w = words(resp)
        rep = repeated_ngram_ratio(resp, 3)
        copy = context_copy_ratio(resp, user_text)
        reasons: list[str] = []
        if not resp.strip() or resp.strip() == "...":
            reasons.append("empty")
        if len(w) > th.max_words:
            reasons.append("too_long")
        if rep > th.max_rep_ngram:
            reasons.append("intra_repetition")
        if copy >= th.max_copy_ratio and len(w) >= th.min_copy_len:
            reasons.append("context_copy")
        ungram = looks_ungrammatical(resp)
        if ungram:
            reasons.append("ungrammatical")
        ai_style = any(m in resp.lower() for m in AI_STYLE_MARKERS)
        if ai_style:
            reasons.append("ai_assistant_style")
        # Echoing a previous assistant reply verbatim.
        if any(" ".join(w) == " ".join(words(prev)) for prev in responses) and w:
            reasons.append("repeats_previous_turn")

        turn_evals.append(
            TurnEval(
                index=turn_no, user=user_text, response=resp, n_words=len(w),
                rep_ngram=round(rep, 4), copy_ratio=round(copy, 4),
                is_question="?" in resp, empty=not resp.strip(),
                ungrammatical=ungram, ai_style=ai_style, broken_reasons=reasons,
            )
        )
        responses.append(resp)

    lengths = [t.n_words for t in turn_evals] or [0]
    metrics = {
        "num_turns": float(len(turn_evals)),
        "mean_len": round(sum(lengths) / len(lengths), 2),
        "max_len": float(max(lengths)),
        "len_in_3_25": round(sum(3 <= n <= 25 for n in lengths) / len(lengths), 4),
        "distinct_1": round(distinct_n(responses, 1), 4),
        "distinct_2": round(distinct_n(responses, 2), 4),
        "ttr": round(type_token_ratio(responses), 4),
        "self_overlap": round(self_bleu_like(responses), 4),
        "mean_rep_ngram": round(sum(t.rep_ngram for t in turn_evals) / len(turn_evals), 4)
        if turn_evals else 0.0,
        "question_ratio": round(question_ratio(responses), 4),
        "mean_copy_ratio": round(
            sum(t.copy_ratio for t in turn_evals) / max(len(turn_evals), 1), 4
        ),
        "loop": 1.0 if loop_detected(responses) else 0.0,
        "ai_style_rate": round(
            sum(t.ai_style for t in turn_evals) / max(len(turn_evals), 1), 4
        ),
        "excessive_questions": 1.0 if question_ratio(responses) > 0.8 else 0.0,
    }

    probe_results = [_run_probe(p, responses) for p in probes]
    for pr in probe_results:
        if not pr["passed"] and 1 <= pr["turn"] <= len(turn_evals):
            turn_evals[pr["turn"] - 1].broken_reasons.append(f"probe_{pr['type']}")

    return ConversationEval(scenario_id, category, turn_evals, metrics, probe_results)


def _run_probe(probe: Any, responses: Sequence[str]) -> dict[str, Any]:
    turn = getattr(probe, "turn", probe.get("turn") if isinstance(probe, dict) else 0)
    ptype = getattr(probe, "type", probe.get("type") if isinstance(probe, dict) else "")
    expect = list(getattr(probe, "expect_any", []) or [])
    forbid = list(getattr(probe, "forbid_any", []) or [])
    forbid_conf = bool(getattr(probe, "forbid_confident", False))

    if turn < 1 or turn > len(responses):
        return {"turn": turn, "type": ptype, "passed": False, "reason": "missing_turn"}
    resp = responses[turn - 1].lower()

    passed, reason = True, "ok"
    if expect and not any(e.lower() in resp for e in expect):
        passed, reason = False, "expected_content_absent"
    if passed and forbid and any(f.lower() in resp for f in forbid):
        passed, reason = False, "forbidden_content_present"
    if passed and forbid_conf:
        fabricated = any(m in resp for m in CONFIDENT_FACT_MARKERS)
        graceful = any(g in resp for g in GRACEFUL_IGNORANCE)
        if fabricated and not graceful:
            passed, reason = False, "fabricated_fact"
    return {"turn": turn, "type": ptype, "passed": passed, "reason": reason}


def aggregate(evals: Sequence[ConversationEval], min_turns: int = 10) -> dict[str, Any]:
    """Aggregate across scenarios. `clean_10turn_rate` is the primary metric."""
    if not evals:
        return {}
    keys = evals[0].metrics.keys()
    agg = {
        k: round(sum(e.metrics.get(k, 0.0) for e in evals) / len(evals), 4) for k in keys
    }
    long_enough = [e for e in evals if e.metrics.get("num_turns", 0) >= min_turns]
    five = [e for e in evals if e.metrics.get("num_turns", 0) >= 5]

    def clean_prefix(e: ConversationEval, n: int) -> bool:
        head = e.turns[:n]
        return bool(head) and not any(t.broken for t in head) and not loop_detected(
            [t.response for t in head]
        )

    probes = [p for e in evals for p in e.probe_results]
    agg.update(
        {
            "scenarios": len(evals),
            "broken_turn_rate": round(
                sum(e.broken_turns for e in evals)
                / max(sum(len(e.turns) for e in evals), 1), 4
            ),
            "clean_conversation_rate": round(
                sum(e.completed_clean for e in evals) / len(evals), 4
            ),
            "clean_5turn_rate": round(
                sum(clean_prefix(e, 5) for e in five) / max(len(five), 1), 4
            ),
            "clean_10turn_rate": round(
                sum(clean_prefix(e, min_turns) for e in long_enough) / max(len(long_enough), 1), 4
            ),
            "loop_rate": round(sum(e.metrics.get("loop", 0.0) for e in evals) / len(evals), 4),
            "probe_pass_rate": round(
                sum(p["passed"] for p in probes) / max(len(probes), 1), 4
            ),
            "grammatical_rate": round(
                1.0 - sum(t.ungrammatical for e in evals for t in e.turns)
                / max(sum(len(e.turns) for e in evals), 1), 4
            ),
        }
    )
    by_cat: dict[str, dict[str, float]] = {}
    for e in evals:
        b = by_cat.setdefault(e.category, {"n": 0, "clean": 0})
        b["n"] += 1
        b["clean"] += float(e.completed_clean)
    agg["by_category"] = {
        c: round(v["clean"] / v["n"], 3) for c, v in sorted(by_cat.items())
    }
    return agg


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))
