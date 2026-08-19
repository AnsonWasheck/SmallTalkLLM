"""Stage 3: candidate scoring, rejection sampling and best-of-N selection.

The scorer is a pluggable protocol so scores can come from a heuristic, a human,
an external LLM or a local model. Only *offline* selection uses it -- the deployed
artifact is the micro model alone, with no teacher at inference time.

Scoring dimensions (all higher = better):
  naturalness, brevity, relevance, emotional_appropriateness, continuation,
  non_repetition, no_unnecessary_facts, non_assistant_style
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from ..data.schema import Candidate, Conversation, write_jsonl
from ..eval.metrics import (
    AI_STYLE_MARKERS,
    CONFIDENT_FACT_MARKERS,
    GRACEFUL_IGNORANCE,
    context_copy_ratio,
    jaccard,
    looks_ungrammatical,
    repeated_ngram_ratio,
    words,
)

DIMENSIONS = (
    "naturalness",
    "brevity",
    "relevance",
    "emotional_appropriateness",
    "continuation",
    "non_repetition",
    "no_unnecessary_facts",
    "non_assistant_style",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "naturalness": 1.5,
    "brevity": 1.0,
    "relevance": 1.5,
    "emotional_appropriateness": 1.0,
    "continuation": 1.0,
    "non_repetition": 1.0,
    "no_unnecessary_facts": 0.75,
    "non_assistant_style": 1.25,
}


class Scorer(Protocol):
    """Return a {dimension: 1-5} dict for one candidate in context."""

    def score(self, context: Sequence[dict[str, str]], candidate: str) -> dict[str, float]:
        ...


# ---------------------------------------------------------------------------
# Heuristic scorer (no external dependency; the default)
# ---------------------------------------------------------------------------
EMOTION_CUES = {
    "negative": ("tired", "sad", "bad", "awful", "brutal", "stressed", "sucks",
                 "hate", "worried", "lonely", "annoyed", "exhausted", "rough"),
    "positive": ("great", "good", "happy", "excited", "awesome", "love", "amazing",
                 "stoked", "nice", "fun", "got the job", "celebrat"),
}
EMPATHY_TOKENS = ("oof", "damn", "rough", "sorry", "that sucks", "ugh", "yikes",
                  "hang in there", "sending", "aw", "ouch")
CELEBRATE_TOKENS = ("congrats", "nice", "yes!", "amazing", "awesome", "let's go",
                    "so happy", "hyped", "love that", "great")
BACKCHANNEL = ("yeah", "oh", "mm", "haha", "lol", "right", "true", "same", "fair",
               "wow", "damn", "nice", "for real", "makes sense", "gotcha")


@dataclass
class HeuristicScorer:
    """Cheap, deterministic proxy for the eight rubric dimensions.

    Documented as a *proxy*: it encodes our stylistic priors so rejection sampling
    can run at scale without an API. Swap in LLMScorer or HumanScorer for the
    headline numbers reported in the paper.
    """

    target_words: tuple[int, int] = (3, 25)
    hard_max_words: int = 45

    def score(self, context: Sequence[dict[str, str]], candidate: str) -> dict[str, float]:
        text = candidate.strip()
        low = text.lower()
        w = words(text)
        n = len(w)
        user_turns = [m["content"] for m in context if m["role"] == "user"]
        last_user = user_turns[-1] if user_turns else ""
        prev_assistant = [m["content"] for m in context if m["role"] == "assistant"]

        if not text:
            return {d: 1.0 for d in DIMENSIONS}

        # brevity: peak inside the target band, decay outside it
        lo, hi = self.target_words
        if n < lo:
            brevity = 2.5 + n * 0.5
        elif n <= hi:
            brevity = 5.0
        else:
            brevity = max(1.0, 5.0 - (n - hi) * 0.15)

        # naturalness: grammatical, informal, not over-punctuated
        naturalness = 4.0
        if looks_ungrammatical(text):
            naturalness -= 2.5
        if re.search(r"\b(?:i'?m|it'?s|that'?s|don'?t|yeah|nah|lol|haha|kinda)\b", low):
            naturalness += 0.75
        if text[0].isupper() and text.endswith((".", "!", "?")) and n > 20:
            naturalness -= 0.25  # essay cadence
        naturalness = _clip(naturalness)

        # relevance: lexical hook into the last user turn, without parroting it
        overlap = jaccard(text, last_user)
        copy = context_copy_ratio(text, last_user)
        relevance = _clip(2.0 + 6.0 * min(overlap, 0.4))
        if copy > 0.85 and n >= 4:
            relevance = 1.0
        if any(b in low for b in BACKCHANNEL):
            relevance = _clip(relevance + 0.5)

        # emotional appropriateness: match the valence of the user's turn
        valence = _valence(last_user)
        emo = 3.0
        if valence == "negative":
            emo = 4.75 if any(t in low for t in EMPATHY_TOKENS) else 2.5
            if any(t in low for t in CELEBRATE_TOKENS):
                emo = 1.5
        elif valence == "positive":
            emo = 4.75 if any(t in low for t in CELEBRATE_TOKENS) else 3.0
            if any(t in low for t in EMPATHY_TOKENS):
                emo = 1.5
        emo = _clip(emo)

        # continuation: invites a next turn without interrogating
        q = text.count("?")
        continuation = 3.0
        if q == 1:
            continuation = 4.75
        elif q > 2:
            continuation = 2.0
        elif q == 0 and any(b in low for b in BACKCHANNEL):
            continuation = 3.75
        if low.rstrip(".!") in ("ok", "yes", "no", "sure", "..."):
            continuation = 1.5
        continuation = _clip(continuation)

        # non-repetition: within the candidate and against earlier replies
        rep = repeated_ngram_ratio(text, 3)
        non_rep = _clip(5.0 - 12.0 * rep)
        for p in prev_assistant[-4:]:
            if jaccard(text, p) > 0.6:
                non_rep = min(non_rep, 1.5)

        # unnecessary factual claims / hallucination risk
        facts = sum(m in low for m in CONFIDENT_FACT_MARKERS)
        digits = sum(c.isdigit() for c in text)
        no_facts = _clip(5.0 - 2.0 * facts - 0.15 * digits)
        if any(g in low for g in GRACEFUL_IGNORANCE) and "?" in last_user:
            no_facts = 5.0

        # assistant-style penalty
        ai_hits = sum(m in low for m in AI_STYLE_MARKERS)
        non_assistant = _clip(5.0 - 2.5 * ai_hits - (1.0 if n > self.hard_max_words else 0.0))
        if re.search(r"^\s*(?:1\.|-|\*)\s", text, re.MULTILINE):
            non_assistant = min(non_assistant, 1.5)

        return {
            "naturalness": naturalness,
            "brevity": brevity,
            "relevance": relevance,
            "emotional_appropriateness": emo,
            "continuation": continuation,
            "non_repetition": non_rep,
            "no_unnecessary_facts": no_facts,
            "non_assistant_style": non_assistant,
        }


def _clip(x: float) -> float:
    return round(max(1.0, min(5.0, x)), 3)


def _valence(text: str) -> str:
    low = text.lower()
    neg = sum(c in low for c in EMOTION_CUES["negative"])
    pos = sum(c in low for c in EMOTION_CUES["positive"])
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


# ---------------------------------------------------------------------------
# Pluggable external scorers
# ---------------------------------------------------------------------------
@dataclass
class CallbackScorer:
    """Wrap any callable, e.g. an API-backed LLM judge or a local model."""

    fn: Callable[[Sequence[dict[str, str]], str], dict[str, float]]

    def score(self, context, candidate):  # noqa: D102
        raw = self.fn(context, candidate)
        return {d: float(raw.get(d, 3.0)) for d in DIMENSIONS}


@dataclass
class PrecomputedScorer:
    """Read scores supplied by a human or offline judge, keyed by (id, index)."""

    table: dict[str, dict[str, float]]
    fallback: Scorer | None = None

    @classmethod
    def from_jsonl(cls, path: str | Path, fallback: Scorer | None = None) -> "PrecomputedScorer":
        table: dict[str, dict[str, float]] = {}
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    table[f"{obj['id']}#{obj.get('candidate', 0)}"] = obj.get("scores", {})
        return cls(table, fallback)

    def score_keyed(self, key: str, context, candidate) -> dict[str, float]:
        if key in self.table:
            return {d: float(self.table[key].get(d, 3.0)) for d in DIMENSIONS}
        if self.fallback:
            return self.fallback.score(context, candidate)
        return {d: 3.0 for d in DIMENSIONS}

    def score(self, context, candidate):  # noqa: D102
        return self.fallback.score(context, candidate) if self.fallback else {d: 3.0 for d in DIMENSIONS}


def llm_judge_packets(conversations: Sequence[Conversation]) -> list[dict[str, Any]]:
    """Emit JSON packets an external LLM (or human) can fill in and hand back."""
    from ..eval.judge import JUDGE_SYSTEM_PROMPT

    packets = []
    for conv in conversations:
        context = [m.to_dict() for m in conv.messages if m.role != "system"]
        if context and context[-1]["role"] == "assistant":
            context = context[:-1]
        for i, cand in enumerate(conv.candidates):
            packets.append(
                {
                    "id": conv.id,
                    "candidate": i,
                    "system_prompt": JUDGE_SYSTEM_PROMPT,
                    "context": context,
                    "response": cand.content,
                    "dimensions": list(DIMENSIONS),
                    "response_format": {
                        "id": conv.id,
                        "candidate": i,
                        "scores": {d: "<int 1-5>" for d in DIMENSIONS},
                    },
                }
            )
    return packets


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
@dataclass
class SelectionConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_score: float = 3.0          # reject the whole example below this
    hard_floors: dict[str, float] = field(
        default_factory=lambda: {"naturalness": 2.0, "relevance": 2.0, "non_assistant_style": 2.0}
    )
    keep_top_k: int = 1


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.get(d, 0.0) for d in DIMENSIONS) or 1.0
    return round(
        sum(scores.get(d, 3.0) * weights.get(d, 0.0) for d in DIMENSIONS) / total_w, 4
    )


def score_conversation(
    conv: Conversation, scorer: Scorer, cfg: SelectionConfig | None = None
) -> Conversation:
    cfg = cfg or SelectionConfig()
    context = [m.to_dict() for m in conv.messages]
    if context and context[-1]["role"] == "assistant":
        context = context[:-1]

    for i, cand in enumerate(conv.candidates):
        if isinstance(scorer, PrecomputedScorer):
            s = scorer.score_keyed(f"{conv.id}#{i}", context, cand.content)
        else:
            s = scorer.score(context, cand.content)
        cand.scores = s
        cand.score = weighted_score(s, cfg.weights)
    if conv.candidates:
        best = max(range(len(conv.candidates)), key=lambda i: conv.candidates[i].score or -1e9)
        conv.chosen = best
    return conv


def rejection_sample(
    conversations: Iterable[Conversation],
    scorer: Scorer | None = None,
    cfg: SelectionConfig | None = None,
) -> tuple[list[Conversation], dict[str, Any]]:
    """Score every candidate, keep the winner, drop examples no candidate rescues."""
    scorer = scorer or HeuristicScorer()
    cfg = cfg or SelectionConfig()
    kept: list[Conversation] = []
    stats = {"seen": 0, "kept": 0, "rejected_low_score": 0, "rejected_floor": 0,
             "no_candidates": 0, "mean_chosen_score": 0.0, "mean_candidates": 0.0}
    total_score = 0.0
    total_cands = 0

    for conv in conversations:
        stats["seen"] += 1
        if not conv.candidates:
            stats["no_candidates"] += 1
            kept.append(conv)  # already-final SFT data passes through untouched
            continue
        total_cands += len(conv.candidates)
        conv = score_conversation(conv, scorer, cfg)
        best = conv.candidates[conv.chosen or 0]
        if (best.score or 0) < cfg.min_score:
            stats["rejected_low_score"] += 1
            continue
        if any(best.scores.get(d, 5.0) < floor for d, floor in cfg.hard_floors.items()):
            stats["rejected_floor"] += 1
            continue
        total_score += best.score or 0.0
        stats["kept"] += 1
        kept.append(conv)

    scored = max(stats["kept"], 1)
    stats["mean_chosen_score"] = round(total_score / scored, 4)
    stats["mean_candidates"] = round(total_cands / max(stats["seen"], 1), 2)
    return kept, stats


def write_selected(path: str | Path, conversations: Sequence[Conversation]) -> int:
    """Materialise winners as plain SFT data (final assistant turn = chosen)."""
    from ..data.dataset import _apply_chosen

    out = [c for c in (_apply_chosen(c) for c in conversations) if c is not None]
    return write_jsonl(path, out)
