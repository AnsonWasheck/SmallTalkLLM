"""LLM-judge and human-evaluation exchange formats.

We never call an external model from this module. It only *emits* JSON packets and
*ingests* returned scores, so the judge can be a human, a hosted LLM, or a local
model without touching evaluation code. The deployed artifact stays the micro model.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

CRITERIA: dict[str, str] = {
    "naturalness": "Does this sound like a real person texting, not a chatbot?",
    "context_relevance": "Does each reply respond to what was actually just said?",
    "emotional_appropriateness": "Does it react to the user's feelings appropriately?",
    "conversational_continuity": "Does the conversation flow and move forward?",
    "response_diversity": "Are replies varied rather than repetitive/templated?",
    "human_likeness": "Overall, would you believe a human wrote these?",
    "memory_of_recent_turns": "Does it remember what was said a few turns ago?",
    "absence_of_ai_verbosity": "Is it free of formal assistant-style padding? (5 = none)",
}

RUBRIC = {
    "scale": "1-5 integers",
    "1": "broken / nonsensical / robotic",
    "2": "clearly off",
    "3": "acceptable but noticeably artificial",
    "4": "good, mostly human",
    "5": "indistinguishable from a friendly human texter",
}

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating a tiny chat model whose ONLY goal is natural casual small talk. "
    "It is NOT an assistant: it should not be knowledgeable, helpful, thorough, or verbose. "
    "Short, warm, human-sounding replies (3-25 words) are ideal. "
    "Penalise assistant-style verbosity, lists, disclaimers and factual lecturing. "
    "Do not reward correct facts. Do not penalise saying it doesn't know something. "
    "Return ONLY the JSON object requested."
)


@dataclass
class JudgeRequest:
    id: str
    model: str
    scenario_id: str
    category: str
    transcript: list[dict[str, str]]
    criteria: dict[str, str] = field(default_factory=lambda: dict(CRITERIA))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "scenario_id": self.scenario_id,
            "category": self.category,
            "system_prompt": JUDGE_SYSTEM_PROMPT,
            "rubric": RUBRIC,
            "criteria": self.criteria,
            "transcript": self.transcript,
            "response_format": {
                "scores": {k: "<int 1-5>" for k in self.criteria},
                "worst_turn": "<int index of weakest assistant turn, 1-based>",
                "comment": "<one short sentence>",
            },
        }


def build_judge_requests(
    transcripts: Iterable[dict[str, Any]], model_name: str
) -> list[JudgeRequest]:
    """`transcripts`: [{"scenario_id","category","messages"}]"""
    reqs = []
    for t in transcripts:
        raw = f"{model_name}|{t['scenario_id']}"
        reqs.append(
            JudgeRequest(
                id=hashlib.blake2b(raw.encode(), digest_size=8).hexdigest(),
                model=model_name,
                scenario_id=t["scenario_id"],
                category=t.get("category", "unknown"),
                transcript=[m for m in t["messages"] if m["role"] != "system"],
            )
        )
    return reqs


def write_judge_file(path: str | Path, requests: Sequence[JudgeRequest]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return p


def read_judge_scores(path: str | Path) -> dict[str, dict[str, float]]:
    """Ingest judge output: one JSON object per line with `id` and `scores`."""
    out: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            scores = obj.get("scores") or {}
            out[str(obj["id"])] = {
                k: float(v) for k, v in scores.items() if isinstance(v, (int, float))
            }
    return out


def aggregate_judge_scores(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    agg: dict[str, float] = {}
    for crit in CRITERIA:
        vals = [s[crit] for s in scores.values() if crit in s]
        if vals:
            agg[crit] = round(sum(vals) / len(vals), 3)
    if agg:
        agg["mean_all_criteria"] = round(sum(agg.values()) / len(agg), 3)
    return agg


# ---------------------------------------------------------------------------
# Pairwise blind evaluation
# ---------------------------------------------------------------------------
@dataclass
class PairwiseItem:
    id: str
    context: list[dict[str, str]]
    option_a: str
    option_b: str
    _mapping: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": "Which reply sounds more like a real human friend?",
            "context": self.context,
            "options": {"A": self.option_a, "B": self.option_b},
            "answer_format": {"id": self.id, "choice": "A|B|tie"},
        }


def build_pairwise(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    left_name: str,
    right_name: str,
    seed: int = 7,
) -> tuple[list[PairwiseItem], dict[str, dict[str, str]]]:
    """Blind A/B items. Returns (items, keyfile). Side assignment is randomised
    and the key is kept separate so the evaluator cannot see model identity."""
    rng = random.Random(seed)
    by_id_right = {r["scenario_id"]: r for r in right}
    items: list[PairwiseItem] = []
    key: dict[str, dict[str, str]] = {}

    for l in left:
        r = by_id_right.get(l["scenario_id"])
        if r is None:
            continue
        l_turns = [m for m in l["messages"] if m["role"] == "assistant"]
        r_turns = [m for m in r["messages"] if m["role"] == "assistant"]
        context = [m for m in l["messages"] if m["role"] != "system"]
        for i, (lt, rt) in enumerate(zip(l_turns, r_turns)):
            if lt["content"].strip() == rt["content"].strip():
                continue
            item_id = hashlib.blake2b(
                f"{l['scenario_id']}|{i}|{left_name}|{right_name}".encode(), digest_size=8
            ).hexdigest()
            flip = rng.random() < 0.5
            a, b = (rt, lt) if flip else (lt, rt)
            mapping = {
                "A": right_name if flip else left_name,
                "B": left_name if flip else right_name,
            }
            prefix = _context_prefix(context, i)
            items.append(
                PairwiseItem(item_id, prefix, a["content"], b["content"], mapping)
            )
            key[item_id] = mapping
    return items, key


def _context_prefix(messages: Sequence[dict[str, str]], assistant_idx: int) -> list[dict[str, str]]:
    """Dialogue up to (not including) the nth assistant reply."""
    seen = 0
    out: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "assistant":
            if seen == assistant_idx:
                break
            seen += 1
        out.append(m)
    return out


def write_pairwise(
    path: str | Path, items: Sequence[PairwiseItem], key: dict[str, dict[str, str]]
) -> tuple[Path, Path]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
    keyfile = p.with_suffix(".key.json")
    keyfile.write_text(json.dumps(key, indent=2))
    return p, keyfile


def score_pairwise(votes_path: str | Path, key_path: str | Path) -> dict[str, Any]:
    key = json.loads(Path(key_path).read_text())
    wins: dict[str, int] = {}
    ties = 0
    total = 0
    with Path(votes_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            v = json.loads(line)
            mapping = key.get(str(v["id"]))
            if not mapping:
                continue
            total += 1
            choice = str(v.get("choice", "")).upper()
            if choice == "TIE":
                ties += 1
            elif choice in mapping:
                wins[mapping[choice]] = wins.get(mapping[choice], 0) + 1
    return {
        "comparisons": total,
        "ties": ties,
        "wins": wins,
        "win_rate": {k: round(v / max(total, 1), 4) for k, v in wins.items()},
    }
