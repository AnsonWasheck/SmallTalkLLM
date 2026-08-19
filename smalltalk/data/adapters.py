"""Import adapters: external corpora -> canonical Conversation records.

Each adapter is `(source_spec) -> Iterator[Conversation]`. All of them are offline
friendly: if `datasets` is installed we use it, otherwise we read the raw files
the original corpora ship as. Missing data raises a clear, actionable error
rather than silently yielding nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

from .schema import Candidate, Conversation, Turn

TRANSCRIPT_RE = re.compile(
    r"^\s*(user|assistant|human|ai|a|b|speaker\s*1|speaker\s*2|me|them|you|bot)\s*[:>-]\s*",
    re.IGNORECASE,
)
ROLE_ALIASES = {
    "user": "user", "human": "user", "a": "user", "speaker 1": "user",
    "speaker1": "user", "me": "user", "you": "user",
    "assistant": "assistant", "ai": "assistant", "bot": "assistant",
    "b": "assistant", "speaker 2": "assistant", "speaker2": "assistant",
    "them": "assistant", "system": "system",
}


def _alternating(utterances: Sequence[str], first_role: str = "user") -> list[Turn]:
    roles = ("user", "assistant") if first_role == "user" else ("assistant", "user")
    turns = []
    for i, u in enumerate(utterances):
        u = u.strip()
        if u:
            turns.append(Turn(roles[i % 2], u))
    return turns


# ---------------------------------------------------------------------------
# DailyDialog
# ---------------------------------------------------------------------------
def load_dailydialog(
    path: str | Path | None = None, split: str = "train", limit: int | None = None
) -> Iterator[Conversation]:
    """DailyDialog: 13k multi-turn everyday English dialogues, 2 speakers.

    Accepts either the HF dataset (`li2017dailydialog/daily_dialog`) or the raw
    release layout: `<path>/dialogues_text.txt` with `__eou__`-separated turns.
    """
    n = 0
    if path is not None:
        p = Path(path)
        files = (
            [p]
            if p.is_file()
            else sorted(list(p.glob("dialogues_*.txt")) + list(p.glob("*/dialogues_*.txt")))
        )
        if not files:
            raise FileNotFoundError(f"no dialogues_*.txt under {p}")
        for f in files:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines()):
                utts = [u.strip() for u in line.split("__eou__") if u.strip()]
                if len(utts) < 2:
                    continue
                yield Conversation(
                    id=f"dd-{f.stem}-{i:06d}",
                    messages=_alternating(utts),
                    source="dailydialog",
                    meta={"split": split},
                )
                n += 1
                if limit and n >= limit:
                    return
        return

    ds = _hf_load("li2017dailydialog/daily_dialog", split)
    for i, row in enumerate(ds):
        utts = [u.strip() for u in row["dialog"] if u.strip()]
        if len(utts) < 2:
            continue
        yield Conversation(
            id=f"dd-{split}-{i:06d}",
            messages=_alternating(utts),
            source="dailydialog",
            meta={"split": split, "topic": row.get("topic")},
        )
        n += 1
        if limit and n >= limit:
            return


# ---------------------------------------------------------------------------
# EmpatheticDialogues
# ---------------------------------------------------------------------------
def load_empathetic_dialogues(
    path: str | Path | None = None, split: str = "train", limit: int | None = None
) -> Iterator[Conversation]:
    """EmpatheticDialogues: ~25k emotionally grounded conversations.

    Accepts the HF dataset (`facebook/empathetic_dialogues`) or the raw CSVs
    (`train.csv` / `valid.csv` / `test.csv`) which are one *utterance* per row and
    must be regrouped by `conv_id`.
    """
    rows: list[dict[str, Any]]
    if path is not None:
        import csv

        p = Path(path)
        f = p if p.is_file() else p / f"{split}.csv"
        if not f.exists():
            raise FileNotFoundError(f"expected EmpatheticDialogues CSV at {f}")
        with f.open(encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = [dict(r) for r in _hf_load("facebook/empathetic_dialogues", split)]

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("conv_id")), []).append(r)

    n = 0
    for conv_id, group in groups.items():
        group.sort(key=lambda r: int(float(r.get("utterance_idx") or 0)))
        utts = [
            str(r.get("utterance", "")).replace("_comma_", ",").strip() for r in group
        ]
        utts = [u for u in utts if u]
        if len(utts) < 2:
            continue
        emotion = str(group[0].get("context") or "")
        prompt = str(group[0].get("prompt") or "").replace("_comma_", ",")
        yield Conversation(
            id=f"ed-{split}-{conv_id}",
            messages=_alternating(utts),
            source="empathetic_dialogues",
            meta={"split": split, "emotion": emotion, "situation": prompt},
        )
        n += 1
        if limit and n >= limit:
            return


# ---------------------------------------------------------------------------
# Generic JSONL / synthetic teacher output
# ---------------------------------------------------------------------------
def load_jsonl_conversations(
    path: str | Path, source: str | None = None, limit: int | None = None
) -> Iterator[Conversation]:
    """Tolerant JSONL reader for hand-written, third-party or teacher data.

    Understands: canonical `messages`, `conversation`/`text` transcripts,
    `dialog`/`turns`/`utterances` lists, and `candidates` (stage-3 records).
    """
    p = Path(path)
    src = source or p.stem
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = normalize_record(obj, conv_id=obj.get("id") or f"{src}-{i:06d}", source=src)
            if conv is not None:
                yield conv
                if limit and i + 1 >= limit:
                    return


def normalize_record(
    obj: dict[str, Any], conv_id: str, source: str = "unknown"
) -> Conversation | None:
    messages: list[Turn] = []

    if isinstance(obj.get("messages"), list):
        for m in obj["messages"]:
            role = ROLE_ALIASES.get(str(m.get("role", "")).lower().strip())
            if role and str(m.get("content", "")).strip():
                messages.append(Turn(role, str(m["content"])))
    elif isinstance(obj.get("conversation"), str) or isinstance(obj.get("text"), str):
        messages = parse_transcript(obj.get("conversation") or obj["text"])
    else:
        for key in ("dialog", "turns", "utterances", "dialogue"):
            seq = obj.get(key)
            if isinstance(seq, list) and seq:
                if isinstance(seq[0], str):
                    messages = _alternating(seq)
                else:
                    return normalize_record(
                        {"messages": seq}, conv_id=conv_id, source=source
                    )
                break

    if len(messages) < 2:
        return None

    raw_cands = obj.get("candidates") or []
    candidates: list[Candidate] = []
    for c in raw_cands:
        if isinstance(c, str):
            candidates.append(Candidate(content=c.strip(), source=obj.get("teacher")))
        elif isinstance(c, dict):
            content = str(c.get("content") or c.get("response") or c.get("text") or "").strip()
            if content:
                candidates.append(
                    Candidate(
                        content=content,
                        scores=c.get("scores", {}) or {},
                        score=c.get("score"),
                        source=c.get("source") or obj.get("teacher"),
                    )
                )

    meta = {k: v for k, v in obj.items() if k in ("topic", "emotion", "scenario", "style", "split")}
    return Conversation(
        id=str(conv_id),
        messages=messages,
        source=obj.get("source", source),
        meta=meta,
        candidates=candidates,
        chosen=obj.get("chosen"),
    )


def parse_transcript(text: str) -> list[Turn]:
    """Parse a `User: ...\\nAssistant: ...` style transcript into turns."""
    turns: list[Turn] = []
    current_role: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if current_role and buf:
            content = " ".join(x.strip() for x in buf).strip()
            if content:
                turns.append(Turn(current_role, content))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = TRANSCRIPT_RE.match(line)
        if m:
            flush()
            label = re.sub(r"\s+", " ", m.group(1).strip().lower())
            current_role = ROLE_ALIASES.get(label, "user")
            buf = [line[m.end() :]]
        elif current_role:
            buf.append(line)
    flush()
    return turns


# ---------------------------------------------------------------------------
ADAPTERS = {
    "dailydialog": load_dailydialog,
    "empathetic_dialogues": load_empathetic_dialogues,
    "jsonl": load_jsonl_conversations,
    "synthetic": load_jsonl_conversations,
}


def _hf_load(name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"loading {name} without a local --path requires `pip install datasets`, "
            "or point the adapter at the raw corpus files"
        ) from exc
    return load_dataset(name, split=split, trust_remote_code=False)
