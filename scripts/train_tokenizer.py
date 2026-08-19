#!/usr/bin/env python3
"""Train the conversational byte-level BPE tokenizer and report its efficiency.

python scripts/train_tokenizer.py --data data/processed/train.jsonl --vocab-size 4096
python scripts/train_tokenizer.py --data data/processed/train.jsonl --vocab-size 4096 6144
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.data.schema import load_conversations
from smalltalk.tokenizer import SmallTalkTokenizer, train_tokenizer

PROBES = [
    "hey", "hey, what's up?", "honestly kinda tired today", "yeah work was brutal",
    "mostly meetings", "That's the worst kind of tired lol.",
    "damn. what happened?", "honestly no idea lol. What's it used for?",
    "i'm gonna head out, night :)", "ugh don't even get me started",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/train.jsonl")
    ap.add_argument("--vocab-size", type=int, nargs="+", default=[4096])
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--no-curated-atoms", action="store_true",
                    help="ablation: skip the forced conversational atoms")
    args = ap.parse_args()

    convs = load_conversations(args.data)
    texts = [m.content for c in convs for m in c.messages if m.content]
    print(f"[data] {len(convs):,} conversations / {len(texts):,} utterances")
    if not texts:
        print("no text found; run prepare_data.py first")
        return 1

    report = {}
    for vs in args.vocab_size:
        out_dir = Path(args.out) / f"tokenizer-{vs}"
        tok = train_tokenizer(
            texts, vocab_size=vs, out_dir=out_dir,
            min_frequency=args.min_frequency,
            always_keep=[] if args.no_curated_atoms else None,
        )
        # round-trip + compression check
        total_tokens = sum(len(tok.encode(t)) for t in texts)
        total_chars = sum(len(t) for t in texts)
        failures = [t for t in PROBES if tok.decode(tok.encode(t)).strip() != t.strip()]
        stats = {
            "vocab_size": tok.vocab_size,
            "requested": vs,
            "chars_per_token": round(total_chars / max(total_tokens, 1), 3),
            "tokens_per_utterance": round(total_tokens / max(len(texts), 1), 2),
            "embedding_params_at_h256": tok.vocab_size * 256,
            "embedding_params_at_h384": tok.vocab_size * 384,
            "roundtrip_failures": failures,
            "path": str(out_dir),
        }
        report[str(vs)] = stats
        print(f"\n[tokenizer-{vs}] {json.dumps(stats, indent=2)}")
        for p in PROBES[:5]:
            ids = tok.encode(p)
            print(f"   {len(ids):>3} tok | {p!r} -> {[tok.id_to_token(i) for i in ids]}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "tokenizer_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
