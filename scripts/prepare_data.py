#!/usr/bin/env python3
"""Ingest, clean, deduplicate and split the conversational corpus.

Examples
--------
# runnable-out-of-the-box: offline synthetic corpus only
python scripts/prepare_data.py --offline 4000 --out data/processed

# real study corpus
python scripts/prepare_data.py \
    --dailydialog data/raw/dailydialog \
    --empathetic data/raw/empatheticdialogues \
    --jsonl data/raw/teacher_conversations.jsonl \
    --out data/processed

# ablation control for research question 5 (generic-LM style filtering)
python scripts/prepare_data.py --offline 4000 --permissive --out data/processed_generic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.data import adapters
from smalltalk.data.clean import (
    FilterConfig,
    clean_conversations,
    corpus_stats,
    split_train_val,
)
from smalltalk.data.schema import Conversation, write_jsonl
from smalltalk.data.synthetic import OfflineConfig, generate_offline_corpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dailydialog", nargs="?", const="", help="path to DailyDialog (omit value to use HF)")
    ap.add_argument("--empathetic", nargs="?", const="", help="path to EmpatheticDialogues (omit value to use HF)")
    ap.add_argument("--jsonl", action="append", default=[], help="extra conversational JSONL (repeatable)")
    ap.add_argument("--offline", type=int, default=0, help="N template-generated conversations")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit-per-source", type=int, default=None)
    ap.add_argument("--permissive", action="store_true", help="ablation: generic-LM filtering")
    ap.add_argument("--downweight", action="store_true", help="keep off-style data with weight<1")
    ap.add_argument("--max-assistant-words", type=int, default=None)
    args = ap.parse_args()

    convs: list[Conversation] = []
    sources: dict[str, int] = {}

    def ingest(label: str, it) -> None:
        before = len(convs)
        convs.extend(it)
        sources[label] = len(convs) - before
        print(f"[ingest] {label}: {sources[label]:,}")

    if args.dailydialog is not None:
        ingest("dailydialog", adapters.load_dailydialog(
            args.dailydialog or None, limit=args.limit_per_source))
    if args.empathetic is not None:
        ingest("empathetic_dialogues", adapters.load_empathetic_dialogues(
            args.empathetic or None, limit=args.limit_per_source))
    for p in args.jsonl:
        ingest(Path(p).stem, adapters.load_jsonl_conversations(p, limit=args.limit_per_source))
    if args.offline:
        ingest("offline_template", generate_offline_corpus(
            OfflineConfig(num_conversations=args.offline, seed=args.seed)))

    if not convs:
        ap.error("no data sources given; try --offline 4000 to get started")

    cfg = FilterConfig.permissive() if args.permissive else FilterConfig()
    cfg.downweight_instead_of_drop = args.downweight
    if args.max_assistant_words:
        cfg.max_assistant_words = args.max_assistant_words

    kept, stats = clean_conversations(convs, cfg)
    print("[clean]", stats.report().replace("\n", "\n[clean] "))
    if not kept:
        print("[clean] everything was filtered out; loosen the filters")
        return 1

    train, val = split_train_val(kept, val_ratio=args.val_ratio, seed=args.seed)
    out = Path(args.out)
    n_train = write_jsonl(out / "train.jsonl", train)
    n_val = write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "all.jsonl", kept)

    report = {
        "ingested": sources,
        "filter_config": cfg.__dict__,
        "dropped": dict(stats.dropped),
        "train": n_train,
        "val": n_val,
        "stats_train": corpus_stats(train),
        "stats_val": corpus_stats(val),
    }
    (out / "corpus_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"[write] {out}/train.jsonl ({n_train:,})  {out}/val.jsonl ({n_val:,})")
    print(json.dumps(report["stats_train"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
