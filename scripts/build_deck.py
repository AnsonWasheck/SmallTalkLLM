#!/usr/bin/env python3
"""Build one training deck with tunable generator knobs.

Used by scripts/loop.sh so each unattended round can vary the corpus instead of
retraining on identical data (which was the flaw in the first autoloop).

python scripts/build_deck.py --combi 10000 --seed 13 --bored 0.20 --out data/processed/sft_train.jsonl

Fails closed on benchmark leakage: exit 1 rather than train on contaminated data.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.data.adapters import load_jsonl_conversations
from smalltalk.data.clean import FilterConfig, clean_conversations, corpus_stats
from smalltalk.data.combi_gen import GenConfig, generate
from smalltalk.data.schema import write_jsonl
from smalltalk.eval.leakage import filter_leaked

ROOT = Path(__file__).resolve().parents[1]
SOCIAL_GOLD = ROOT / "data/raw/quinn/quinn7m_social_gold/social_gold_train.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combi", type=int, default=10000, help="n combinatorial dialogues")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/sft_train.jsonl")
    ap.add_argument("--stats-out", default="artifacts/loop/deck_stats.json")
    # valence mix
    ap.add_argument("--pos", type=float, default=0.24)
    ap.add_argument("--mild", type=float, default=0.26)
    ap.add_argument("--heavy", type=float, default=0.11)
    ap.add_argument("--neu", type=float, default=0.19)
    ap.add_argument("--bored", type=float, default=0.20)
    # structural features
    ap.add_argument("--p-memory", type=float, default=0.22)
    ap.add_argument("--p-topic-switch", type=float, default=0.18)
    ap.add_argument("--p-unknown", type=float, default=0.10)
    ap.add_argument("--max-pairs", type=int, default=9)
    ap.add_argument("--no-social-gold", action="store_true")
    args = ap.parse_args()

    cfg = GenConfig(
        n=args.combi, seed=args.seed, max_pairs=args.max_pairs,
        p_memory_probe=args.p_memory, p_topic_switch=args.p_topic_switch,
        p_unknown_fact=args.p_unknown,
        valence_mix={"pos": args.pos, "mild": args.mild, "heavy": args.heavy,
                     "neu": args.neu, "bored": args.bored},
    )
    convs = list(generate(cfg))
    sources = {"combi_gen": len(convs)}

    if not args.no_social_gold and SOCIAL_GOLD.exists():
        c = list(load_jsonl_conversations(SOCIAL_GOLD, source="social_gold"))
        convs += c
        sources["social_gold"] = len(c)
    for f in sorted(glob.glob(str(ROOT / "data/generated/*.jsonl"))):
        c = list(load_jsonl_conversations(f, source="authored"))
        convs += c
        sources[f"authored:{Path(f).stem}"] = len(c)

    kept, st = clean_conversations(convs, FilterConfig())
    kept, leak = filter_leaked(kept)
    print(leak.summary().split("\n")[0])
    if leak.flagged:
        print(f"[deck] dropped {len(leak.flagged)} leaked conversations")
    if not kept:
        print("[deck] FATAL: nothing survived cleaning", file=sys.stderr)
        return 1

    n = write_jsonl(args.out, kept)
    stats = {
        "generator": {k: v for k, v in vars(args).items()},
        "sources": sources,
        "dropped": dict(st.dropped),
        "leaked_dropped": len(leak.flagged),
        "written": n,
        "corpus": corpus_stats(kept),
    }
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2, default=str))
    print(f"[deck] wrote {n:,} -> {args.out}")
    print(json.dumps(stats["corpus"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
