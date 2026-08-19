#!/usr/bin/env python3
"""Stage 3: rejection-selected SFT via teacher best-of-N selection.

Three sub-commands:

  prompts   emit teacher generation requests (send these to a larger model)
              python scripts/distill.py prompts --n 5000 --out data/raw/teacher_prompts.jsonl
  select    score candidates and materialise the winners as SFT data
              python scripts/distill.py select --in data/raw/teacher_candidates.jsonl \
                  --out data/processed/distill_train.jsonl
  judge     emit per-candidate scoring packets for an external LLM or human
              python scripts/distill.py judge --in data/raw/teacher_candidates.jsonl \
                  --out artifacts/judge/candidates.jsonl

Then train:  python scripts/sft.py --config configs/train/distill_4m.yaml

This is deliberately not called logit distillation: it optimizes a chosen target
string. Online CE+KL transfer from a same-tokenizer teacher lives in
``scripts/distill_student.py``.
No teacher is required at inference time -- the deployed artifact is the micro model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.data.adapters import load_jsonl_conversations
from smalltalk.data.clean import FilterConfig, clean_conversations
from smalltalk.data.synthetic import build_teacher_prompts, write_teacher_prompts
from smalltalk.distill.scoring import (
    DEFAULT_WEIGHTS,
    HeuristicScorer,
    PrecomputedScorer,
    SelectionConfig,
    llm_judge_packets,
    rejection_sample,
    write_selected,
)


def cmd_prompts(args) -> int:
    reqs = build_teacher_prompts(n=args.n, seed=args.seed)
    p = write_teacher_prompts(args.out, reqs)
    print(f"[prompts] {len(reqs):,} teacher requests -> {p}")
    print("Send each record's system_prompt + user_prompt to the teacher; save the "
          "returned JSON lines and feed them back via `distill select` or prepare_data.py --jsonl")
    return 0


def cmd_select(args) -> int:
    convs = list(load_jsonl_conversations(args.inp, source="teacher"))
    print(f"[load] {len(convs):,} records ({sum(bool(c.candidates) for c in convs):,} with candidates)")

    if not args.no_clean:
        convs, stats = clean_conversations(convs, FilterConfig())
        print("[clean]", stats.report().replace("\n", "\n[clean] "))

    scorer = (
        PrecomputedScorer.from_jsonl(args.scores, fallback=HeuristicScorer())
        if args.scores else HeuristicScorer()
    )
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.loads(args.weights))
    cfg = SelectionConfig(weights=weights, min_score=args.min_score)

    kept, stats = rejection_sample(convs, scorer, cfg)
    n = write_selected(args.out, kept)
    print(f"[select] {json.dumps(stats, indent=2)}")
    print(f"[write] {n:,} distillation examples -> {args.out}")
    report = Path(args.out).with_suffix(".selection_report.json")
    report.write_text(json.dumps({"stats": stats, "config": {
        "weights": weights, "min_score": args.min_score,
        "scorer": type(scorer).__name__}}, indent=2))
    return 0


def cmd_judge(args) -> int:
    convs = list(load_jsonl_conversations(args.inp, source="teacher"))
    packets = llm_judge_packets(convs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in packets:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[judge] {len(packets):,} scoring packets -> {out}")
    print("Return one line per packet: {\"id\":..., \"candidate\":..., \"scores\":{...}} "
          "then pass it to `distill select --scores`")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prompts")
    p1.add_argument("--n", type=int, default=5000)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--out", default="data/raw/teacher_prompts.jsonl")
    p1.set_defaults(func=cmd_prompts)

    p2 = sub.add_parser("select")
    p2.add_argument("--in", dest="inp", required=True)
    p2.add_argument("--out", default="data/processed/distill_train.jsonl")
    p2.add_argument("--scores", help="JSONL of externally supplied per-candidate scores")
    p2.add_argument("--weights", help="JSON dict overriding dimension weights")
    p2.add_argument("--min-score", type=float, default=3.0)
    p2.add_argument("--no-clean", action="store_true")
    p2.set_defaults(func=cmd_select)

    p3 = sub.add_parser("judge")
    p3.add_argument("--in", dest="inp", required=True)
    p3.add_argument("--out", default="artifacts/judge/candidates.jsonl")
    p3.set_defaults(func=cmd_judge)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
