#!/usr/bin/env python3
"""Evaluate a checkpoint on Core-Bench at temperature 0.

  python scripts/eval_core.py --checkpoint artifacts/loop/best --tag v0.2-core

Deterministic by construction: greedy decoding, fixed scenarios, no sampling.
Two runs of the same checkpoint must produce byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.core.bench_core import (TIER_TARGETS, build_scenarios, checksum,
                                       freeze, score, verify_frozen)
from smalltalk.infer.generate import GenerationConfig, load_engine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer-4096")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frozen", default="artifacts/core_bench_frozen.json")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--out", default="reports/core")
    ap.add_argument("--freeze", action="store_true", help="write the frozen manifest")
    args = ap.parse_args()

    if args.freeze:
        print(f"froze Core-Bench checksum={freeze(args.frozen)}")
        return 0
    verify_frozen(args.frozen)

    # The measurement setting: pure argmax. Sampling knobs are all neutralised
    # so a failure is a failure of the distribution, not of the draw.
    gen = GenerationConfig(temperature=0.0, top_p=1.0, top_k=0, greedy=True,
                           repetition_penalty=1.0, presence_penalty=0.0,
                           no_repeat_ngram_size=0, seed=0,
                           max_new_tokens=args.max_new_tokens)
    engine = load_engine(args.checkpoint, args.tokenizer, gen=gen)

    scenarios = build_scenarios()
    per_intent: dict[str, list[int]] = defaultdict(list)
    per_tier: dict[int, list[int]] = defaultdict(list)
    records = []
    t0 = time.time()
    for s in scenarios:
        engine.reset()
        reply = engine.reply(s.prompt)
        ok = bool(score(s.intent, reply))
        per_intent[s.intent].append(int(ok))
        per_tier[s.tier].append(int(ok))
        records.append({"intent": s.intent, "tier": s.tier, "prompt": s.prompt,
                        "reply": reply, "correct": ok})

    overall = sum(v for r in per_intent.values() for v in r) / len(scenarios)
    print(f"\nCore-Bench {checksum()}  n={len(scenarios)}  "
          f"{time.time() - t0:.0f}s\n")
    print(f"{'intent':26s} {'tier':>4s} {'n':>4s} {'pass':>7s}  target  status")
    for name, vals in sorted(per_intent.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        it_tier = next(r["tier"] for r in records if r["intent"] == name)
        p = sum(vals) / len(vals)
        tgt = TIER_TARGETS[it_tier]
        print(f"{name:26s} {it_tier:>4d} {len(vals):>4d} {p:>6.1%}  {tgt:>5.0%}   "
              f"{'ok' if p >= tgt else 'FAIL'}")
    print()
    for tier in sorted(per_tier):
        vals = per_tier[tier]
        print(f"tier {tier}: {sum(vals) / len(vals):.1%}  (target {TIER_TARGETS[tier]:.0%})")
    print(f"\nOVERALL {overall:.1%}")

    out = Path(args.out) / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": args.tag, "checkpoint": args.checkpoint, "checksum": checksum(),
        "overall": overall,
        "per_intent": {k: sum(v) / len(v) for k, v in per_intent.items()},
        "per_tier": {str(k): sum(v) / len(v) for k, v in per_tier.items()},
        "generations": records,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
