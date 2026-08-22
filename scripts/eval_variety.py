#!/usr/bin/env python3
"""Score a checkpoint on VarietyBench (deterministic, greedy).

    python scripts/eval_variety.py --checkpoint <ckpt> --tag <name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.core import varietybench
from smalltalk.infer.generate import GenerationConfig, load_engine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer", default="artifacts/state/tokenizer-4096")
    ap.add_argument("--tag")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frozen", default="benchmarks/variety_frozen.json")
    ap.add_argument("--out", default="reports/variety")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    if args.freeze:
        print(f"froze VarietyBench checksum={varietybench.freeze(args.frozen)}")
        return 0
    varietybench.verify_frozen(args.frozen)

    gen = GenerationConfig(temperature=0.0, top_p=1.0, top_k=0, greedy=True,
                           repetition_penalty=1.0, max_new_tokens=20, seed=0)
    e = load_engine(args.checkpoint, args.tokenizer, device=args.device, gen=gen)

    transcripts = {}
    for name, turns in varietybench.CONVERSATIONS:
        e.reset()
        transcripts[name] = [e.reply(t) for t in turns]

    res = varietybench.score(transcripts)
    res.update({"tag": args.tag, "checkpoint": args.checkpoint,
                "transcripts": transcripts})
    print(f"\nVarietyBench {res['checksum']}")
    print(f"  repeat_rate    {res['repeat_rate']:6.1%}   <- lower is better")
    print(f"  distinct_ratio {res['distinct_ratio']:6.1%}   <- higher is better")
    print(f"  top1_share     {res['top1_share']:6.1%}")
    out = Path(args.out) / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
