#!/usr/bin/env python3
"""Score a checkpoint on ElaborationBench (deterministic, greedy).

    python scripts/eval_elaboration.py --checkpoint <ckpt> --tag <name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.core import elaboration as eb
from smalltalk.infer.generate import GenerationConfig, load_engine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer", default="artifacts/state/tokenizer-4096")
    ap.add_argument("--tag")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frozen", default="benchmarks/elaboration_frozen.json")
    ap.add_argument("--out", default="reports/elaboration")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.freeze:
        print(f"froze ElaborationBench checksum={eb.freeze(args.frozen)}")
        return 0
    eb.verify_frozen(args.frozen)

    gen = GenerationConfig(temperature=0.0, top_p=1.0, top_k=0, greedy=True,
                           repetition_penalty=1.0, max_new_tokens=20, seed=0)
    e = load_engine(args.checkpoint, args.tokenizer, device=args.device, gen=gen)

    replies = {}
    for s in eb.SCENARIOS:
        e.reset()
        replies[s.id] = e.reply(s.prompt)

    res = eb.score(replies)
    res.update({"tag": args.tag, "checkpoint": args.checkpoint, "replies": replies})
    print(f"\nElaborationBench {res['checksum']}")
    print(f"  elaboration_rate       {res['elaboration_rate']:6.1%}  (reuses the user's referent)")
    print(f"     known                {res['elaboration_known']:6.1%}")
    print(f"     unknown              {res['elaboration_unknown']:6.1%}  <- generalisation")
    print(f"  false_elaboration_rate {res['false_elaboration_rate']:6.1%}  (invents specificity)")
    print(f"  hedge_rate             {res['hedge_rate']:6.1%}  (universal follow-up)")
    print(f"  COMPOSITE              {res['composite']:6.3f}")
    print(f"  known   {res['known_breakdown']}")
    print(f"  unknown {res['unknown_breakdown']}")
    if args.show:
        for s in eb.SCENARIOS:
            print(f"    [{'K' if s.known else 'U'}] {s.prompt[:30]:32s} -> "
                  f"{replies[s.id][:34]!r:38s} {eb.classify(replies[s.id], s)}")
    out = Path(args.out) / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
