#!/usr/bin/env python3
"""Upper bound on what ANY reranker could achieve.

For each scenario, generate the model's top-k deterministic candidates and ask:
is a correct reply present anywhere in that set? This separates two very
different failures:

  * the right answer is in the candidate set but the harness picks a wrong one
    -> selection is the bottleneck, and a better policy layer will pay
  * the right answer is not in the set at all
    -> realisation is the bottleneck, and no harness can fix it

Without this number, an oracle-policy result is uninterpretable: a small gain
could mean policy does not matter, or that policy matters but there was nothing
better to choose.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from smalltalk.core import bench_core
from smalltalk.harness import Harness, MODES
from smalltalk.model import SmallTalkModel
from smalltalk.tokenizer import SmallTalkTokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/state/tokenizer-4096")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 8])
    args = ap.parse_args()

    model = SmallTalkModel.from_pretrained(args.checkpoint, device=args.device).eval()
    tok = SmallTalkTokenizer.load(args.tokenizer)
    h = Harness(model=model, tokenizer=tok, cfg=MODES["A_RAW"])

    scenarios = bench_core.build_scenarios()
    step = max(1, len(scenarios) // args.limit)
    scenarios = scenarios[::step][:args.limit]

    for k in args.k:
        hits = 0
        for s in scenarios:
            h.reset()
            for i, turn in enumerate(s.context):
                h.history.append({"role": "user" if i % 2 == 0 else "assistant",
                                  "content": turn})
            h.history.append({"role": "user", "content": s.prompt})
            ids = h._encode(h.history)
            cands = h._candidates(ids, 20, k)
            if any(bench_core.score(s.intent, c) for c in cands):
                hits += 1
        print(f"  top-{k:<2d} contains a correct reply: {hits / len(scenarios):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
