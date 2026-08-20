#!/usr/bin/env python3
"""Score a checkpoint on StateBench-v1 (deterministic, greedy).

  python scripts/eval_state.py --checkpoint <ckpt> --tag r004 [--device cpu]
  python scripts/eval_state.py --freeze

Reports, in order of how much they matter:

  divergence   -- fraction of counterfactual pairs where the two sides get
                  DIFFERENT replies. The probe turns are byte-identical, so a
                  model conditioning only on the last user turn scores 0.0 here
                  by construction, however good its reflexes are.
  directional  -- fraction of pairs where both sides are not merely different
                  but correct: positive side positive, negative side negative.
  accuracy     -- per-trajectory valence correctness.
  neutral_rate -- declining to commit. Not scored as correct: a model that
                  always says "yeah" is not tracking state, it is abstaining.
  attractor_rate -- collapse into the generic high-prior replies.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.core.statebench import (checksum, classify, freeze, is_correct,
                                       pairs, trajectories, verify_frozen)
from smalltalk.infer.generate import GenerationConfig, load_engine


def run(engine, gen) -> dict:
    replies: dict[str, str] = {}
    for t in trajectories():
        engine.reset()
        for i, turn in enumerate(t.turns):
            engine.history.append(
                {"role": "user" if i % 2 == 0 else "assistant", "content": turn})
        replies[t.id] = engine.reply(t.probe, gen=gen)
    return replies


def summarise(replies: dict[str, str]) -> dict:
    trajs = trajectories()
    acc = [is_correct(t, replies[t.id]) for t in trajs]
    kinds = Counter(classify(replies[t.id]) for t in trajs)

    div, directional, rows = 0, 0, []
    for a, b in pairs():
        ra, rb = replies[a.id], replies[b.id]
        differs = ra.strip().lower() != rb.strip().lower()
        both = is_correct(a, ra) and is_correct(b, rb)
        div += differs
        directional += both
        rows.append({"pair": a.pair_id, "topic": a.topic, "probe": a.probe,
                     "positive_reply": ra, "negative_reply": rb,
                     "differs": differs, "both_correct": both})

    n_pairs = len(pairs())
    return {
        "checksum": checksum(),
        "divergence": div / n_pairs,
        "directional": directional / n_pairs,
        "accuracy": sum(acc) / len(acc),
        "neutral_rate": kinds["neutral"] / len(trajs),
        "attractor_rate": kinds["attractor"] / len(trajs),
        "other_rate": kinds["other"] / len(trajs),
        "pairs": rows,
        "replies": replies,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer", default="artifacts/core/tokenizer-4096")
    ap.add_argument("--tag")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--frozen", default="benchmarks/statebench_frozen.json")
    ap.add_argument("--out", default="reports/state")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.freeze:
        print(f"froze StateBench checksum={freeze(args.frozen)}")
        return 0
    if not args.checkpoint or not args.tag:
        ap.error("--checkpoint and --tag are required unless --freeze")
    verify_frozen(args.frozen)

    gen = GenerationConfig(temperature=0.0, top_p=1.0, top_k=0, greedy=True,
                           repetition_penalty=1.0, max_new_tokens=16, seed=0)
    engine = load_engine(args.checkpoint, args.tokenizer, device=args.device, gen=gen)

    t0 = time.time()
    res = summarise(run(engine, gen))
    res.update({"tag": args.tag, "checkpoint": args.checkpoint})

    print(f"\nStateBench {res['checksum']}  {len(pairs())} pairs  "
          f"{time.time() - t0:.0f}s\n")
    print(f"  divergence     {res['divergence']:6.1%}   <- pairs answered differently")
    print(f"  directional    {res['directional']:6.1%}   <- and answered correctly")
    print(f"  accuracy       {res['accuracy']:6.1%}")
    print(f"  neutral_rate   {res['neutral_rate']:6.1%}   (abstaining)")
    print(f"  attractor_rate {res['attractor_rate']:6.1%}")
    print(f"  other_rate     {res['other_rate']:6.1%}")

    if not args.quiet:
        print("\n  pair                 positive-side / negative-side")
        for r in res["pairs"]:
            mark = "DIFF" if r["differs"] else "same"
            print(f"  {r['pair']:10s} [{mark}] {r['positive_reply'][:32]!r} / "
                  f"{r['negative_reply'][:32]!r}")

    out = Path(args.out) / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
