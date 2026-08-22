#!/usr/bin/env python3
"""Evaluate harness modes against the visible development benchmarks.

    python scripts/eval_harness.py --checkpoint <ckpt> --modes A_RAW C_POLICY ...

Scores Core-Bench (conversational reflexes) and StateBench (short-horizon state)
through the harness. Reports the behavioural metrics the harness is supposed to
move -- question rate, response length, repetition -- alongside cost, because a
mechanism that buys two points for four model calls is a different proposition
from one that buys two points for free.

The frozen SmallTalkBench-v2 is deliberately NOT used: it is the held-out
instrument and must not become a development target.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.core import bench_core, statebench
from smalltalk.harness import Harness, MODES
from smalltalk.harness.policy import BY_INTENT
from smalltalk.harness.trace import Trace
from smalltalk.model import SmallTalkModel
from smalltalk.tokenizer import SmallTalkTokenizer

# Core-Bench intent -> policy ontology. Only where the mapping is genuinely
# unambiguous; unmapped intents get no oracle, which keeps the oracle bound
# honest rather than flattering.
ORACLE_MAP = {
    "greeting": "GREETING", "greeting_how_are_you": "GREETING_HOW_ARE_YOU",
    "how_are_you": "HOW_ARE_YOU", "thanks": "THANKS", "apology": "APOLOGY",
    "goodbye": "GOODBYE", "good_news": "GOOD_NEWS", "bad_news": "BAD_NEWS",
    "tired": "TIRED", "bored": "BORED", "user_vents": "VENTING",
    "confused": "CONFUSION", "agreement": "AGREEMENT",
    "disagreement": "DISAGREEMENT", "user_jokes": "JOKE",
    "topic_statement": "TOPIC_STATEMENT", "out_of_scope": "UNKNOWN",
}


def core_eval(h: Harness, limit: int | None) -> dict:
    scenarios = bench_core.build_scenarios()
    if limit:
        step = max(1, len(scenarios) // limit)
        scenarios = scenarios[::step][:limit]
    ok, qs, lens, calls, lat = [], [], [], [], []
    per_intent = defaultdict(list)
    replies = []
    for s in scenarios:
        h.reset()
        for i, turn in enumerate(s.context):
            h.history.append({"role": "user" if i % 2 == 0 else "assistant",
                              "content": turn})
        oracle = None
        if h.cfg.oracle_policy:
            name = ORACLE_MAP.get(s.intent)
            oracle = BY_INTENT.get(name) if name else None
        tr = Trace()
        r = h.reply(s.prompt, oracle=oracle, trace=tr)
        good = bool(bench_core.score(s.intent, r))
        ok.append(good)
        per_intent[s.intent].append(good)
        qs.append(r.strip().endswith("?"))
        lens.append(len(h.tokenizer.encode(r)))
        calls.append(tr.model_calls)
        lat.append(tr.latency_ms)
        replies.append({"intent": s.intent, "prompt": s.prompt, "reply": r,
                        "correct": good, "policy": tr.policy,
                        "policy_source": tr.policy_source})
    dupes = len(replies) - len({r["reply"].strip().lower() for r in replies})
    return {
        "n": len(scenarios),
        "pass_rate": sum(ok) / len(ok),
        "question_rate": sum(qs) / len(qs),
        "mean_tokens": statistics.mean(lens),
        "distinct_reply_ratio": 1 - dupes / len(replies),
        "mean_model_calls": statistics.mean(calls),
        "mean_latency_ms": statistics.mean(lat),
        "per_intent": {k: sum(v) / len(v) for k, v in per_intent.items()},
        "replies": replies,
    }


def state_eval(h: Harness) -> dict:
    reps = {}
    for t in statebench.trajectories():
        h.reset()
        for i, turn in enumerate(t.turns):
            h.history.append({"role": "user" if i % 2 == 0 else "assistant",
                              "content": turn})
        reps[t.id] = h.reply(t.probe)
    div = sum(1 for a, b in statebench.pairs()
              if reps[a.id].strip().lower() != reps[b.id].strip().lower())
    direct = sum(1 for a, b in statebench.pairs()
                 if statebench.is_correct(a, reps[a.id])
                 and statebench.is_correct(b, reps[b.id]))
    n = len(statebench.pairs())
    return {"divergence": div / n, "directional": direct / n, "replies": reps}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/state/tokenizer-4096")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--limit", type=int, default=120,
                    help="Core-Bench scenarios (evenly sampled); 0 = all")
    ap.add_argument("--out", default="reports/harness")
    args = ap.parse_args()

    model = SmallTalkModel.from_pretrained(args.checkpoint, device=args.device).eval()
    tok = SmallTalkTokenizer.load(args.tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 6_689_024, f"base model changed: {n_params:,}"

    results = {}
    for name in args.modes:
        h = Harness(model=model, tokenizer=tok, cfg=MODES[name])
        t0 = time.perf_counter()
        core = core_eval(h, args.limit or None)
        h.reset()
        st = state_eval(h)
        results[name] = {"core": core, "state": st,
                         "wall_s": time.perf_counter() - t0}
        print(f"{name:22s} core {core['pass_rate']:.3f}  "
              f"state_dir {st['directional']:.3f}  "
              f"q {core['question_rate']:.2f}  "
              f"tok {core['mean_tokens']:.1f}  "
              f"calls {core['mean_model_calls']:.1f}  "
              f"{core['mean_latency_ms']:.0f}ms", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "params": n_params, "results": results},
        indent=2))
    print(f"\nwrote {out}/ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
