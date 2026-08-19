#!/usr/bin/env python3
"""Evaluate a checkpoint against the frozen SmallTalkBench-v2 (396 scenarios).

python scripts/eval_bench_v2.py --checkpoint artifacts/loop/best --tag pre_teacher_v2

Refuses to run if the benchmark checksum has drifted. Saves EVERY raw generation
so any reported number can be reconstructed. Reports aggregate + per-skill +
length/diversity/memory/epistemic breakdowns.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.eval.bench_v2 import SKILLS, build_scenarios, verify_frozen
from smalltalk.eval.metrics import (
    aggregate,
    distinct_n,
    evaluate_transcript,
    loop_detected,
    type_token_ratio,
    words,
)
from smalltalk.infer.generate import GenerationConfig, load_engine


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- honest CIs on small per-skill samples."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer-4096")
    ap.add_argument("--tag", required=True, help="baseline name, e.g. pre_teacher_v2")
    ap.add_argument("--out-root", default="reports/evals")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20240819)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--use-state", action="store_true", help="production track: zero-param memory")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N scenarios")
    args = ap.parse_args()

    verify_frozen()  # fail closed if the benchmark drifted
    scenarios = build_scenarios()
    if args.limit:
        scenarios = scenarios[: args.limit]

    gen = GenerationConfig(temperature=args.temperature, top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty,
                           max_new_tokens=args.max_new_tokens, max_context=1024)
    engine = load_engine(args.checkpoint, args.tokenizer, device=args.device,
                         gen=gen, use_state=args.use_state)
    n_params = engine.model.num_parameters()

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    evals, transcripts = [], []
    for i, sc in enumerate(scenarios):
        cfg = gen.with_(seed=args.seed + i)
        msgs = engine.run_scenario(sc.user_turns, gen=cfg)
        ev = evaluate_transcript(msgs, scenario_id=sc.id, category=sc.category,
                                 probes=sc.probes)
        evals.append(ev)
        transcripts.append({"scenario_id": sc.id, "skill": sc.category,
                            "messages": msgs})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(scenarios)} scenarios ({time.time()-t0:.0f}s)")

    # ---- per-skill ---------------------------------------------------------
    per_skill: dict[str, dict] = {}
    for skill in SKILLS:
        sk = [e for e in evals if e.category == skill]
        if not sk:
            continue
        probes = [p for e in sk for p in e.probe_results]
        n_pass = sum(p["passed"] for p in probes)
        clean = sum(e.completed_clean for e in sk)
        lo, hi = wilson(n_pass, len(probes)) if probes else (0.0, 0.0)
        per_skill[skill] = {
            "scenarios": len(sk),
            "probe_pass_rate": round(n_pass / len(probes), 4) if probes else None,
            "probe_ci95": [round(lo, 3), round(hi, 3)] if probes else None,
            "clean_conversation_rate": round(clean / len(sk), 4),
            "broken_turn_rate": round(
                sum(e.broken_turns for e in sk) / max(sum(len(e.turns) for e in sk), 1), 4),
            "mean_reply_words": round(statistics.mean(
                [t.n_words for e in sk for t in e.turns] or [0]), 2),
        }

    replies = [t.response for e in evals for t in e.turns]
    lens = [len(words(r)) for r in replies]
    agg = aggregate(evals, min_turns=10)

    def band(lo, hi):
        return round(sum(lo <= x <= hi for x in lens) / max(len(lens), 1), 4)

    summary = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "params": n_params,
        "track": "production(state)" if args.use_state else "naked",
        "benchmark": {"name": "SmallTalkBench-v2", "checksum": "a2ce68928e780ce5",
                      "scenarios": len(scenarios)},
        "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                     "repetition_penalty": args.repetition_penalty,
                     "max_new_tokens": args.max_new_tokens, "seed_base": args.seed},
        "aggregate": {
            "probe_pass_rate": agg.get("probe_pass_rate"),
            "clean_conversation_rate": agg.get("clean_conversation_rate"),
            "clean_10turn_rate": agg.get("clean_10turn_rate"),
            "broken_turn_rate": agg.get("broken_turn_rate"),
            "grammatical_rate": agg.get("grammatical_rate"),
            "loop_rate": agg.get("loop_rate"),
        },
        "length": {
            "mean_words": round(statistics.mean(lens), 2) if lens else 0,
            "median_words": statistics.median(lens) if lens else 0,
            "p90_words": sorted(lens)[int(0.9 * len(lens))] if lens else 0,
            "frac_1_word": band(1, 1), "frac_2_12": band(2, 12),
            "frac_13_30": band(13, 30), "frac_over_30": round(
                sum(x > 30 for x in lens) / max(len(lens), 1), 4),
        },
        "diversity": {
            "distinct_1": round(distinct_n(replies, 1), 4),
            "distinct_2": round(distinct_n(replies, 2), 4),
            "type_token_ratio": round(type_token_ratio(replies), 4),
            "unique_reply_frac": round(len(set(replies)) / max(len(replies), 1), 4),
        },
        "memory": {k: per_skill.get(k, {}).get("probe_pass_rate")
                   for k in ("long_memory", "memory_update", "memory_absent",
                             "persona_consistency")},
        "epistemic": {k: per_skill.get(k, {}).get("probe_pass_rate")
                      for k in ("no_fabrication", "epistemic_discrimination",
                                "memory_absent")},
        "per_skill": per_skill,
        "runtime_s": round(time.time() - t0, 1),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    with (out_dir / "transcripts.jsonl").open("w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with (out_dir / "per_scenario.jsonl").open("w", encoding="utf-8") as f:
        for e in evals:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False, default=float) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "per_skill"},
                     indent=2, default=float))
    print(f"\n{'skill':<26}{'probes':>7}{'pass':>7}{'ci95':>16}{'clean':>7}{'words':>7}")
    for s, v in sorted(per_skill.items(), key=lambda kv: (kv[1]['probe_pass_rate'] is None,
                                                          kv[1]['probe_pass_rate'] or 0)):
        ci = f"[{v['probe_ci95'][0]:.2f},{v['probe_ci95'][1]:.2f}]" if v["probe_ci95"] else "-"
        pp = f"{v['probe_pass_rate']:.3f}" if v["probe_pass_rate"] is not None else "-"
        print(f"{s:<26}{v['scenarios']:>7}{pp:>7}{ci:>16}"
              f"{v['clean_conversation_rate']:>7.2f}{v['mean_reply_words']:>7.1f}")
    print(f"\n[saved] {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
