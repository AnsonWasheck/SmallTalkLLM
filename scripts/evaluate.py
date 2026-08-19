#!/usr/bin/env python3
"""Run SmallTalkBench on one or more checkpoints.

python scripts/evaluate.py --checkpoint artifacts/runs/sft-4m/best
python scripts/evaluate.py --checkpoint A B C --out artifacts/eval        # scaling curve
python scripts/evaluate.py --checkpoint A --sweep                          # decoding sweep
python scripts/evaluate.py --checkpoint A --judge-out artifacts/eval/judge.jsonl
python scripts/evaluate.py --checkpoint A B --pairwise artifacts/eval/pairwise.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.eval.bench import load_scenarios, save_scenarios
from smalltalk.eval.judge import (
    aggregate_judge_scores,
    build_judge_requests,
    build_pairwise,
    read_judge_scores,
    write_judge_file,
    write_pairwise,
)
from smalltalk.eval.runner import (
    DEFAULT_SWEEP_GRID,
    evaluate_model,
    run_bench,
    sweep_generation,
)
from smalltalk.infer.generate import GenerationConfig, load_engine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--scenarios", default=None, help="JSONL of scenarios (default: built-in)")
    ap.add_argument("--val-data", default="data/processed/val.jsonl")
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--use-state", action="store_true", help="enable zero-parameter dialogue state")
    # decoding
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--sweep", action="store_true", help="sweep decoding params")
    # judge / human eval
    ap.add_argument("--judge-out", default=None)
    ap.add_argument("--judge-scores", default=None, help="ingest completed judge scores")
    ap.add_argument("--pairwise", default=None, help="write blind A/B file (needs 2 checkpoints)")
    ap.add_argument("--export-scenarios", default=None)
    args = ap.parse_args()

    if args.export_scenarios:
        p = save_scenarios(args.export_scenarios)
        print(f"[bench] scenarios -> {p}")

    scenarios = load_scenarios(args.scenarios)
    gen = GenerationConfig(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens,
    )
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    transcripts_by_model: dict[str, list] = {}

    for ckpt in args.checkpoint:
        name = Path(ckpt).parent.name + "/" + Path(ckpt).name
        engine = load_engine(ckpt, args.tokenizer, device=args.device, gen=gen,
                             use_state=args.use_state)
        print(f"\n=== {name}  ({engine.model.num_parameters():,} params) ===")

        if args.sweep:
            results = sweep_generation(engine, DEFAULT_SWEEP_GRID, scenarios, seed=args.seed)
            p = out_root / f"{name.replace('/', '_')}-decoding-sweep.json"
            p.write_text(json.dumps(results, indent=2, default=float))
            print(f"[sweep] best: {json.dumps(results[0], default=float)}\n -> {p}")
            gen = gen.with_(**results[0]["params"])
            engine.gen = gen

        model_out = out_root / name.replace("/", "_")
        summary = evaluate_model(
            engine, name, scenarios, gen, val_path=args.val_data,
            out_dir=model_out, seed=args.seed,
        )
        b = summary["smalltalkbench"]
        print(f"  params                 {summary['params']:,}")
        print(f"  clean_10turn_rate      {b.get('clean_10turn_rate')}   <-- PRIMARY METRIC")
        print(f"  clean_5turn_rate       {b.get('clean_5turn_rate')}")
        print(f"  broken_turn_rate       {b.get('broken_turn_rate')}")
        print(f"  grammatical_rate       {b.get('grammatical_rate')}")
        print(f"  loop_rate              {b.get('loop_rate')}")
        print(f"  probe_pass_rate        {b.get('probe_pass_rate')}")
        print(f"  distinct_2 / mean_len  {b.get('distinct_2')} / {b.get('mean_len')}")
        if "validation" in summary:
            print(f"  validation             {summary['validation']}")
        all_summaries.append(summary)

        _, transcripts = run_bench(engine, scenarios, gen, seed=args.seed)
        transcripts_by_model[name] = transcripts

        if args.judge_out:
            reqs = build_judge_requests(transcripts, name)
            p = write_judge_file(
                Path(args.judge_out).with_name(
                    f"{Path(args.judge_out).stem}-{name.replace('/', '_')}.jsonl"
                ), reqs,
            )
            print(f"[judge] {len(reqs)} requests -> {p}")

    if args.judge_scores:
        scores = read_judge_scores(args.judge_scores)
        agg = aggregate_judge_scores(scores)
        print(f"[judge] aggregated {len(scores)} judgements: {json.dumps(agg, indent=2)}")
        (out_root / "judge_aggregate.json").write_text(json.dumps(agg, indent=2))

    if args.pairwise:
        names = list(transcripts_by_model)
        if len(names) < 2:
            print("[pairwise] needs two checkpoints; skipped")
        else:
            items, key = build_pairwise(
                transcripts_by_model[names[0]], transcripts_by_model[names[1]],
                names[0], names[1],
            )
            p, kp = write_pairwise(args.pairwise, items, key)
            print(f"[pairwise] {len(items)} blind comparisons -> {p} (key: {kp})")

    combined = out_root / "all_models.json"
    combined.write_text(json.dumps(all_summaries, indent=2, default=float))
    print(f"\n[write] {combined}")
    if len(all_summaries) > 1:
        print("\nparams        clean_10turn  clean_5turn  broken  grammatical  val_ppl")
        for s in sorted(all_summaries, key=lambda s: s["params"]):
            b = s["smalltalkbench"]
            v = s.get("validation", {})
            print(
                f"{s['params']:>11,}  {b.get('clean_10turn_rate', 0):>11}  "
                f"{b.get('clean_5turn_rate', 0):>10}  {b.get('broken_turn_rate', 0):>6}  "
                f"{b.get('grammatical_rate', 0):>11}  {v.get('val_ppl_assistant', '-'):>7}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
