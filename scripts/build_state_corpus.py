#!/usr/bin/env python3
"""Build the v0.3-Core-State corpus.

  python scripts/build_state_corpus.py --out data/state

Three components, mixed deliberately:

  state_gen  counterfactual trajectory pairs. The only part of any corpus this
             project has had where the correct answer depends on conversational
             state rather than on the last user turn.
  core_gen   the v0.2 reflex curriculum, retained at reduced share. Reflexes are
             still needed and still measured by Core-Bench; the failure was never
             that they are wrong, only that they are insufficient.
  real       DailyDialog / EmpatheticDialogues for language. Truncated, because
             long real conversations bought a disproportionate free-form prior.

Two fail-closed gates: no StateBench opener may appear, and no Core-Bench probe
surface may appear. Either one aborts the build.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.core.bench_core import build_scenarios as core_scenarios
from smalltalk.core.core_gen import CoreConfig, generate as core_generate
from smalltalk.core.intents import normalise
from smalltalk.core.state_gen import BENCH_SURFACES, StateConfig, generate as state_generate
from smalltalk.data import adapters
from smalltalk.data.clean import FilterConfig, clean_conversations, corpus_stats
from smalltalk.data.schema import write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dailydialog", default="data/raw/dailydialog")
    ap.add_argument("--empathetic", default="data/raw/empathetic")
    ap.add_argument("--state", type=int, default=40000)
    ap.add_argument("--core", type=int, default=24000)
    ap.add_argument("--real-in-sft", type=int, default=4000)
    ap.add_argument("--max-real-turns", type=int, default=6)
    ap.add_argument("--cap-frac", type=float, default=0.020,
                    help="max share of assistant turns any single target may hold")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/state")
    args = ap.parse_args()

    real = list(adapters.load_dailydialog(args.dailydialog))
    real += list(adapters.load_empathetic_dialogues(args.empathetic))
    real, stats = clean_conversations(real, FilterConfig())
    print(f"[real] {len(real):,} after cleaning")

    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", t.lower())).strip()

    state = list(state_generate(StateConfig(n=args.state, seed=args.seed)))
    print(f"[state] {len(state):,} trajectories ({len(state)//2:,} counterfactual pairs)")

    # Real dialogue genuinely contains phrases like "i got a raise", which are
    # StateBench openers. core_gen draws its conversational context from this
    # pool, so an unfiltered pool leaks the test into training as context. Filter
    # the pool rather than dropping the generated examples afterwards: the aim is
    # for the corpus to be clean by construction, not cleaned up after the fact.
    def clean_of_bench(c) -> bool:
        return not any(norm(m.content) in BENCH_SURFACES
                       for m in c.messages if m.role == "user")

    pool = [c for c in real if clean_of_bench(c)]
    print(f"[pool] {len(real) - len(pool)} real conversations dropped for StateBench overlap")
    core = list(core_generate(CoreConfig(n=args.core, seed=args.seed), real_convs=pool))
    print(f"[core] {len(core):,} reflex examples")

    # ---- gate 1: StateBench openers -------------------------------------
    hits = [c.id for c in state + core
            for m in c.messages if m.role == "user" and norm(m.content) in BENCH_SURFACES]
    if hits:
        raise SystemExit(f"LEAKAGE: {len(hits)} examples reproduce a StateBench opener "
                         f"(e.g. {hits[:3]}). Refusing to write.")
    print(f"[leak] 0 examples touch the {len(BENCH_SURFACES)} StateBench openers")

    # ---- gate 2: Core-Bench probes --------------------------------------
    probes = {normalise(s.prompt) for s in core_scenarios()}
    hits = [c.id for c in state + core
            for m in c.messages if m.role == "user" and normalise(m.content) in probes]
    if hits:
        raise SystemExit(f"LEAKAGE: {len(hits)} examples reproduce a Core-Bench probe "
                         f"(e.g. {hits[:3]}). Refusing to write.")
    print(f"[leak] 0 examples touch the {len(probes)} Core-Bench probes")

    r = random.Random(args.seed)
    r.shuffle(real)
    n_val = max(1, int(len(real) * args.val_ratio))
    real_val, real_train = real[:n_val], real[n_val:]

    def truncate(c):
        if len(c.messages) > args.max_real_turns:
            c.messages = c.messages[: args.max_real_turns]
            if c.messages and c.messages[-1].role == "user":
                c.messages = c.messages[:-1]
        return c

    # Split state pairs by FAMILY so counterfactual partners never separate: if
    # one side landed in val the model could infer the other from its complement.
    fams = sorted({c.meta["family"] for c in state})
    r.shuffle(fams)
    val_fams = set(fams[: max(1, int(len(fams) * args.val_ratio))])
    state_val = [c for c in state if c.meta["family"] in val_fams]
    state_train = [c for c in state if c.meta["family"] not in val_fams]

    r.shuffle(core)
    c_val = max(1, int(len(core) * args.val_ratio))
    core_val, core_train = core[:c_val], core[c_val:]

    sft_train = state_train + core_train + [truncate(c) for c in real_train[: args.real_in_sft]]
    sft_val = state_val + core_val + [truncate(c) for c in real_val[:80]]
    r.shuffle(sft_train)

    # ---- attractor report (not enforced; measured so the loop can react) --
    tgt = Counter(m.content for c in sft_train for m in c.messages if m.role == "assistant")
    n_assist = sum(tgt.values())
    top = tgt.most_common(1)[0]
    print(f"[attractor] most frequent target {top[1] / n_assist:.2%} "
          f"(cap {args.cap_frac:.1%}): {top[0][:44]!r}")

    out = Path(args.out)
    counts = {
        "pretrain_train": write_jsonl(out / "pretrain_train.jsonl", real_train),
        "pretrain_val": write_jsonl(out / "pretrain_val.jsonl", real_val),
        "sft_train": write_jsonl(out / "sft_train.jsonl", sft_train),
        "sft_val": write_jsonl(out / "sft_val.jsonl", sft_val),
    }
    (out / "corpus_report.json").write_text(json.dumps({
        "counts": counts, "state": len(state), "core": len(core),
        "state_pairs": len(state) // 2,
        "top_target_share": top[1] / n_assist,
        "stats_sft_train": corpus_stats(sft_train),
    }, indent=2, default=str))
    for k, v in counts.items():
        print(f"[write] {out}/{k}.jsonl  {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
