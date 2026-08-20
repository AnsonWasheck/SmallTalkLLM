#!/usr/bin/env python3
"""Build the v0.2-Core corpus: a dense language base + a narrow reflex curriculum.

  python scripts/build_core_corpus.py --out data/core

Two files come out, and they are deliberately different in character:

  pretrain_*.jsonl -- real DailyDialog/EmpatheticDialogues, cleaned. This teaches
      English and conversational rhythm. Loss on all tokens.
  sft_*.jsonl      -- the Core curriculum: broad input surface, one canonical
      target per intent, prefixed with a length-policy token. Loss on assistant
      tokens only. A slice of real dialogue is mixed in so SFT does not erase the
      language model, but Core dominates because reliability is the objective.

Splits are family-level (`core:<intent>` families are split by *paraphrase*, not
by example, upstream in intents.py: held-out surfaces never appear here at all).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.core.core_gen import CoreConfig, generate as core_generate
from smalltalk.core.bench_core import build_scenarios
from smalltalk.core.intents import normalise
from smalltalk.data import adapters
from smalltalk.data.clean import FilterConfig, clean_conversations, corpus_stats
from smalltalk.data.schema import write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dailydialog", default="data/raw/dailydialog")
    ap.add_argument("--empathetic", default="data/raw/empatheticdialogues")
    ap.add_argument("--core", type=int, default=60000)
    ap.add_argument("--real-in-sft", type=int, default=4000,
                    help="real conversations mixed into SFT to preserve language")
    ap.add_argument("--max-real-turns", type=int, default=6,
                    help="truncate real SFT conversations; long ones dominate the "
                         "assistant-token budget and teach a free-form-reply prior")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="data/core")
    args = ap.parse_args()

    real = list(adapters.load_dailydialog(args.dailydialog))
    real += list(adapters.load_empathetic_dialogues(args.empathetic))
    print(f"[real] ingested {len(real):,}")
    real, stats = clean_conversations(real, FilterConfig())
    print("[clean]", stats.report().replace("\n", "\n[clean] "))

    # The autonomous loop up-weights intents that missed their tier target. The
    # multipliers live outside the code so a round's curriculum is reconstructable
    # from artifacts alone, and so the loop never edits importable source.
    cfg = CoreConfig(n=args.core, seed=args.seed)
    boosts = Path("artifacts/core_loop/weights.json")
    if boosts.exists():
        for name, mult in json.loads(boosts.read_text()).items():
            if name in cfg.weights:
                cfg.weights[name] = round(cfg.weights[name] * float(mult), 3)
        print(f"[weights] applied {boosts}: "
              + ", ".join(f"{k}={v}" for k, v in sorted(cfg.weights.items())))

    core = list(core_generate(cfg, real_convs=real))
    print(f"[core] generated {len(core):,}")

    # Fail-closed leakage gate: no Core-Bench surface may appear as a user turn.
    probes = {normalise(s.prompt) for s in build_scenarios()}
    hits = [c.id for c in core
            for m in c.messages if m.role == "user" and normalise(m.content) in probes]
    if hits:
        raise SystemExit(f"LEAKAGE: {len(hits)} Core examples reproduce a bench prompt "
                         f"(e.g. {hits[:3]}). Refusing to write.")
    print(f"[leak] 0 of {len(core):,} Core examples touch the {len(probes)} bench surfaces")

    r = random.Random(args.seed)
    r.shuffle(real)
    n_val = max(1, int(len(real) * args.val_ratio))
    real_val, real_train = real[:n_val], real[n_val:]

    r.shuffle(core)
    c_val = max(1, int(len(core) * args.val_ratio))
    core_val, core_train = core[:c_val], core[c_val:]

    # Measured on the first baseline: real dialogue was 27.5% of assistant tokens
    # and produced 25% of Core-Bench errors as free-form chat ("who wrote hamlet"
    # -> "Oh that's a good idea. I hope you get a raise."). The model had learned a
    # ~27% prior on "just talk" and applied it regardless of input. Real dialogue
    # still has to be here or the language model degrades and ordinary off-topic
    # chat gets a canned deflection -- so trim its share rather than remove it.
    def _truncate(c):
        if len(c.messages) <= args.max_real_turns:
            return c
        c.messages = c.messages[: args.max_real_turns]
        if c.messages[-1].role == "user":       # never end on a user turn
            c.messages = c.messages[:-1]
        return c

    real_for_sft = [_truncate(c) for c in real_train[: args.real_in_sft]]
    sft_train = core_train + real_for_sft
    sft_val = core_val + [_truncate(c) for c in real_val[: max(1, args.real_in_sft // 50)]]
    r.shuffle(sft_train)

    out = Path(args.out)
    counts = {
        "pretrain_train": write_jsonl(out / "pretrain_train.jsonl", real_train),
        "pretrain_val": write_jsonl(out / "pretrain_val.jsonl", real_val),
        "sft_train": write_jsonl(out / "sft_train.jsonl", sft_train),
        "sft_val": write_jsonl(out / "sft_val.jsonl", sft_val),
    }
    (out / "corpus_report.json").write_text(json.dumps({
        "counts": counts,
        "core_generated": len(core),
        "real_in_sft": min(args.real_in_sft, len(real_train)),
        "dropped": dict(stats.dropped),
        "stats_sft_train": corpus_stats(sft_train),
    }, indent=2, default=str))
    for k, v in counts.items():
        print(f"[write] {out}/{k}.jsonl  {v:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
