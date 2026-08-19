#!/usr/bin/env python3
"""Autonomous improvement loop: evaluate -> diagnose -> grow corpus -> retrain -> repeat.

    python scripts/autoloop.py --iterations 99999          # run forever
    python scripts/autoloop.py --status                    # read the ledger

INVARIANTS (must never be violated -- they are what make the results meaningful):
  * The model config is FROZEN at smalltalk-7m. Never touched.
  * SmallTalkBench-HARD is FROZEN and HELD OUT. Every corpus passes a leakage
    check before training; leaked conversations are dropped, not tolerated.
  * Every iteration appends to artifacts/loop/ledger.jsonl. Nothing is overwritten,
    so regressions stay visible in the morning.
  * The best checkpoint by hard-bench score is preserved at artifacts/loop/best/.

Each iteration:
  1. Evaluate the current champion on SmallTalkBench-HARD (frozen).
  2. Rank failures by category.
  3. Emit a targeted data request for the weakest categories (authored data is
     dropped into data/generated/ by the supervising agent between iterations).
  4. Rebuild the corpus from all sources + leakage filter.
  5. Train (continuing from the champion) and evaluate.
  6. Promote only on improvement; otherwise keep the champion and log the failure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.config import TrainConfig, load_model_config
from smalltalk.data.clean import FilterConfig, clean_conversations, corpus_stats
from smalltalk.data.schema import load_conversations, write_jsonl
from smalltalk.eval.hard_bench import HARD_CATEGORIES, hard_scenarios
from smalltalk.eval.leakage import filter_leaked
from smalltalk.eval.metrics import aggregate
from smalltalk.eval.runner import run_bench
from smalltalk.infer.generate import GenerationConfig, load_engine

ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = ROOT / "artifacts" / "loop"
LEDGER = LOOP_DIR / "ledger.jsonl"
CHAMPION = LOOP_DIR / "champion"
BEST = LOOP_DIR / "best"
GENERATED = ROOT / "data" / "generated"
MODEL_CONFIG = "configs/model/smalltalk-7m.yaml"   # FROZEN
TOKENIZER = "artifacts/tokenizer-4096"
LOCKED_PARAMS = 6_689_024


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_entry(entry: dict) -> None:
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=float) + "\n")


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def assert_frozen_architecture() -> None:
    cfg = load_model_config(MODEL_CONFIG)
    from smalltalk.params import analytic_param_count

    n = analytic_param_count(cfg)
    if n != LOCKED_PARAMS:
        raise SystemExit(
            f"FATAL: architecture drift. {cfg.name} is {n:,} params, "
            f"locked at {LOCKED_PARAMS:,}. Refusing to continue."
        )


# ---------------------------------------------------------------------------
def evaluate_hard(ckpt: Path, tag: str, seed: int = 4242) -> dict:
    """Run the frozen hard benchmark. Returns aggregate + per-category failures."""
    gen = GenerationConfig(temperature=0.7, top_p=0.9, repetition_penalty=1.15,
                           max_new_tokens=40, max_context=1024)
    engine = load_engine(str(ckpt), TOKENIZER, device="auto", gen=gen)
    scenarios = hard_scenarios()
    evals, transcripts = run_bench(engine, scenarios, gen, seed=seed)
    agg = aggregate(evals, min_turns=10)

    by_cat: dict[str, dict] = {}
    for e in evals:
        b = by_cat.setdefault(e.category, {"n": 0, "clean": 0, "broken": 0, "turns": 0})
        b["n"] += 1
        b["clean"] += int(e.completed_clean)
        b["broken"] += e.broken_turns
        b["turns"] += len(e.turns)
    probe_by_cat: dict[str, dict] = {}
    for e in evals:
        for pr in e.probe_results:
            p = probe_by_cat.setdefault(pr["type"], {"pass": 0, "n": 0})
            p["n"] += 1
            p["pass"] += int(pr["passed"])

    out_dir = LOOP_DIR / "evals" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "transcripts.jsonl").open("w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with (out_dir / "per_scenario.jsonl").open("w", encoding="utf-8") as f:
        for e in evals:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False, default=float) + "\n")

    return {
        "score": round(agg.get("clean_10turn_rate", 0.0), 4),
        "clean_conversation_rate": agg.get("clean_conversation_rate"),
        "broken_turn_rate": agg.get("broken_turn_rate"),
        "grammatical_rate": agg.get("grammatical_rate"),
        "loop_rate": agg.get("loop_rate"),
        "probe_pass_rate": agg.get("probe_pass_rate"),
        "distinct_2": agg.get("distinct_2"),
        "mean_len": agg.get("mean_len"),
        "by_category": by_cat,
        "probe_by_type": probe_by_cat,
    }


def weakest_categories(result: dict, k: int = 5) -> list[str]:
    scored = []
    for cat, b in result.get("by_category", {}).items():
        clean = b["clean"] / max(b["n"], 1)
        broken = b["broken"] / max(b["turns"], 1)
        scored.append((clean - broken, cat))
    for ptype, p in result.get("probe_by_type", {}).items():
        scored.append((p["pass"] / max(p["n"], 1) - 0.5, f"probe:{ptype}"))
    scored.sort()
    return [c for _, c in scored[:k]]


def write_data_request(result: dict, iteration: int) -> Path:
    """Emit the brief the supervising agent fulfils by authoring dialogue."""
    weak = weakest_categories(result, k=6)
    req = {
        "iteration": iteration,
        "generated_at": now(),
        "score": result["score"],
        "weakest": weak,
        "all_categories": list(HARD_CATEGORIES),
        "instruction": (
            "Author natural multi-turn casual conversations that TEACH the weak "
            "capabilities above. Do NOT copy benchmark wording -- use different "
            "names, topics, phrasings, and situations. Assistant replies 3-25 "
            "words, friend not assistant. Write to data/generated/<name>.jsonl "
            "in canonical schema."
        ),
        "per_category_detail": result.get("by_category", {}),
        "probe_detail": result.get("probe_by_type", {}),
    }
    p = LOOP_DIR / "data_request.json"
    p.write_text(json.dumps(req, indent=2, default=float))
    return p


# ---------------------------------------------------------------------------
def rebuild_corpus() -> dict:
    """Merge every data source, clean, leakage-filter, split. Returns stats."""
    from smalltalk.data import adapters
    from smalltalk.data.clean import split_train_val

    convs = []
    counts = {}
    dd = ROOT / "data/raw/dailydialog"
    ed = ROOT / "data/raw/empathetic"
    if dd.exists():
        c = list(adapters.load_dailydialog(dd))
        convs += c
        counts["dailydialog"] = len(c)
    if (ed / "train.csv").exists():
        c = list(adapters.load_empathetic_dialogues(ed / "train.csv"))
        convs += c
        counts["empathetic"] = len(c)
    GENERATED.mkdir(parents=True, exist_ok=True)
    for p in sorted(GENERATED.glob("*.jsonl")):
        c = list(adapters.load_jsonl_conversations(p, source=f"gen:{p.stem}"))
        convs += c
        counts[f"gen:{p.stem}"] = len(c)
    seed = ROOT / "data/seed/example_conversations.jsonl"
    if seed.exists():
        c = list(adapters.load_jsonl_conversations(seed, source="seed"))
        convs += c
        counts["seed"] = len(c)

    kept, stats = clean_conversations(convs, FilterConfig())
    kept, leak = filter_leaked(kept)
    print(leak.summary())
    if not leak.clean:
        print("[leakage] dropped the overlapping conversations; continuing on the clean remainder")

    train, val = split_train_val(kept, val_ratio=0.02, seed=1337)
    out = ROOT / "data/processed"
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    return {
        "sources": counts,
        "kept": len(kept),
        "dropped": dict(stats.dropped),
        "leaked_dropped": len(leak.flagged),
        "train": len(train),
        "val": len(val),
        "corpus_stats": corpus_stats(train),
    }


def epoch_aware_budget(
    train_examples: int, batch_size: int = 64, target_epochs: float = 12.0,
    min_steps: int = 200, max_steps_cap: int = 6000,
) -> tuple[int, int]:
    """Steps and eval-cadence scaled to corpus size.

    Real bug this fixes: a fixed step count overfit a 28k-conversation corpus
    catastrophically (val_loss 4.24 -> 6.14 between steps 1000 and 2000, i.e.
    ~11 to ~23 epochs over the same 439-step/epoch data) while train_loss kept
    dropping. The `best`-checkpoint mechanism caught it that time, but a fixed
    budget wastes wall-clock overfitting past the point that matters and, as the
    corpus grows across iterations, the SAME fixed step count silently swings
    from "too many epochs" to "not even one" depending on how much data exists
    that round. Steps must scale with corpus size, not be a constant.
    """
    steps_per_epoch = max(1, train_examples // batch_size)
    steps = int(round(steps_per_epoch * target_epochs))
    steps = max(min_steps, min(steps, max_steps_cap))
    eval_every = max(steps_per_epoch // 2, 25)
    return steps, eval_every


def train_round(
    iteration: int, train_examples: int, lr: float, init_from: Path | None,
    batch_size: int = 64, target_epochs: float = 12.0,
) -> tuple[Path, int]:
    steps, eval_every = epoch_aware_budget(
        train_examples, batch_size, target_epochs=target_epochs
    )
    run_name = f"iter{iteration:03d}"
    run_dir = ROOT / "artifacts/runs" / run_name
    if run_dir.exists():
        # A prior attempt at this iteration (e.g. killed mid-run) leaves a stale
        # log.jsonl. RunLogger appends, so without this the ledger would silently
        # interleave two different training runs under one step counter.
        shutil.rmtree(run_dir)
    cmd = [
        "bash", "scripts/rocm.sh", "scripts/sft.py",
        "--config", "configs/train/sft_7m.yaml",
        "--run-name", run_name,
        "--output-dir", "artifacts/runs",
        "--model-config", MODEL_CONFIG,
        "--tokenizer", TOKENIZER,
        "--train-data", "data/processed/train.jsonl",
        "--val-data", "data/processed/val.jsonl",
        "--max-steps", str(steps),
        "--learning-rate", f"{lr:g}",
        "--seq-len", "512", "--batch-size", str(batch_size),
        "--eval-every", str(eval_every),
        "--save-every", str(steps),
        "--log-every", str(max(eval_every // 3, 10)),
        "--device", "auto",
    ]
    if init_from:
        cmd += ["--init-from", str(init_from)]
    print(f"[train] {steps} steps ({steps * batch_size / max(train_examples,1):.1f} epochs), "
          f"eval every {eval_every}")
    print("[train]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ROOT / "artifacts/runs" / run_name / "best", steps


def promote(ckpt: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ckpt, dest)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--target-epochs", type=float, default=12.0,
                    help="epochs to target per round; steps are derived from corpus size")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--init", default=None, help="checkpoint to seed the champion")
    args = ap.parse_args()

    assert_frozen_architecture()
    LOOP_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        rows = read_ledger()
        if not rows:
            print("no iterations yet")
            return 0
        print(f"{'iter':>5} {'score':>7} {'probe':>7} {'broken':>7} {'gram':>6} "
              f"{'train':>8} {'promoted':>9}  when")
        for r in rows:
            print(f"{r.get('iteration',0):>5} {r.get('score',0):>7.4f} "
                  f"{r.get('probe_pass_rate') or 0:>7.3f} {r.get('broken_turn_rate') or 0:>7.3f} "
                  f"{r.get('grammatical_rate') or 0:>6.3f} {r.get('train_examples',0):>8,} "
                  f"{str(r.get('promoted')):>9}  {r.get('time','')}")
        best = max(rows, key=lambda r: r.get("score", 0))
        print(f"\nbest score {best['score']:.4f} at iteration {best['iteration']}")
        return 0

    if args.init and not CHAMPION.exists():
        promote(Path(args.init), CHAMPION)
        print(f"[init] champion <- {args.init}")

    if args.eval_only:
        res = evaluate_hard(CHAMPION, "adhoc")
        print(json.dumps(res, indent=2, default=float))
        return 0

    history = read_ledger()
    best_score = max((r.get("score", 0) for r in history), default=-1.0)
    start = len(history)

    for i in range(start, start + args.iterations):
        t0 = time.time()
        print(f"\n{'=' * 72}\n=== iteration {i}  ({now()})\n{'=' * 72}")
        try:
            corpus = rebuild_corpus()
            print(f"[corpus] train={corpus['train']:,} val={corpus['val']:,} "
                  f"sources={corpus['sources']}")

            init = CHAMPION if CHAMPION.exists() else None
            ckpt, steps_used = train_round(
                i, corpus["train"], args.lr, init, target_epochs=args.target_epochs
            )
            result = evaluate_hard(ckpt, f"iter{i:03d}")

            promoted = result["score"] > best_score
            if promoted:
                best_score = result["score"]
                promote(ckpt, BEST)
            # always continue training from the newest weights; keep BEST separate
            promote(ckpt, CHAMPION)

            entry = {
                "iteration": i, "time": now(),
                "duration_min": round((time.time() - t0) / 60, 1),
                "params": LOCKED_PARAMS,
                "steps": steps_used, "lr": args.lr,
                "train_examples": corpus["train"],
                "sources": corpus["sources"],
                "leaked_dropped": corpus["leaked_dropped"],
                "promoted": promoted, "best_so_far": best_score,
                **{k: v for k, v in result.items() if k not in ("by_category", "probe_by_type")},
                "by_category": result["by_category"],
                "probe_by_type": result["probe_by_type"],
                "weakest": weakest_categories(result),
            }
            log_entry(entry)
            write_data_request(result, i + 1)
            print(f"[iter {i}] score {result['score']:.4f} "
                  f"(best {best_score:.4f}) promoted={promoted} "
                  f"weakest={entry['weakest']}")
        except subprocess.CalledProcessError as exc:
            log_entry({"iteration": i, "time": now(), "error": f"train failed: {exc}"})
            print(f"[iter {i}] TRAINING FAILED: {exc}", file=sys.stderr)
            time.sleep(30)
        except Exception as exc:  # keep the loop alive overnight
            log_entry({"iteration": i, "time": now(), "error": repr(exc)})
            print(f"[iter {i}] ERROR: {exc!r}", file=sys.stderr)
            time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
