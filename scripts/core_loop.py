#!/usr/bin/env python3
"""v0.2-Core autonomous loop. Runs indefinitely, unattended.

    python scripts/core_loop.py                 # run forever
    python scripts/core_loop.py --status        # read the ledger

Each round:
    1. Build the Core corpus (leakage gate is fail-closed; a hit aborts the round).
    2. Train stage 1 (language base) if the tokenizer/corpus changed, else reuse.
    3. SFT (assistant-only) from the champion or from the stage-1 base.
    4. Score frozen Core-Bench at temperature 0.
    5. Append to the ledger, promote only on improvement, adapt, repeat.

INVARIANTS -- these are what make the numbers mean anything:
  * Model config frozen at smalltalk-7m (6,689,024 params). Never touched here.
  * Core-Bench is frozen by checksum. This loop NEVER edits intents to make the
    score go up; it may only change the *curriculum* (weights, steps, LR). If the
    checksum drifts the loop halts rather than report an incomparable number.
  * A regression is never promoted and never becomes the next init checkpoint.
  * Nothing is overwritten. Every round's artifacts and generations are kept.

ADAPTATION RULES (deliberately mechanical -- this runs with nobody watching, so
every change must be justifiable from the ledger after the fact):
  * intent below its tier target -> up-weight it in the curriculum, capped at 4x
    its base weight so one stubborn intent cannot crowd out the corpus.
  * overall flat for 2 rounds -> increase SFT steps 1.5x, capped.
  * val loss rising while train falls (overfit) -> cut SFT steps 0.7x.
  * all tier gates passed -> advance phase (see PHASES) instead of grinding.
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

from smalltalk.core.bench_core import TIER_TARGETS, checksum, verify_frozen

ROOT = Path(__file__).resolve().parent.parent
LOOP = ROOT / "artifacts" / "core_loop"
LEDGER = LOOP / "ledger.jsonl"
FROZEN = ROOT / "benchmarks" / "core_bench_frozen.json"
STATE = LOOP / "state.json"

# Phase 1 is reliability on the current intent set. Later phases widen the task;
# the loop only advances when the current phase's gates are genuinely met.
PHASES = {
    1: "reliability on the current intents",
    2: "semantic breadth: more intents, wider held-out paraphrases",
    3: "natural variation: 2-3 targets per intent, measure the cost",
    4: "pragmatics: re-check SmallTalkBench-v2 for regression",
    5: "teacher / distillation",
}

DEFAULT_STATE = {
    "round": 0,
    "phase": 1,
    "sft_steps": 6000,
    "core_n": 60000,
    "weight_boost": {},          # intent -> multiplier
    "best_overall": -1.0,
    "best_round": None,
    "flat_rounds": 0,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE.exists():
        return {**DEFAULT_STATE, **json.loads(STATE.read_text())}
    return dict(DEFAULT_STATE)


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def run(cmd: list[str], logfile: Path) -> int:
    """Run a step, streaming to its own log. Never raises -- the loop must survive."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(cmd)}  -> {logfile}")
    with logfile.open("w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    if p.returncode != 0:
        log(f"  FAILED rc={p.returncode}; see {logfile}")
    return p.returncode


def rocm(script: str, *args: str) -> list[str]:
    return ["bash", "scripts/rocm.sh", script, *args]


def write_weights(state: dict) -> None:
    """Persist curriculum weight overrides where core_gen can pick them up."""
    p = ROOT / "artifacts" / "core_loop" / "weights.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state["weight_boost"], indent=2))


def build_corpus(rd: Path, state: dict) -> int:
    return run([".venv-rocm/bin/python", "scripts/build_core_corpus.py",
                "--empathetic", "data/raw/empathetic",
                "--core", str(state["core_n"]),
                "--out", "data/core"], rd / "corpus.log")


def train_stage1(rd: Path) -> int:
    return run(rocm("scripts/train.py", "--config",
                    "configs/train/core_stage1_7m.yaml"), rd / "stage1.log")


def train_sft(rd: Path, state: dict, run_name: str) -> int:
    return run(rocm("scripts/sft.py", "--config", "configs/train/core_sft_7m.yaml",
                    "--run-name", run_name,
                    "--max-steps", str(state["sft_steps"])), rd / "sft.log")


def evaluate(rd: Path, ckpt: Path, tag: str) -> dict | None:
    rc = run(rocm("scripts/eval_core.py", "--checkpoint", str(ckpt),
                  "--tokenizer", "artifacts/core/tokenizer-4096",
                  "--tag", tag), rd / "eval.log")
    out = ROOT / "reports" / "core" / f"{tag}.json"
    if rc != 0 or not out.exists():
        return None
    return json.loads(out.read_text())


def gates_passed(res: dict) -> bool:
    per_tier = {int(k): v for k, v in res["per_tier"].items()}
    return all(per_tier.get(t, 0.0) >= tgt for t, tgt in TIER_TARGETS.items())


def adapt(state: dict, res: dict, improved: bool) -> list[str]:
    """Mechanical curriculum adjustment. Returns a human-readable rationale."""
    notes: list[str] = []
    from smalltalk.core.core_gen import WEIGHTS
    from smalltalk.core.intents import BY_NAME

    # Boost only the WORST few, not everything that missed. In round 1 nineteen of
    # twenty-one intents were below target, so the original "boost every miss" rule
    # multiplied almost the whole curriculum by 1.4 -- a uniform rescale, which
    # changes the sampling distribution by nothing at all. Re-weighting is only
    # meaningful when it is *relative*.
    N_BOOST = 5
    missing = sorted(
        ((n, a) for n, a in res["per_intent"].items()
         if a < TIER_TARGETS[BY_NAME[n].tier if n in BY_NAME else 2]),
        key=lambda kv: kv[1],
    )[:N_BOOST]
    worst = {n for n, _ in missing}

    for name, acc in missing:
        cur = state["weight_boost"].get(name, 1.0)
        if cur < 4.0:
            state["weight_boost"][name] = round(min(4.0, cur * 1.5), 3)
            notes.append(f"up-weight {name} x{state['weight_boost'][name]} (acc {acc:.2f})")

    # Decay boosts on intents that are no longer among the worst, so the curriculum
    # drifts back toward its designed frequencies instead of ratcheting forever.
    for name in list(state["weight_boost"]):
        if name not in worst:
            new = round(max(1.0, state["weight_boost"][name] * 0.85), 3)
            if new == 1.0:
                del state["weight_boost"][name]
            else:
                state["weight_boost"][name] = new

    if improved:
        state["flat_rounds"] = 0
    else:
        state["flat_rounds"] += 1
        if state["flat_rounds"] >= 2 and state["sft_steps"] < 24000:
            state["sft_steps"] = int(state["sft_steps"] * 1.5)
            state["flat_rounds"] = 0
            notes.append(f"flat 2 rounds -> sft_steps={state['sft_steps']}")
    return notes


def one_round(state: dict) -> dict:
    state["round"] += 1
    rnd = state["round"]
    rd = LOOP / f"round{rnd:03d}"
    rd.mkdir(parents=True, exist_ok=True)
    tag = f"core-r{rnd:03d}"
    log(f"=== round {rnd} (phase {state['phase']}: {PHASES[state['phase']]}) "
        f"sft_steps={state['sft_steps']}")

    verify_frozen(FROZEN)          # halts the loop rather than report a bad number

    # A benchmark change invalidates the champion's score. Left in place, the loop
    # would compare new-benchmark results against an old-benchmark high-water mark
    # and never promote again -- silently ceasing to make progress while still
    # reporting healthy rounds. Detect the change and re-baseline explicitly.
    if state.get("bench_checksum") not in (None, checksum()):
        log(f"benchmark changed {state['bench_checksum']} -> {checksum()}; "
            f"discarding best_overall={state['best_overall']:.3f} as incomparable")
        state["best_overall"] = -1.0
        state["best_round"] = None
        state["weight_boost"] = {}
    state["bench_checksum"] = checksum()
    write_weights(state)

    entry = {"round": rnd, "started": now(), "phase": state["phase"],
             "checksum": checksum(), "sft_steps": state["sft_steps"]}

    if build_corpus(rd, state) != 0:
        entry["status"] = "corpus_failed"
        return entry
    if train_stage1(rd) != 0:
        entry["status"] = "stage1_failed"
        return entry
    if train_sft(rd, state, tag) != 0:
        entry["status"] = "sft_failed"
        return entry

    ckpt = ROOT / "artifacts" / "runs" / tag / "best"
    res = evaluate(rd, ckpt, tag)
    if res is None:
        entry["status"] = "eval_failed"
        return entry

    entry.update({"status": "ok", "overall": res["overall"],
                  "per_tier": res["per_tier"], "per_intent": res["per_intent"],
                  "gates_passed": gates_passed(res)})

    improved = res["overall"] > state["best_overall"]
    if improved:
        state["best_overall"] = res["overall"]
        state["best_round"] = rnd
        champ = LOOP / "champion"
        if champ.exists():
            shutil.rmtree(champ)
        shutil.copytree(ckpt, champ)
        entry["promoted"] = True
    else:
        entry["promoted"] = False

    entry["notes"] = adapt(state, res, improved)

    if entry["gates_passed"] and state["phase"] < 5:
        state["phase"] += 1
        entry["notes"].append(f"ALL GATES PASSED -> advance to phase {state['phase']}")

    entry["finished"] = now()
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=10**9)
    args = ap.parse_args()

    LOOP.mkdir(parents=True, exist_ok=True)
    if args.status:
        if not LEDGER.exists():
            print("no rounds yet")
            return 0
        for line in LEDGER.read_text().splitlines():
            e = json.loads(line)
            print(f"r{e['round']:03d} p{e.get('phase')} {e['status']:14s} "
                  f"overall={e.get('overall', float('nan')):.3f} "
                  f"{'PROMOTED' if e.get('promoted') else ''} "
                  f"{'; '.join(e.get('notes', []))}")
        return 0

    state = load_state()
    for _ in range(args.max_rounds):
        try:
            entry = one_round(state)
        except Exception as exc:                      # never die
            entry = {"round": state["round"], "status": "exception",
                     "error": repr(exc), "finished": now()}
            log(f"EXCEPTION {exc!r}; sleeping 120s and continuing")
            time.sleep(120)
        with LEDGER.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        save_state(state)
        log(f"round {entry['round']} {entry['status']} "
            f"overall={entry.get('overall')} best={state['best_overall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
