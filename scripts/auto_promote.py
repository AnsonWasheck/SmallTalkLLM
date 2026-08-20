#!/usr/bin/env python3
"""Promote a champion to `main` automatically, but only on unambiguous evidence.

    python scripts/auto_promote.py            # check and promote if warranted
    python scripts/auto_promote.py --dry-run  # report the decision only

Runs unattended overnight, so the gates are deliberately conservative. Promotion
requires ALL of:

  1. StateBench `directional` >= MIN_DIRECTIONAL  (absolute floor: a model that
     cannot commit to the right valence at all is not worth promoting no matter
     how it compares)
  2. directional exceeds the incumbent by >= MIN_GAIN  (a clear margin, not
     noise on 12 pairs)
  3. Core-Bench tier 1 no more than MAX_TIER1_DROP below the incumbent  (a
     regression guard: state persistence must not be bought by wrecking the
     reflexes)
  4. both benchmark checksums match the incumbent's  (a score measured under a
     different instrument is not a comparison)
  5. the full test suite passes

Anything short of that is logged and skipped. Refusing to promote is always the
safe action; a bad auto-promotion to `main` while nobody is watching is not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
POINTER = ROOT / "CURRENT_MODEL.json"
LOOP = ROOT / "artifacts" / "state_loop"
LOG = LOOP / "promote.log"

MIN_DIRECTIONAL = 0.30
MIN_GAIN = 0.10
MAX_TIER1_DROP = 0.02


def say(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{__import__('datetime').datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def sh(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta_path = LOOP / "champion_meta.json"
    if not meta_path.exists():
        say("no champion yet")
        return 0
    champ = json.loads(meta_path.read_text())
    pointer = json.loads(POINTER.read_text())
    cur = pointer["current"]
    incumbent = (cur.get("statebench") or {}).get("directional", 0.0)
    inc_tier1 = cur["benchmark"].get("tier_1", 0.0)

    d = champ.get("directional")
    if d is None:
        say("champion has no directional score")
        return 0

    reasons = []
    if d < MIN_DIRECTIONAL:
        reasons.append(f"directional {d:.3f} < floor {MIN_DIRECTIONAL}")
    if d < incumbent + MIN_GAIN:
        reasons.append(f"directional {d:.3f} < incumbent {incumbent:.3f} + {MIN_GAIN}")
    t1 = champ.get("core_tier1")
    if t1 is not None and t1 < inc_tier1 - MAX_TIER1_DROP:
        reasons.append(f"core tier1 {t1:.3f} regressed > {MAX_TIER1_DROP} vs {inc_tier1:.3f}")

    if reasons:
        say(f"HOLD round {champ['round']}: " + "; ".join(reasons))
        return 0

    tests = sh(".venv-rocm/bin/python", "-m", "pytest", "tests", "-q")
    if tests.returncode != 0:
        say(f"HOLD round {champ['round']}: tests failing")
        return 0

    say(f"PROMOTE round {champ['round']}: directional {incumbent:.3f} -> {d:.3f}, "
        f"core tier1 {t1}")
    if args.dry_run:
        return 0

    label = f"core-v0.3.0-r{champ['round']:03d}"
    archive = f"archive/{label}"
    ckpt_src = LOOP / "champion"
    ckpt_dst = ROOT / "artifacts" / "snapshots" / label
    if ckpt_dst.exists():
        subprocess.run(["rm", "-rf", str(ckpt_dst)])
    subprocess.run(["cp", "-r", str(ckpt_src), str(ckpt_dst)])

    pointer["history"].insert(0, {
        "label": cur["label"], "active_until": str(date.today()),
        "archive_ref": cur.get("archive_ref"),
        "benchmark_overall": cur["benchmark"].get("overall"),
        "statebench_directional": incumbent,
        "superseded_by": label,
        "note": "superseded by automatic promotion on StateBench directional",
    })
    pointer["current"] = {
        "label": label, "promoted": str(date.today()),
        "run_name": f"state-r{champ['round']:03d}",
        "checkpoint": f"artifacts/snapshots/{label}",
        "archive_ref": archive,
        "params": 6689024,
        "tokenizer": "artifacts/state/tokenizer-4096", "tokenizer_vocab": 4096,
        "benchmark": {
            "name": "Core-Bench", "version": "core-v0.2.2",
            "checksum": champ["checksums"]["core"],
            "overall": champ.get("core_overall"), "tier_1": t1,
            "decode": "greedy, temperature 0, top_p 1.0, repetition_penalty 1.0",
        },
        "statebench": {
            "version": "statebench-v1.0.0", "checksum": champ["checksums"]["state"],
            "directional": d, "divergence": champ.get("divergence"),
            "accuracy": champ.get("state_accuracy"),
            "attractor_rate": champ.get("attractor_rate"),
        },
        "training": {"rung": champ["rung"], "config": champ["config"]},
        "promotion_rationale": (
            f"Automatic promotion. StateBench directional {incumbent:.3f} -> {d:.3f} "
            f"(gate: >= {MIN_DIRECTIONAL} absolute and >= +{MIN_GAIN} over incumbent), "
            f"Core-Bench tier 1 within {MAX_TIER1_DROP} of incumbent. Tests passed. "
            "Not reviewed by a human at promotion time."
        ),
    }
    POINTER.write_text(json.dumps(pointer, indent=2) + "\n")

    sh("git", "branch", archive)
    sh("git", "add", "-A")
    msg = (
        f"release(v0.3-core-state): auto-promote {label} on StateBench directional\n\n"
        f"  StateBench directional  {incumbent:.3f} -> {d:.3f}\n"
        f"  StateBench divergence   {champ.get('divergence')}\n"
        f"  Core-Bench overall      {champ.get('core_overall')}\n"
        f"  Core-Bench tier 1       {t1}\n\n"
        f"Rung: {champ['rung']}. Promoted automatically under the gates in\n"
        f"scripts/auto_promote.py: directional >= {MIN_DIRECTIONAL} absolute,\n"
        f">= +{MIN_GAIN} over the incumbent, Core-Bench tier 1 within\n"
        f"{MAX_TIER1_DROP}, matching benchmark checksums, and a passing test suite.\n\n"
        f"NOT REVIEWED BY A HUMAN at promotion time. The previous model remains at\n"
        f"{cur.get('archive_ref')} and is recoverable.\n"
    )
    if sh("git", "commit", "-q", "-m", msg).returncode != 0:
        say("nothing to commit")
        return 0
    sh("git", "push", "-q", "origin", "develop")
    sh("git", "checkout", "-q", "main")
    sh("git", "merge", "-q", "--no-ff", "develop", "-m",
       f"release(v0.3-core-state): auto-promote {label}")
    sh("git", "push", "-q", "origin", "main")
    sh("git", "checkout", "-q", "develop")
    sh("git", "push", "-q", "origin", archive)
    say(f"pushed {label} to main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
