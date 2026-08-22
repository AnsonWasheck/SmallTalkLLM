#!/usr/bin/env python3
"""Measure every harness-owned persistent asset against the FP32 model budget.

    python scripts/report_harness_size.py

The budget is the base model's own FP32 weight size: the harness must never cost
more bytes than the network it is assisting, or the comparison stops being
interesting. Fails closed if exceeded.

Counted: harness source, policy/ontology data, any auxiliary learned weights.
Not counted: base model weights, the tokenizer, Python, PyTorch, the OS -- the
unchanged runtime already requires those.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
PARAMS = 6_689_024
BYTES_PER_PARAM = 4
BUDGET = PARAMS * BYTES_PER_PARAM


def tally(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    harness = sorted((ROOT / "smalltalk" / "harness").glob("*.py"))
    scripts = [ROOT / "scripts" / n for n in
               ("chat_harness.py", "eval_harness.py", "report_harness_size.py",
                "compare_harness_modes.py")]
    aux = sorted((ROOT / "artifacts" / "harness").glob("*.safetensors")) \
        if (ROOT / "artifacts" / "harness").exists() else []

    src = tally(harness)
    scr = tally(scripts)
    auxb = tally(aux)
    total = src + scr + auxb

    print("SmallTalkLLM FP32 weight reference")
    print("-" * 42)
    print(f"Parameters:              {PARAMS:,}")
    print(f"Bytes/parameter:         {BYTES_PER_PARAM}")
    print(f"Weight bytes:            {BUDGET:,}")
    print(f"Weight MiB:              {BUDGET / 1024 / 1024:.4f}")
    print()
    print("Harness")
    print("-" * 42)
    print(f"Source modules ({len(harness)}):     {src:>10,}")
    print(f"Scripts ({len(scripts)}):            {scr:>10,}")
    print(f"Auxiliary weights ({len(aux)}):      {auxb:>10,}")
    print("-" * 42)
    print(f"Total harness bytes:     {total:>10,}")
    print()
    print(f"Harness/model ratio:     {total / BUDGET * 100:.3f}%")
    print(f"Remaining budget:        {BUDGET - total:,}")
    status = "PASS" if total <= BUDGET else "FAIL"
    print(f"\nSTATUS: {status}")
    if aux:
        print("\nAuxiliary learned weights:")
        for p in aux:
            print(f"  {p.name}: {p.stat().st_size:,} bytes")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
