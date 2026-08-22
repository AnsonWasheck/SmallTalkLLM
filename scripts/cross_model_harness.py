#!/usr/bin/env python3
"""Is harness responsiveness a property of the harness, or of the model?

    python scripts/cross_model_harness.py

Runs A_RAW and ORACLE_POLICY over every checkpoint in the project's history.
ORACLE_POLICY is the right probe here because it needs no per-model assets: the
learned policy head and the hidden-state centroids were fitted to ONE
checkpoint's representation and do not transfer, so any mode using them would
measure asset mismatch rather than model behaviour.

The quantity of interest is HEADROOM -- oracle minus raw. A model where a correct
policy buys a lot is one whose representation already separates conversational
acts; a model where it buys nothing cannot be helped from outside, however good
the controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.harness import Harness, MODES
from smalltalk.model import SmallTalkModel
from smalltalk.tokenizer import SmallTalkTokenizer
from eval_harness import core_eval, state_eval

ROOT = Path(__file__).resolve().parent.parent

# Each generation used its own tokenizer; pairing a checkpoint with the wrong one
# produces fluent-looking nonsense, so the mapping is explicit.
CHECKPOINTS = [
    ("v0.1 real7m",      "artifacts/runs/real7m/best",     "artifacts/tokenizer-4096"),
    ("v0.2 core-r001",   "artifacts/runs/core-r001/best",  "artifacts/core/tokenizer-4096"),
    ("v0.2 core-r002",   "artifacts/runs/core-r002/best",  "artifacts/core/tokenizer-4096"),
    ("v0.2 core-r004",   "artifacts/runs/core-r004/best",  "artifacts/core/tokenizer-4096"),
    ("v0.3 state-r001",  "artifacts/runs/state-r001/best", "artifacts/state/tokenizer-4096"),
    ("v0.3 state-r002",  "artifacts/runs/state-r002/best", "artifacts/state/tokenizer-4096"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/harness/cross_model.json")
    args = ap.parse_args()

    rows = []
    print(f"{'checkpoint':20s} {'raw':>7s} {'oracle':>7s} {'headroom':>9s} "
          f"{'q_raw':>6s} {'q_orc':>6s} {'state':>6s}")
    for name, ckpt, tokdir in CHECKPOINTS:
        if not (ROOT / ckpt).exists() or not (ROOT / tokdir).exists():
            print(f"{name:20s}   (missing)")
            continue
        try:
            tok = SmallTalkTokenizer.load(ROOT / tokdir)
        except ValueError as exc:
            # Pre-v0.2 tokenizers lack the length-policy specials. The loader is
            # right to refuse them; that contract is what stops a checkpoint being
            # silently paired with an incompatible vocabulary.
            print(f"{name:20s}   (incompatible tokenizer: {exc})")
            continue
        model = SmallTalkModel.from_pretrained(ROOT / ckpt, device=args.device).eval()
        assert sum(p.numel() for p in model.parameters()) == 6_689_024

        out = {}
        for mode in ("A_RAW", "ORACLE_POLICY"):
            h = Harness(model=model, tokenizer=tok, cfg=MODES[mode])
            out[mode] = core_eval(h, args.limit)
            if mode == "A_RAW":
                h.reset()
                out["state"] = state_eval(h)
        raw = out["A_RAW"]["pass_rate"]
        orc = out["ORACLE_POLICY"]["pass_rate"]
        rows.append({"name": name, "raw": raw, "oracle": orc,
                     "headroom": orc - raw,
                     "q_raw": out["A_RAW"]["question_rate"],
                     "q_oracle": out["ORACLE_POLICY"]["question_rate"],
                     "state_directional": out["state"]["directional"]})
        print(f"{name:20s} {raw:7.3f} {orc:7.3f} {orc - raw:+9.3f} "
              f"{out['A_RAW']['question_rate']:6.2f} "
              f"{out['ORACLE_POLICY']['question_rate']:6.2f} "
              f"{out['state']['directional']:6.3f}", flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
