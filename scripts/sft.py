#!/usr/bin/env python3
"""Stage 2: assistant-response SFT (loss masked to assistant tokens only).

python scripts/sft.py --config configs/train/sft_4m.yaml
python scripts/sft.py --config configs/train/sft_4m.yaml --init-from artifacts/runs/stage1-4m/best

Ablation: --no-mask-non-assistant trains on all tokens for comparison.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from smalltalk.train.trainer import Trainer
from train import add_override_args, build_config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="training YAML")
    add_override_args(ap)
    args = ap.parse_args()

    cfg = build_config(args, default_stage="sft")
    if cfg.stage not in ("sft", "distill"):
        print(f"[sft] overriding stage {cfg.stage!r} -> 'sft'")
        cfg.stage = "sft"
    if not cfg.init_from:
        print("[sft] WARNING: no --init-from; training assistant-only loss from scratch "
              "is a valid ablation but not the intended stage-2 recipe")
    Trainer(cfg).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
