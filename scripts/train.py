#!/usr/bin/env python3
"""Stage 1: conversational causal-LM pretraining (loss on all tokens).

python scripts/train.py --config configs/train/stage1_4m.yaml
python scripts/train.py --config configs/train/stage1_4m.yaml --learning-rate 8e-4 --run-name lr8e4
python scripts/train.py --config configs/train/stage1_4m.yaml --sweep-lr 3e-4 5e-4 8e-4 1e-3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.config import TrainConfig
from smalltalk.train.trainer import Trainer


def add_override_args(ap: argparse.ArgumentParser) -> None:
    """Expose every TrainConfig field as --kebab-case, so nothing is hard-coded."""
    for f in fields(TrainConfig):
        if f.name == "extra":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, dest=f.name, action="store_true", default=None)
            ap.add_argument("--no-" + f.name.replace("_", "-"), dest=f.name,
                            action="store_false", default=None)
        elif f.type in ("int", int, "int | None"):
            ap.add_argument(flag, dest=f.name, type=int, default=None)
        elif f.type in ("float", float):
            ap.add_argument(flag, dest=f.name, type=float, default=None)
        else:
            ap.add_argument(flag, dest=f.name, type=str, default=None)


def build_config(args: argparse.Namespace, default_stage: str) -> TrainConfig:
    cfg = TrainConfig.load(args.config) if args.config else TrainConfig(stage=default_stage)
    for f in fields(TrainConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    if not args.config:
        cfg.stage = default_stage
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="training YAML")
    ap.add_argument("--sweep-lr", type=float, nargs="+", help="run a small LR sweep")
    add_override_args(ap)
    args = ap.parse_args()

    cfg = build_config(args, default_stage="clm")

    if args.sweep_lr:
        results = []
        base_name = cfg.run_name
        for lr in args.sweep_lr:
            cfg_lr = TrainConfig.from_dict(cfg.to_dict())
            cfg_lr.learning_rate = lr
            cfg_lr.run_name = f"{base_name}-lr{lr:g}"
            print(f"\n===== sweep lr={lr:g} =====")
            summary = Trainer(cfg_lr).train()
            results.append({"lr": lr, **summary})
        results.sort(key=lambda r: r.get("best_val_loss") or float("inf"))
        out = Path(cfg.output_dir) / f"{base_name}-lr-sweep.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=float))
        print(f"\n[sweep] best lr = {results[0]['lr']:g}  -> {out}")
        return 0

    Trainer(cfg).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
