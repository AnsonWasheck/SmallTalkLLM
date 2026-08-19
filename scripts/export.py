#!/usr/bin/env python3
"""Export a checkpoint as a deployable artifact and report its true shipped size.

python scripts/export.py --checkpoint artifacts/runs/sft-7m/best --precision int4 \
    --out artifacts/deploy/smalltalk-7m-int4 --budget-mb 4

Reports the size of everything you actually have to ship (weights + tokenizer +
config), excluding optimizer state, and fails loudly if a stated budget is missed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

import _bootstrap  # noqa: F401

from smalltalk.model import SmallTalkModel
from smalltalk.quantize import artifact_bytes, load_quantized, save_quantized
from smalltalk.tokenizer import SmallTalkTokenizer


def mb(n: int) -> float:
    return n / 1048576


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="int4", choices=["fp32", "bf16", "int8", "int4"])
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--budget-mb", type=float, default=None)
    ap.add_argument("--verify", action="store_true", help="reload and compare logits")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = SmallTalkModel.from_pretrained(ckpt, device="cpu").eval()
    n_params = model.num_parameters()

    if args.precision in ("int8", "int4"):
        report = save_quantized(model, out, args.precision)
        print(f"[quantize] {json.dumps(report.to_dict(), indent=2)}")
    else:
        if args.precision == "bf16":
            for p in model.parameters():
                p.data = p.data.to(torch.bfloat16)
        model.save_pretrained(out)
        print(f"[export] saved {args.precision}")

    # ship the tokenizer alongside
    tok_src = Path(args.tokenizer) if args.tokenizer else None
    if tok_src is None:
        for cand in (ckpt / "tokenizer", ckpt.parent / "tokenizer", ckpt):
            if (cand / "tokenizer.json").exists():
                tok_src = cand
                break
    if tok_src and (tok_src / "tokenizer.json").exists():
        (out / "tokenizer").mkdir(exist_ok=True)
        shutil.copy(tok_src / "tokenizer.json", out / "tokenizer" / "tokenizer.json")
        tok = SmallTalkTokenizer.load(tok_src)
        print(f"[tokenizer] vocab {tok.vocab_size}")
    else:
        print("[tokenizer] WARNING: none found; artifact is NOT self-contained")

    if args.verify and args.precision in ("int8", "int4"):
        x = torch.randint(0, model.cfg.vocab_size, (1, 16))
        with torch.no_grad():
            ref, _ = model(x)
            got, _ = load_quantized(out).eval()(x)
        corr = float(torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1])
        rel = float((got - ref).abs().mean() / ref.abs().mean())
        print(f"[verify] logit correlation {corr:.4f}, mean rel error {rel:.4f}")

    total = artifact_bytes(out)
    print("\nShipped artifact")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {str(p.relative_to(out)):<32} {mb(p.stat().st_size):8.3f} MB")
    print(f"  {'TOTAL':<32} {mb(total):8.3f} MB   ({n_params:,} params, {args.precision})")
    print(f"  {'bytes per parameter':<32} {total / n_params:8.3f}")

    if args.budget_mb:
        ok = mb(total) <= args.budget_mb
        print(f"\nBudget {args.budget_mb:g} MB: {'FITS' if ok else 'OVER'} "
              f"({mb(total):.3f} MB, {'-' if ok else '+'}{abs(args.budget_mb - mb(total)):.3f} MB)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
