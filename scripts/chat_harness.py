#!/usr/bin/env python3
"""Interactive chat through the harness.

    python scripts/chat_harness.py --checkpoint <ckpt> --mode F_FULL_HARNESS --trace

`--mode A_RAW` is the unmodified model, for side-by-side comparison in the same
interface. `--trace` prints the full decision record for each turn.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from smalltalk.harness import Harness, MODES
from smalltalk.harness.trace import Trace
from smalltalk.infer.generate import GenerationConfig
from smalltalk.model import SmallTalkModel
from smalltalk.tokenizer import SmallTalkTokenizer


def build(checkpoint: str, tokenizer: str, mode: str, device: str) -> Harness:
    model = SmallTalkModel.from_pretrained(checkpoint, device=device).eval()
    tok = SmallTalkTokenizer.load(tokenizer)
    return Harness(model=model, tokenizer=tok, cfg=MODES[mode])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/state/tokenizer-4096")
    ap.add_argument("--mode", default="F_FULL_HARNESS", choices=list(MODES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    h = build(args.checkpoint, args.tokenizer, args.mode, args.device)
    n = sum(p.numel() for p in h.model.parameters())
    print(f"smalltalk-ai harness  ({n:,} params, {args.device})")
    print(f"mode={args.mode}  decode=GREEDY (temp 0)")
    print("commands: /reset /trace /mode <NAME> /quit\n")

    show = args.trace
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            h.reset()
            print("(reset)")
            continue
        if line == "/trace":
            show = not show
            print(f"(trace {'on' if show else 'off'})")
            continue
        if line.startswith("/mode "):
            name = line.split(None, 1)[1].strip()
            if name in MODES:
                h.cfg = MODES[name]
                h.reset()
                print(f"(mode {name}, conversation reset)")
            else:
                print(f"(unknown mode; one of {list(MODES)})")
            continue

        tr = Trace()
        out = h.reply(line, trace=tr)
        print(f"ai > {out}")
        if show:
            print("--- trace ---")
            print(tr.render())
            print("-------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
