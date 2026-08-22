#!/usr/bin/env python3
"""Estimate per-policy directions in the frozen model's hidden-state space.

    python scripts/mine_centroids.py --checkpoint <ckpt>

direction[policy] = mean(h | policy) - mean(h)

Phase 2 showed a linear probe reads the policy out of this space at 76.9% on
held-out benchmark surfaces. If the policy is linearly decodable, the natural
question is whether pushing the representation along that same direction makes
the frozen LM head express it. This asset is what makes that testable.

Directions are unit-normalised; the generation-time push is scaled relative to
the hidden state's own norm, so alpha is comparable across positions rather than
depending on the absolute scale of the representation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

import _bootstrap  # noqa: F401

from smalltalk.harness.head import hidden_state
from smalltalk.harness.policy import BY_INTENT
from smalltalk.model import SmallTalkModel
from smalltalk.tokenizer import SmallTalkTokenizer
from mine_prefixes import SKILL_TO_POLICY


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/state/tokenizer-4096")
    ap.add_argument("--corpus", default="data/state/sft_train.jsonl")
    ap.add_argument("--max-examples", type=int, default=2400)
    ap.add_argument("--out", default="artifacts/harness/centroids.pt")
    args = ap.parse_args()

    model = SmallTalkModel.from_pretrained(args.checkpoint, device="cpu").eval()
    tok = SmallTalkTokenizer.load(args.tokenizer)

    rows = []
    for line in Path(args.corpus).open():
        d = json.loads(line)
        pol = SKILL_TO_POLICY.get((d.get("meta") or {}).get("skill"))
        if not pol:
            continue
        msgs = d["messages"]
        idx = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        rows.append((BY_INTENT[pol].pid, msgs[: idx + 1]))
    random.Random(0).shuffle(rows)
    rows = rows[: args.max_examples]

    sums: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(model.cfg.hidden_size))
    counts: dict[str, int] = defaultdict(int)
    total = torch.zeros(model.cfg.hidden_size)
    for i, (pid, ctx) in enumerate(rows):
        ids, _ = tok.encode_conversation(ctx, add_bos=True, add_generation_prompt=True)
        h = hidden_state(model, ids[-512:])
        sums[pid] += h
        counts[pid] += 1
        total += h
        if (i + 1) % 400 == 0:
            print(f"  {i + 1:,}/{len(rows):,}", flush=True)

    mu = total / len(rows)
    dirs = {}
    for pid, s in sums.items():
        d = s / counts[pid] - mu
        dirs[pid] = d / (d.norm() + 1e-6)     # unit direction; alpha carries scale
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dirs, out)
    print(f"\n{len(dirs)} policy directions -> {out} ({out.stat().st_size:,} bytes)")
    print("mean pairwise cosine:",
          round(float(torch.stack([torch.stack(list(dirs.values())) @ v
                                   for v in dirs.values()]).mean()), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
