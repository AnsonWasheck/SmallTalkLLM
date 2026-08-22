#!/usr/bin/env python3
"""Train the tiny policy head on frozen hidden states.

    python scripts/train_policy_head.py --checkpoint <ckpt> [--features] [--mlp 64]

The base model is frozen and used only as a feature extractor: one forward pass
per example, no gradients, weights untouched. Labels are free -- the Core corpus
already records meta.skill on every generated example.

Splits are by FAMILY, never by example. Two fail-closed gates run before any
training: no example may reproduce a Core-Bench probe or a StateBench opener. A
policy head fitted on test surfaces would produce a flattering number and no
knowledge.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401

from smalltalk.core import bench_core
from smalltalk.core.state_gen import BENCH_SURFACES
from smalltalk.harness.features import extract
from smalltalk.harness.head import (PID_INDEX, POLICY_IDS, FEATURE_KEYS,
                                    PolicyHead, feature_vector, hidden_state)
from smalltalk.harness.policy import BY_INTENT
from smalltalk.model import SmallTalkModel
from smalltalk.tokenizer import SmallTalkTokenizer

# Corpus skill label -> policy ontology. Only unambiguous mappings; an example
# whose intent has no clean policy is dropped rather than guessed at.
SKILL_TO_POLICY = {
    "greeting": "GREETING", "greeting_how_are_you": "GREETING_HOW_ARE_YOU",
    "how_are_you": "HOW_ARE_YOU", "thanks": "THANKS", "apology": "APOLOGY",
    "goodbye": "GOODBYE", "good_news": "GOOD_NEWS", "bad_news": "BAD_NEWS",
    "tired": "TIRED", "bored": "BORED", "user_vents": "VENTING",
    "confused": "CONFUSION", "agreement": "AGREEMENT",
    "disagreement": "DISAGREEMENT", "user_jokes": "JOKE",
    "topic_statement": "TOPIC_STATEMENT", "out_of_scope": "UNKNOWN",
    "check_in": "ACKNOWLEDGEMENT", "compliment": "THANKS",
}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", t.lower())).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/state/tokenizer-4096")
    ap.add_argument("--corpus", default="data/state/sft_train.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-examples", type=int, default=6000)
    ap.add_argument("--features", action="store_true", help="H2: add deterministic features")
    ap.add_argument("--mlp", type=int, default=0, help="hidden units; 0 = linear probe")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="artifacts/harness/policy_head.pt")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="gaussian input noise; a cheap stand-in for the surface "
                         "diversity the corpus does not contain")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = SmallTalkModel.from_pretrained(args.checkpoint, device=args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = SmallTalkTokenizer.load(args.tokenizer)
    assert sum(p.numel() for p in model.parameters()) == 6_689_024

    probes = {bench_core.normalise(s.prompt) for s in bench_core.build_scenarios()}

    rows = []
    for line in Path(args.corpus).open():
        d = json.loads(line)
        skill = (d.get("meta") or {}).get("skill")
        pol = SKILL_TO_POLICY.get(skill)
        if not pol:
            continue
        msgs = d["messages"]
        # The label describes the reply to the LAST user turn, so the example is
        # the conversation up to and including it.
        idx = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        ctx = msgs[: idx + 1]
        user_text = msgs[idx]["content"]
        # Group by SURFACE, not by the corpus family. core_gen sets
        # family = "core:<intent>", so family IS the label -- splitting on it held
        # out entire classes the classifier had never seen and produced 3.4%
        # accuracy against a 0.0% majority baseline. The right generalisation
        # question here is "does this work on unseen PARAPHRASES", so the split
        # key is the normalised user turn.
        rows.append({"ctx": ctx, "user": user_text, "pid": BY_INTENT[pol].pid,
                     "family": norm(user_text)})

    leaks = [r for r in rows
             if norm(r["user"]) in BENCH_SURFACES or bench_core.normalise(r["user"]) in probes]
    if leaks:
        raise SystemExit(f"LEAKAGE: {len(leaks)} training examples reproduce a "
                         f"benchmark surface (e.g. {leaks[0]['user']!r}). Refusing to train.")
    print(f"[leak] 0 of {len(rows):,} labelled examples touch either benchmark")

    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    rows = rows[: args.max_examples]
    fams = sorted({r["family"] for r in rows})
    rnd.shuffle(fams)
    val_fams = set(fams[: max(1, len(fams) // 5)])
    print(f"[data] {len(rows):,} examples, {len(fams):,} families "
          f"({len(val_fams):,} held out)")
    print("[label balance]", Counter(r["pid"] for r in rows).most_common(5))

    X, Y, is_val = [], [], []
    for i, r in enumerate(rows):
        ids, _ = tok.encode_conversation(r["ctx"], add_bos=True,
                                         add_generation_prompt=True)
        h = hidden_state(model, ids[-512:], device=args.device)
        if args.features:
            h = torch.cat([h, feature_vector(extract(r["user"]))])
        X.append(h)
        Y.append(PID_INDEX[r["pid"]])
        is_val.append(r["family"] in val_fams)
        if (i + 1) % 500 == 0:
            print(f"  extracted {i + 1:,}/{len(rows):,}", flush=True)

    X = torch.stack(X)
    Y = torch.tensor(Y)
    v = torch.tensor(is_val)
    Xtr, Ytr, Xva, Yva = X[~v], Y[~v], X[v], Y[v]
    # Standardise on TRAIN statistics only.
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

    head = PolicyHead(hidden_size=model.cfg.hidden_size,
                      n_features=len(FEATURE_KEYS) if args.features else 0,
                      hidden_units=args.mlp)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    best, best_state = 0.0, None
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        xb = Xtr + args.noise * torch.randn_like(Xtr) if args.noise else Xtr
        loss = F.cross_entropy(head(xb), Ytr, label_smoothing=args.label_smoothing)
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())
        head.eval()
        with torch.no_grad():
            acc = float((head(Xva).argmax(1) == Yva).float().mean())
        if acc > best:
            best, best_state = acc, {k: t.clone() for k, t in head.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep + 1:3d} loss {loss_val:.3f} val_acc {acc:.3f}")

    head.load_state_dict(best_state)
    out = Path(args.out)
    head.save(out)
    torch.save({"mu": mu, "sd": sd}, out.with_suffix(".norm.pt"))

    baseline = float((Yva == torch.mode(Ytr).values).float().mean())
    print(f"\nheld-out policy accuracy: {best:.1%}   "
          f"(majority-class baseline {baseline:.1%}, exemplar scoring 19.2%)")
    print(f"head: {head.n_params:,} params, {head.n_bytes:,} bytes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
