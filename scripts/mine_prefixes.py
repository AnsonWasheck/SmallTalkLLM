#!/usr/bin/env python3
"""Mine policy-conditioned opening tokens from TRAINING data only.

    python scripts/mine_prefixes.py --tokenizer artifacts/state/tokenizer-4096

For each policy, find the first tokens that are specific to it rather than merely
common everywhere, by conditional lift:

    lift(tok, policy) = P(tok | policy) / P(tok)

A token like "i" opens replies under every policy and carries no steering signal;
a token that is three times more likely to open a CLOSE than to open anything
does. Estimates are add-k smoothed and low-support tokens are discarded.

This is mined, never authored. The output is a statistical asset derived from the
corpus, which is what separates it from hand-written templating -- and it only
constrains the OPENING of a reply. The model generates the rest freely.

Benchmark surfaces are excluded before mining; the gate aborts on any hit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.core import bench_core
from smalltalk.core.state_gen import BENCH_SURFACES
from smalltalk.harness.policy import BY_INTENT
from smalltalk.tokenizer import SmallTalkTokenizer

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
LEN_TOKENS = re.compile(r"^<\|len_[a-z]+\|>\s*")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="artifacts/state/tokenizer-4096")
    ap.add_argument("--corpus", default="data/state/sft_train.jsonl")
    ap.add_argument("--min-support", type=int, default=25)
    ap.add_argument("--min-lift", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default="artifacts/harness/prefixes.json")
    args = ap.parse_args()

    tok = SmallTalkTokenizer.load(args.tokenizer)
    probes = {bench_core.normalise(s.prompt) for s in bench_core.build_scenarios()}

    first_by_policy: dict[str, Counter] = defaultdict(Counter)
    bigram_by_policy: dict[str, Counter] = defaultdict(Counter)
    global_first = Counter()
    n_by_policy = Counter()

    for line in Path(args.corpus).open():
        d = json.loads(line)
        pol = SKILL_TO_POLICY.get((d.get("meta") or {}).get("skill"))
        if not pol:
            continue
        msgs = d["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or i == 0:
                continue
            u = msgs[i - 1]["content"]
            if bench_core.normalise(u) in probes or \
                    re.sub(r"[^a-z0-9' ]+", " ", u.lower()).strip() in BENCH_SURFACES:
                raise SystemExit(f"LEAKAGE: benchmark surface in mining corpus: {u!r}")
            ids = tok.encode(LEN_TOKENS.sub("", m["content"]))
            if not ids:
                continue
            pid = BY_INTENT[pol].pid
            first_by_policy[pid][ids[0]] += 1
            if len(ids) > 1:
                bigram_by_policy[pid][(ids[0], ids[1])] += 1
            global_first[ids[0]] += 1
            n_by_policy[pid] += 1

    total = sum(global_first.values())
    out: dict[str, dict] = {}
    for pid, counts in first_by_policy.items():
        n = n_by_policy[pid]
        scored = []
        for tid, c in counts.items():
            if c < args.min_support:
                continue
            p_cond = (c + 1) / (n + len(counts))
            p_glob = (global_first[tid] + 1) / (total + len(global_first))
            lift = p_cond / p_glob
            if lift >= args.min_lift:
                scored.append({"token": tid, "text": tok.decode([tid]),
                               "count": c, "p_cond": round(p_cond, 5),
                               "lift": round(lift, 3)})
        scored.sort(key=lambda r: -r["lift"])
        out[pid] = {"n": n, "first": scored[: args.top_k]}

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    kept = sum(len(v["first"]) for v in out.values())
    print(f"mined {kept} policy-specific opening tokens across {len(out)} policies "
          f"-> {p} ({p.stat().st_size:,} bytes)")
    for pid in sorted(out)[:8]:
        toks = ", ".join(f"{r['text'].strip()!r}x{r['lift']:.1f}" for r in out[pid]["first"][:4])
        print(f"  {pid} n={out[pid]['n']:6,}  {toks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
