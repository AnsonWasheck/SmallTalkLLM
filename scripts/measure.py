#!/usr/bin/env python3
"""Behavioural measurement over MULTIPLE samples per prompt.

Single-draw taste tests are cherry-picking: temperature 0.7 makes any one reply
unrepresentative. Every number here is over N seeds, so rounds are comparable.

python scripts/measure.py --checkpoint artifacts/runs/gold04/best --samples 8

Emits JSON to stdout and (optionally) appends a ledger row. The composite
`score` is what loop.sh uses to decide whether to promote a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.eval.metrics import words
from smalltalk.infer.generate import GenerationConfig, load_engine

CHEER = re.compile(r"\b(nice|great|congrats|amazing|solid|lol|haha|cool|awesome|"
                   r"brilliant|yesss|buzzing|win)\b", re.I)
SUPPORT = re.compile(r"\b(sorry|awful|hard|heavy|holding up|anyone with you|"
                     r"take your time|take whatever time|here for|thinking of you|"
                     r"close|rush|talk about it)\b", re.I)

# Probe suites. Each is (name, user_turns, checker).
PROBES = [
    ("grief", ["hey", "my grandad passed away on tuesday"], "no_cheer"),
    ("grief2", ["hey", "my dog died this morning"], "no_cheer"),
    ("bored", ["hey", "im so bored"], "engage"),
    ("tired", ["hey", "honestly kinda tired today"], "no_cheer_soft"),
    ("goodnews", ["hey", "i got the job!!"], "celebrate"),
    ("greet", ["hey"], "short"),
    ("unknown", ["hey", "what year was the sextant invented"], "decline"),
    ("memory", ["hey im nadia", "i work as a vet nurse", "yeah long hours",
                "what do i do for work again?"], "recall_vet"),
]


def run(ckpt: str, tokenizer: str, samples: int, temperature: float,
        top_p: float, rep_pen: float, max_new: int, device: str) -> dict:
    gen = GenerationConfig(temperature=temperature, top_p=top_p,
                           repetition_penalty=rep_pen, max_new_tokens=max_new)
    eng = load_engine(ckpt, tokenizer, device=device, gen=gen)

    results: dict[str, dict] = {}
    all_replies: list[str] = []
    for name, turns, check in PROBES:
        outs = []
        for s in range(samples):
            eng.reset()
            eng.gen = eng.gen.with_(seed=s)
            reply = ""
            for u in turns:
                reply = eng.reply(u)
            outs.append(reply)
        all_replies += outs
        uniq = len(set(outs)) / len(outs)
        lens = [len(words(o)) for o in outs]
        fails = 0
        for o in outs:
            low = o.lower()
            if check == "no_cheer":
                fails += bool(CHEER.search(o)) or not SUPPORT.search(o)
            elif check == "no_cheer_soft":
                fails += bool(re.search(r"\b(congrats|amazing|so happy|brilliant|"
                                        r"you earned|buzzing)\b", low))
            elif check == "celebrate":
                fails += not re.search(r"\b(nice|congrats|amazing|great|yes|"
                                       r"happy|earned|brilliant|solid|buzzing)\b", low)
            elif check == "engage":
                fails += len(words(o)) < 3 or bool(re.search(r"\b(nice one|cool|solid)\b", low))
            elif check == "decline":
                fails += not re.search(r"\b(no idea|clue|dunno|beats me|not sure|"
                                       r"never heard|don'?t know)\b", low)
            elif check == "recall_vet":
                fails += not re.search(r"\b(vet|nurse|animal)\b", low)
            elif check == "short":
                fails += len(words(o)) > 12
        results[name] = {
            "pass_rate": round(1 - fails / len(outs), 3),
            "unique_rate": round(uniq, 3),
            "mean_words": round(sum(lens) / len(lens), 2),
            "samples": outs,
        }

    global_uniq = len(set(all_replies)) / max(len(all_replies), 1)
    mean_pass = sum(r["pass_rate"] for r in results.values()) / len(results)
    mean_uniq = sum(r["unique_rate"] for r in results.values()) / len(results)
    mean_words = sum(r["mean_words"] for r in results.values()) / len(results)
    # Composite: behaviour correctness dominates, diversity is a real second axis,
    # and we softly reward replies landing in the 3-25 word conversational band.
    band = 1.0 if 4.0 <= mean_words <= 14.0 else 0.6
    score = round(0.6 * mean_pass + 0.3 * mean_uniq + 0.1 * band, 4)
    return {
        "checkpoint": ckpt,
        "samples_per_probe": samples,
        "decoding": {"temperature": temperature, "top_p": top_p,
                     "repetition_penalty": rep_pen, "max_new_tokens": max_new},
        "mean_pass_rate": round(mean_pass, 4),
        "mean_unique_rate": round(mean_uniq, 4),
        "global_unique_rate": round(global_uniq, 4),
        "mean_words": round(mean_words, 2),
        "score": score,
        "probes": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer-4096")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    res = run(args.checkpoint, args.tokenizer, args.samples, args.temperature,
              args.top_p, args.repetition_penalty, args.max_new_tokens, args.device)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(res, indent=2))
    if args.quiet:
        print(res["score"])
    else:
        summary = {k: v for k, v in res.items() if k != "probes"}
        print(json.dumps(summary, indent=2))
        for name, r in res["probes"].items():
            print(f"  {name:<9} pass={r['pass_rate']:<5} uniq={r['unique_rate']:<5} "
                  f"words={r['mean_words']:<5} e.g. {r['samples'][0]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
