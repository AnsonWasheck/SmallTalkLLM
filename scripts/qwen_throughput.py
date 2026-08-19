#!/usr/bin/env python3
"""Batch-scaling + JSON-failure diagnostic. One model load, both answers.

The smoke test measured 13.6 tok/s at batch=4. If that is the ceiling, a 5M-token
pilot is ~6 days and the whole plan is infeasible; if throughput scales with batch
(decoder-only generation is memory-bandwidth bound at small batch), it is an overnight
job. This measures it rather than guessing, and simultaneously captures WHY samples
fail to parse as JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.qwen.client import PROFILES, QwenGenerator, THINK_RE, extract_json, vram_report
from smalltalk.qwen.planner import generate_specs
from smalltalk.qwen.prompts import writer_messages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 24, 48])
    ap.add_argument("--max-new-tokens", type=int, default=1100)
    args = ap.parse_args()

    gen = QwenGenerator(args.model)
    print(f"[load] {gen.load_seconds}s | {vram_report()}")

    specs = [s.to_dict() for s in generate_specs(400, seed=5)]
    prof = PROFILES["A_conservative"]
    prof.max_new_tokens = args.max_new_tokens

    rows = []
    diag = {"truncated": 0, "no_json": 0, "bad_shape": 0, "ok": 0, "think": 0}
    samples_bad = []

    for bs in args.batches:
        batch = [writer_messages(s) for s in specs[:bs]]
        t0 = time.time()
        outs = gen.generate(batch, prof, seed=11)
        dt = time.time() - t0
        ntok = sum(len(gen.tokenizer.encode(o)) for o in outs)
        tps = ntok / max(dt, 1e-9)
        conv_per_hr = (bs / dt) * 3600
        rows.append({"batch": bs, "seconds": round(dt, 1), "gen_tokens": ntok,
                     "tok_per_s": round(tps, 1),
                     "conversations_per_hour": round(conv_per_hr)})
        print(f"  batch={bs:>3}  {dt:6.1f}s  {ntok:>6} tok  {tps:7.1f} tok/s  "
              f"{conv_per_hr:8.0f} conv/hr  vram={vram_report()['used_gb']}GB")

        for o in outs:
            if THINK_RE.search(o):
                diag["think"] += 1
            j = extract_json(o)
            if j is None:
                # distinguish truncation (unbalanced braces) from no-JSON-at-all
                if o.count("{") > o.count("}"):
                    diag["truncated"] += 1
                else:
                    diag["no_json"] += 1
                if len(samples_bad) < 3:
                    samples_bad.append(o[-300:])
            elif not (isinstance(j.get("messages"), list) and j["messages"]):
                diag["bad_shape"] += 1
                if len(samples_bad) < 3:
                    samples_bad.append(json.dumps(j)[:300])
            else:
                diag["ok"] += 1

    total = sum(diag[k] for k in ("truncated", "no_json", "bad_shape", "ok"))
    print(f"\nJSON diagnosis over {total} samples: {diag}")
    print(f"  parse rate: {diag['ok']/max(total,1):.1%}")
    for s in samples_bad:
        print(f"  --- bad tail ---\n  {s!r}\n")

    best = max(rows, key=lambda r: r["tok_per_s"])
    print(f"\nbest: batch={best['batch']} at {best['tok_per_s']} tok/s "
          f"({best['conversations_per_hour']} conv/hr)")
    # 5M student tokens, ~180 student tok/conversation, /acceptance
    for acc in (0.5, 0.7):
        convs_needed = 5_000_000 / 180 / acc
        hrs = convs_needed / max(best["conversations_per_hour"], 1)
        print(f"  5M-token pilot @ {acc:.0%} acceptance: "
              f"{convs_needed:,.0f} conversations -> {hrs:.1f} h")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/qwen_throughput.json").write_text(json.dumps(
        {"rows": rows, "json_diagnosis": diag,
         "max_new_tokens": args.max_new_tokens}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
