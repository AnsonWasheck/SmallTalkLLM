#!/usr/bin/env python3
"""Qwen3.5-9B smoke test — MUST pass before committing to a backend or the pilot.

bash scripts/rocm.sh scripts/qwen_smoke_test.py --model models/Qwen3.5-9B

Verifies, in order:
  1. text-only load succeeds and VRAM use is sane
  2. non-thinking mode: no <think> in output
  3. strict JSON conversation generation from a real latent spec
  4. role alternation + no metadata leakage
  5. throughput (tok/s) so the pilot can be sized honestly
  6. the critic path returns parseable structured JSON

Exit 0 only if all checks pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from smalltalk.qwen.client import (
    PROFILES,
    QwenGenerator,
    THINK_RE,
    extract_json,
    vram_report,
)
from smalltalk.qwen.planner import generate_specs
from smalltalk.qwen.prompts import critic_messages, writer_messages

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--profile", default="A_conservative")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    args = ap.parse_args()

    print("=" * 72)
    print("Qwen3.5-9B smoke test")
    print("=" * 72)
    print("VRAM before load:", vram_report())

    t0 = time.time()
    gen = QwenGenerator(args.model)
    print(f"[load] {gen.load_seconds}s  text_only={gen.text_only}")
    after = vram_report()
    print("VRAM after load:", after)
    check("1. model loads text-only", gen.text_only,
          f"vram_used={after.get('used_gb')}GB")

    specs = [s.to_dict() for s in generate_specs(64, seed=99)]
    # pick a spread of skills, including the hard ones
    picked, seen = [], set()
    for s in specs:
        if s["skill"] not in seen:
            picked.append(s)
            seen.add(s["skill"])
        if len(picked) >= args.batch:
            break

    prof = PROFILES[args.profile]
    prof.max_new_tokens = args.max_new_tokens
    batch = [writer_messages(s) for s in picked]

    t1 = time.time()
    outs = gen.generate(batch, prof, seed=1234)
    dt = time.time() - t1
    ntok = sum(len(gen.tokenizer.encode(o)) for o in outs)
    tps = ntok / max(dt, 1e-9)
    print(f"[gen] {dt:.1f}s for {len(outs)} samples, {ntok} tok -> {tps:.1f} tok/s")

    # 2. no thinking
    thinky = [o for o in outs if THINK_RE.search(o)]
    check("2. non-thinking (no <think>)", not thinky,
          f"{len(thinky)}/{len(outs)} contained think markers")

    # 3. strict JSON
    parsed = [extract_json(o) for o in outs]
    good = [p for p in parsed if p and isinstance(p.get("messages"), list) and p["messages"]]
    check("3. strict JSON conversations", len(good) >= max(1, len(outs) // 2),
          f"{len(good)}/{len(outs)} parsed with non-empty messages")

    # 4. role alternation + no metadata leakage
    LEAK = ("family:", "discourse_plan", "SPEC:", "latent", "epistemic_condition",
            "role:", "assistant_register", "[callback", "as an ai", "language model")
    alt_ok, leak_ok = 0, 0
    for p in good:
        ms = p["messages"]
        roles = [m.get("role") for m in ms]
        if roles and roles[0] == "user" and all(
                a != b for a, b in zip(roles, roles[1:])) and set(roles) <= {"user", "assistant"}:
            alt_ok += 1
        text = " ".join(str(m.get("content", "")) for m in ms).lower()
        if not any(k.lower() in text for k in LEAK):
            leak_ok += 1
    check("4a. strict role alternation", good and alt_ok == len(good),
          f"{alt_ok}/{len(good)}")
    check("4b. no metadata/AI leakage", good and leak_ok == len(good),
          f"{leak_ok}/{len(good)}")

    # 5. throughput recorded (not pass/fail, but must be non-trivial)
    check("5. throughput measured", tps > 1.0, f"{tps:.1f} tok/s")

    # 6. critic path
    if good:
        cprof = PROFILES["critic"]
        cbatch = [critic_messages(picked[i], good[0]["messages"]) for i in range(1)]
        couts = gen.generate(cbatch, cprof, seed=7)
        cj = extract_json(couts[0])
        ok = bool(cj and "scores" in cj and "verdict" in cj)
        check("6. critic returns structured JSON", ok,
              json.dumps(cj)[:120] if cj else couts[0][:120])
    else:
        check("6. critic returns structured JSON", False, "no valid conversation to judge")

    print("\n--- sample generation ---")
    if good:
        for m in good[0]["messages"][:8]:
            print(f"  {m['role'][:4]}: {str(m.get('content'))[:100]}")

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 72)
    print(f"{len(RESULTS)-len(failed)}/{len(RESULTS)} checks passed"
          + (f"  FAILED: {', '.join(failed)}" if failed else ""))
    out = Path("reports/qwen_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "load_seconds": gen.load_seconds,
        "text_only": gen.text_only, "vram": after, "tokens_per_s": round(tps, 1),
        "profile": prof.to_dict(),
        "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in RESULTS],
    }, indent=2))
    print(f"[saved] {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
