#!/usr/bin/env python3
"""Interactive terminal chat (and `--web` for the minimal browser demo).

python scripts/chat.py --checkpoint artifacts/runs/sft-4m/best
python scripts/chat.py --checkpoint ... --temperature 0.8 --top-p 0.9 --state
python scripts/chat.py --checkpoint ... --web --port 8000

In-chat commands: /reset /state /params /set k=v /history /quit
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from smalltalk.infer.generate import GenerationConfig, load_engine

BANNER = """\
smalltalk-ai  ({params:,} params, {device})
decode={decode} top_p={top_p} top_k={top_k} rep_pen={repetition_penalty} max_new={max_new_tokens}
commands: /reset /state /params /set temperature=0.8 /history /quit
"""

TUNABLE = {
    "temperature": float, "top_p": float, "top_k": int,
    "repetition_penalty": float, "presence_penalty": float,
    "max_new_tokens": int, "min_new_tokens": int, "max_context": int,
    "no_repeat_ngram_size": int, "greedy": lambda v: v.lower() in ("1", "true", "yes"),
}


def repl(engine) -> None:
    g = engine.gen
    # --greedy takes argmax regardless of the temperature value, so printing the
    # unused temperature made a greedy session look like it was sampling at 0.7.
    decode = "GREEDY (temp 0)" if (g.greedy or g.temperature <= 0) else f"temp={g.temperature}"
    print(BANNER.format(
        decode=decode,
        params=engine.model.num_parameters(),
        device=next(engine.model.parameters()).device,
        **{k: getattr(g, k) for k in
           ("top_p", "top_k", "repetition_penalty", "max_new_tokens")},
    ))
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/reset":
            engine.reset()
            print("[context cleared]")
            continue
        if line == "/state":
            print(engine.state.to_dict() if engine.state else "[state disabled]")
            continue
        if line == "/params":
            print(engine.gen.__dict__)
            continue
        if line == "/history":
            for m in engine.history:
                print(f"  {m['role']:>9}: {m['content']}")
            continue
        if line.startswith("/set "):
            try:
                k, v = line[5:].split("=", 1)
                k = k.strip()
                if k not in TUNABLE:
                    print(f"[unknown param {k}; try {', '.join(TUNABLE)}]")
                    continue
                engine.gen = engine.gen.with_(**{k: TUNABLE[k](v.strip())})
                print(f"[{k} = {getattr(engine.gen, k)}]")
            except ValueError:
                print("[usage: /set temperature=0.8]")
            continue
        print(f"ai > {engine.reply(line)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-context", type=int, default=1024)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--state", action="store_true",
                    help="enable the optional zero-parameter conversation state")
    ap.add_argument("--web", action="store_true", help="serve the minimal web demo instead")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    gen = GenerationConfig(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens,
        max_context=args.max_context, greedy=args.greedy, seed=args.seed,
    )
    engine = load_engine(args.checkpoint, args.tokenizer, device=args.device,
                         gen=gen, use_state=args.state)

    if args.web:
        from smalltalk.infer.web import serve

        serve(engine, port=args.port)
    else:
        repl(engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
