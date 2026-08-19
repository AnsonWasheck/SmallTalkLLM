#!/usr/bin/env python3
"""Build the scaling curves and answer the eight research questions from evidence.

python scripts/scaling_report.py --eval-dir artifacts/eval --out docs/RESULTS.md

Reads the per-model summaries written by evaluate.py plus each run's summary.json,
then emits a markdown report with ASCII scaling curves (no matplotlib dependency)
and an explicit, thresholded answer to each research question. Where the evidence
is insufficient it says so rather than inventing a number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Emergence thresholds. These are the operational definitions used throughout;
# change them here and every reported answer changes consistently.
THRESHOLDS = {
    "grammatical_english": ("grammatical_rate", 0.90),
    "context_sensitive": ("probe_pass_rate", 0.60),
    "five_turn_reliable": ("clean_5turn_rate", 0.80),
    "ten_turn_reliable": ("clean_10turn_rate", 0.80),
    "human_judged_conversational": ("human_likeness", 3.5),
}


def load_summaries(eval_dir: Path) -> list[dict[str, Any]]:
    combined = eval_dir / "all_models.json"
    if combined.exists():
        return json.loads(combined.read_text())
    out = []
    for p in sorted(eval_dir.glob("*/summary.json")):
        out.append(json.loads(p.read_text()))
    return out


def sparkline(values: list[float], labels: list[str], width: int = 46) -> list[str]:
    if not values:
        return ["  (no data)"]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    rows = []
    for lab, v in zip(labels, values):
        n = int(round((v - lo) / span * width))
        rows.append(f"  {lab:>14}  {'#' * n:<{width}} {v:.4g}")
    return rows


def first_crossing(rows: list[dict[str, Any]], metric: str, threshold: float) -> dict | None:
    """Smallest model (by params) that reaches the threshold and is not undercut
    by a larger one -- i.e. the cliff edge, monotonicity permitting."""
    ordered = sorted(rows, key=lambda r: r["params"])
    for i, r in enumerate(ordered):
        v = r.get(metric)
        if v is None or v < threshold:
            continue
        if all((o.get(metric) or 0) >= threshold for o in ordered[i:]):
            return r
    return None


def flatten(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for s in summaries:
        b = s.get("smalltalkbench", {})
        v = s.get("validation", {})
        j = s.get("judge", {})
        rows.append(
            {
                "model": s.get("model", "?"),
                "params": s.get("params", 0),
                "val_loss": v.get("val_loss_assistant") or v.get("val_loss_all"),
                "val_ppl": v.get("val_ppl_assistant") or v.get("val_ppl_all"),
                "grammatical_rate": b.get("grammatical_rate"),
                "probe_pass_rate": b.get("probe_pass_rate"),
                "clean_5turn_rate": b.get("clean_5turn_rate"),
                "clean_10turn_rate": b.get("clean_10turn_rate"),
                "loop_rate": b.get("loop_rate"),
                "broken_turn_rate": b.get("broken_turn_rate"),
                "distinct_2": b.get("distinct_2"),
                "mean_len": b.get("mean_len"),
                "human_likeness": j.get("human_likeness"),
                "naturalness": j.get("naturalness"),
            }
        )
    return sorted(rows, key=lambda r: r["params"])


def fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def answer(rows, key: str) -> str:
    metric, thr = THRESHOLDS[key]
    have = [r for r in rows if r.get(metric) is not None]
    if not have:
        return f"**Not yet measured** (no `{metric}` values in the eval directory)."
    hit = first_crossing(have, metric, thr)
    if hit is None:
        best = max(have, key=lambda r: r[metric])
        return (
            f"**No model reached the threshold** (`{metric}` >= {thr}). "
            f"Best observed: {best['model']} at {best['params']:,} params "
            f"with {metric} = {fmt(best[metric])}. The cliff is above the largest "
            f"model tested, or the recipe/data is the binding constraint."
        )
    return (
        f"**~{hit['params'] / 1e6:.2f}M params** ({hit['model']}), the smallest model with "
        f"`{metric}` >= {thr} (observed {fmt(hit[metric])}) that is not undercut by any "
        f"larger model."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default="artifacts/eval")
    ap.add_argument("--out", default="docs/RESULTS.md")
    ap.add_argument("--ablations", default=None,
                    help="JSON file: {label: {params, clean_10turn_rate, ...}} for RQ5-7")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    summaries = load_summaries(eval_dir)
    if not summaries:
        print(f"no evaluation summaries under {eval_dir}; run scripts/evaluate.py first")
        return 1
    rows = flatten(summaries)
    ablations = json.loads(Path(args.ablations).read_text()) if args.ablations else {}

    L = []
    L.append("# smalltalk-ai — scaling results\n")
    L.append(f"Generated from `{eval_dir}` over {len(rows)} model(s).\n")
    L.append("Primary research metric: **clean_10turn_rate** — the fraction of "
             "held-out SmallTalkBench scenarios completed for 10 turns with zero "
             "obviously broken, repetitive or contextually nonsensical replies.\n")

    L.append("## Results table\n")
    hdr = ["model", "params", "val_ppl", "gram", "probe", "clean5", "clean10",
           "loop", "broken", "dist2", "len", "human"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for r in rows:
        L.append("| " + " | ".join([
            str(r["model"]), f"{r['params']:,}", fmt(r["val_ppl"]),
            fmt(r["grammatical_rate"]), fmt(r["probe_pass_rate"]),
            fmt(r["clean_5turn_rate"]), fmt(r["clean_10turn_rate"]),
            fmt(r["loop_rate"]), fmt(r["broken_turn_rate"]),
            fmt(r["distinct_2"]), fmt(r["mean_len"]), fmt(r["human_likeness"]),
        ]) + " |")
    L.append("")

    labels = [f"{r['params'] / 1e6:.2f}M" for r in rows]
    for metric, title in (
        ("val_loss", "Validation loss (assistant tokens) vs parameters"),
        ("clean_10turn_rate", "PRIMARY: 10-turn clean-conversation rate vs parameters"),
        ("clean_5turn_rate", "5-turn clean-conversation rate vs parameters"),
        ("grammatical_rate", "Grammaticality vs parameters"),
        ("human_likeness", "LLM/human-judged human-likeness (1-5) vs parameters"),
    ):
        vals = [r[metric] for r in rows]
        if all(v is None for v in vals):
            continue
        L.append(f"### {title}\n```")
        L += sparkline([v or 0.0 for v in vals], labels)
        L.append("```\n")

    # capability cliff
    L.append("## Capability cliff\n")
    have = [r for r in rows if r.get("clean_10turn_rate") is not None]
    if len(have) >= 2:
        worst_drop, edge = 0.0, None
        for a, b in zip(have, have[1:]):
            drop = (b["clean_10turn_rate"] or 0) - (a["clean_10turn_rate"] or 0)
            if drop > worst_drop:
                worst_drop, edge = drop, (a, b)
        if edge:
            a, b = edge
            L.append(
                f"Largest single-step gain in the primary metric occurs between "
                f"**{a['params']:,}** ({fmt(a['clean_10turn_rate'])}) and "
                f"**{b['params']:,}** ({fmt(b['clean_10turn_rate'])}) params: "
                f"+{worst_drop:.3f}. Conversational coherence collapses below "
                f"~{b['params'] / 1e6:.2f}M under this data and recipe.\n")
    else:
        L.append("Need at least two evaluated models to locate the cliff.\n")

    L.append("## Research questions\n")
    qs = [
        ("1. At what parameter count does grammatical English emerge?", "grammatical_english"),
        ("2. At what parameter count does context-sensitive response generation emerge?", "context_sensitive"),
        ("3. At what parameter count does five-turn conversation become reliable?", "five_turn_reliable"),
        ("4. At what parameter count does ten-turn conversation become reliable?", "ten_turn_reliable"),
    ]
    for q, key in qs:
        metric, thr = THRESHOLDS[key]
        L.append(f"**{q}**\n\n{answer(rows, key)}\n\n*Criterion: `{metric}` >= {thr}.*\n")

    for i, (q, label_a, label_b) in enumerate([
        ("5. How much does conversationally focused training outperform generic LM training "
         "at equal parameter count?", "conversational", "generic"),
        ("6. How much does reducing vocabulary size help?", "vocab_4096", "vocab_6144"),
        ("7. How much does teacher-generated conversational distillation help?", "distill", "sft_only"),
    ], start=5):
        a, b = ablations.get(label_a), ablations.get(label_b)
        if a and b:
            da = (a.get("clean_10turn_rate") or 0) - (b.get("clean_10turn_rate") or 0)
            L.append(f"**{q}**\n\n`{label_a}` = {fmt(a.get('clean_10turn_rate'))} vs "
                     f"`{label_b}` = {fmt(b.get('clean_10turn_rate'))} on the primary metric "
                     f"(delta {da:+.3f}); val_ppl {fmt(a.get('val_ppl'))} vs "
                     f"{fmt(b.get('val_ppl'))}.\n")
        else:
            L.append(f"**{q}**\n\n**Not yet measured.** Run the matched ablation and pass it via "
                     f"`--ablations` with keys `{label_a}` and `{label_b}`. "
                     f"See README 'Ablations' for the exact commands.\n")

    L.append("**8. What is the smallest model that humans consistently judge as conversational "
             "rather than obviously broken?**\n")
    L.append(answer(rows, "human_judged_conversational") + "\n")
    L.append("*This is the headline claim and requires human pairwise votes "
             "(`evaluate.py --pairwise`) or an LLM judge (`--judge-out`), not automatic "
             "metrics alone. Automatic `clean_10turn_rate` is the screening proxy.*\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    (out.with_suffix(".json")).write_text(json.dumps(
        {"rows": rows, "thresholds": THRESHOLDS, "ablations": ablations}, indent=2, default=float))
    print("\n".join(L))
    print(f"\n[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
