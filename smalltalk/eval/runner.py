"""Drive a model through SmallTalkBench and compute all metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from ..data.schema import load_conversations
from ..infer.generate import ConversationEngine, GenerationConfig
from .bench import Scenario, load_scenarios
from .metrics import BrokenThresholds, ConversationEval, aggregate, evaluate_transcript


def run_bench(
    engine: ConversationEngine,
    scenarios: Sequence[Scenario] | None = None,
    gen: GenerationConfig | None = None,
    thresholds: BrokenThresholds | None = None,
    seed: int = 1234,
) -> tuple[list[ConversationEval], list[dict[str, Any]]]:
    scenarios = list(scenarios or load_scenarios())
    evals: list[ConversationEval] = []
    transcripts: list[dict[str, Any]] = []

    for i, sc in enumerate(scenarios):
        cfg = (gen or engine.gen)
        cfg = cfg.with_(seed=seed + i)  # reproducible but not identical per scenario
        messages = engine.run_scenario(sc.user_turns, gen=cfg)
        evals.append(
            evaluate_transcript(
                messages, scenario_id=sc.id, category=sc.category,
                probes=sc.probes, thresholds=thresholds,
            )
        )
        transcripts.append(
            {"scenario_id": sc.id, "category": sc.category, "messages": messages}
        )
    return evals, transcripts


@torch.no_grad()
def validation_loss(
    engine: ConversationEngine,
    val_path: str | Path,
    seq_len: int = 1024,
    assistant_only: bool = True,
    max_examples: int = 512,
) -> dict[str, float]:
    """Val loss over held-out conversations, both all-token and assistant-only."""
    from ..data.dataset import SFTDataset, collate

    convs = load_conversations(val_path)[:max_examples]
    if not convs:
        return {}
    device = next(engine.model.parameters()).device
    out: dict[str, float] = {}
    for name, mask_only in (("val_loss_all", False), ("val_loss_assistant", True)):
        if mask_only and not assistant_only:
            continue
        ds = SFTDataset(convs, engine.tokenizer, seq_len, mask_non_assistant=mask_only)
        total, n = 0.0, 0
        for i in range(0, len(ds), 8):
            batch = collate([ds[j] for j in range(i, min(i + 8, len(ds)))])
            batch = {k: v.to(device) for k, v in batch.items()}
            _, loss = engine.model(
                batch["input_ids"], labels=batch["labels"], loss_mask=batch["loss_mask"]
            )
            total += float(loss)
            n += 1
        out[name] = round(total / max(n, 1), 4)
        out[name.replace("loss", "ppl")] = round(math.exp(min(out[name], 20)), 3)
    return out


def evaluate_model(
    engine: ConversationEngine,
    model_name: str,
    scenarios: Sequence[Scenario] | None = None,
    gen: GenerationConfig | None = None,
    val_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    seed: int = 1234,
) -> dict[str, Any]:
    evals, transcripts = run_bench(engine, scenarios, gen, seed=seed)
    summary: dict[str, Any] = {
        "model": model_name,
        "params": engine.model.num_parameters(),
        "generation": (gen or engine.gen).__dict__,
        "smalltalkbench": aggregate(evals),
    }
    if val_path and Path(val_path).exists():
        summary["validation"] = validation_loss(engine, val_path)

    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        with (d / "transcripts.jsonl").open("w", encoding="utf-8") as f:
            for t in transcripts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        with (d / "per_scenario.jsonl").open("w", encoding="utf-8") as f:
            for e in evals:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False, default=float) + "\n")
    return summary


def sweep_generation(
    engine: ConversationEngine,
    grid: Sequence[dict[str, Any]],
    scenarios: Sequence[Scenario] | None = None,
    seed: int = 1234,
) -> list[dict[str, Any]]:
    """Sweep decoding parameters -- they are hypotheses, not fixed truths."""
    results = []
    base = engine.gen
    for point in grid:
        cfg = base.with_(**point)
        evals, _ = run_bench(engine, scenarios, cfg, seed=seed)
        agg = aggregate(evals)
        results.append(
            {
                "params": point,
                "clean_10turn_rate": agg.get("clean_10turn_rate"),
                "clean_conversation_rate": agg.get("clean_conversation_rate"),
                "broken_turn_rate": agg.get("broken_turn_rate"),
                "loop_rate": agg.get("loop_rate"),
                "mean_len": agg.get("mean_len"),
                "distinct_2": agg.get("distinct_2"),
            }
        )
    results.sort(
        key=lambda r: (
            -(r["clean_10turn_rate"] or 0),
            r["broken_turn_rate"] if r["broken_turn_rate"] is not None else 1,
        )
    )
    return results


DEFAULT_SWEEP_GRID = [
    {"temperature": t, "top_p": p, "repetition_penalty": rp, "max_new_tokens": 48}
    for t in (0.6, 0.7, 0.8, 0.9)
    for p in (0.85, 0.9, 0.95)
    for rp in (1.0, 1.1, 1.2)
]
