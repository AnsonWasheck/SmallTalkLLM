#!/usr/bin/env python3
"""Online native-tokenizer KD: frozen 50M teacher -> immutable smalltalk-7m.

Example:
  python scripts/distill_student.py --teacher artifacts/teacher-50m/best \
    --config configs/train/distill_7m_v2.yaml --alpha 0.5 --temperature 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from smalltalk.config import TrainConfig, load_model_config
from smalltalk.data.dataset import build_dataset, collate
from smalltalk.data.schema import load_conversations
from smalltalk.model import SmallTalkModel
from smalltalk.params import analytic_param_count
from smalltalk.tokenizer import SmallTalkTokenizer
from smalltalk.train.distillation import causal_ce_and_kl
from smalltalk.train.utils import build_optimizer, resolve_device, resolve_dtype, set_seed

LOCKED_STUDENT_PARAMS = 6_689_024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", required=True, help="same-tokenizer teacher checkpoint")
    ap.add_argument("--config", default="configs/train/distill_7m_v2.yaml")
    ap.add_argument("--alpha", type=float, default=0.5, help="gold CE weight")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = TrainConfig.load(args.config)
    student_cfg = load_model_config(cfg.model_config)
    if analytic_param_count(student_cfg) != LOCKED_STUDENT_PARAMS:
        raise SystemExit("refusing KD: student is not the immutable 6,689,024-param model")
    device, dtype = resolve_device(cfg.device), resolve_dtype(cfg.dtype, resolve_device(cfg.device))
    tok = SmallTalkTokenizer.load(cfg.tokenizer)
    teacher = SmallTalkModel.from_pretrained(args.teacher, device=device).eval()
    if teacher.cfg.vocab_size != tok.vocab_size or student_cfg.vocab_size != tok.vocab_size:
        raise SystemExit("teacher, student, and tokenizer must use identical vocabulary IDs")
    for p in teacher.parameters():
        p.requires_grad_(False)
    set_seed(cfg.seed)
    student = SmallTalkModel(student_cfg).to(device).train()
    opt = build_optimizer(student, cfg.learning_rate, (cfg.beta1, cfg.beta2), cfg.weight_decay)
    ds = build_dataset("clm", load_conversations(cfg.train_data), tok, cfg.seq_len, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    out = Path(cfg.output_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    steps = args.max_steps or cfg.max_steps
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits, _ = teacher(batch["input_ids"], segment_ids=batch.get("segment_ids"))
        student_logits, _ = student(batch["input_ids"], segment_ids=batch.get("segment_ids"))
        loss, metrics = causal_ce_and_kl(student_logits, teacher_logits, batch["labels"],
                                         batch.get("loss_mask"), alpha=args.alpha,
                                         temperature=args.temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.log_every == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), **metrics}))
    student.save_pretrained(out / "final")
    (out / "distillation.json").write_text(json.dumps({
        "teacher": args.teacher, "student_params": student.num_parameters(),
        "alpha": args.alpha, "temperature": args.temperature, "steps": steps,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
