"""One trainer for all three stages. Stage differences live entirely in the dataset
and the loss mask, which keeps the optimisation path identical across the scaling
study -- matched conditions are what make the comparison meaningful.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import ModelConfig, TrainConfig, load_model_config
from ..data.dataset import build_dataset, collate
from ..data.schema import load_conversations
from ..model import SmallTalkModel
from ..params import check_config
from ..tokenizer import SmallTalkTokenizer
from .utils import (
    RunLogger,
    build_optimizer,
    lr_at_step,
    resolve_device,
    resolve_dtype,
    set_seed,
)


@dataclass
class TrainState:
    step: int = 0
    best_val: float = float("inf")
    tokens_seen: int = 0


class Trainer:
    def __init__(self, cfg: TrainConfig, model_cfg: ModelConfig | None = None):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)

        self.model_cfg = model_cfg or load_model_config(cfg.model_config)
        self.tokenizer = SmallTalkTokenizer.load(cfg.tokenizer)
        if self.tokenizer.vocab_size != self.model_cfg.vocab_size:
            raise ValueError(
                f"tokenizer vocab {self.tokenizer.vocab_size} != model vocab "
                f"{self.model_cfg.vocab_size}; retrain the tokenizer or pick another config"
            )

        chk = check_config(self.model_cfg, empirical=False)
        if not chk.ok:
            print("[params] WARNING: config does not match its documented target")
            print(chk.report())

        if cfg.init_from:
            self.model = SmallTalkModel.from_pretrained(cfg.init_from, device="cpu")
            print(f"[init] warm-started from {cfg.init_from}")
        else:
            self.model = SmallTalkModel(self.model_cfg)
        self.model.to(self.device)
        self.n_params = self.model.num_parameters()

        self.opt = build_optimizer(
            self.model, cfg.learning_rate, (cfg.beta1, cfg.beta2), cfg.weight_decay
        )
        self.state = TrainState()

        self.out_dir = Path(cfg.output_dir) / cfg.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = RunLogger(
            self.out_dir,
            use_wandb=cfg.wandb,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=cfg.run_name,
            config={
                "train": cfg.to_dict(),
                "model": self.model_cfg.to_dict(),
                "params": self.n_params,
                "device": str(self.device),
                "dtype": str(self.dtype),
            },
        )

        self.train_loader, self.val_loader = self._loaders()
        if cfg.resume:
            self._load_checkpoint(cfg.resume)
        if cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]

    # ---- data --------------------------------------------------------------
    def _loaders(self) -> tuple[DataLoader, DataLoader | None]:
        cfg = self.cfg
        train_convs = load_conversations(cfg.train_data)
        train_ds = build_dataset(
            cfg.stage, train_convs, self.tokenizer, cfg.seq_len,
            mask_non_assistant=cfg.mask_non_assistant, seed=cfg.seed,
        )
        g = torch.Generator().manual_seed(cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
            collate_fn=collate, generator=g, num_workers=cfg.num_workers,
        )
        val_loader = None
        val_path = Path(cfg.val_data) if cfg.val_data else None
        if val_path and val_path.exists():
            val_convs = load_conversations(val_path)
            try:
                val_ds = build_dataset(
                    cfg.stage, val_convs, self.tokenizer, cfg.seq_len,
                    mask_non_assistant=cfg.mask_non_assistant, seed=cfg.seed,
                )
                val_loader = DataLoader(
                    val_ds, batch_size=cfg.batch_size, shuffle=False,
                    drop_last=False, collate_fn=collate,
                )
            except ValueError as exc:
                print(f"[data] no validation set usable: {exc}")
        print(
            f"[data] stage={cfg.stage} train examples={len(train_ds):,} "
            f"val={len(val_loader.dataset) if val_loader else 0:,} params={self.n_params:,}"
        )
        return train_loader, val_loader

    # ---- steps -------------------------------------------------------------
    def _forward(self, batch: dict[str, torch.Tensor]):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        autocast = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.dtype in (torch.bfloat16, torch.float16) and self.device.type in ("cuda", "cpu")
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast:
            _, loss_sum = self.model(
                batch["input_ids"], labels=batch["labels"], loss_mask=batch.get("loss_mask"),
                segment_ids=batch.get("segment_ids"), reduction="sum"
            )
        labels = batch["labels"][:, 1:]
        mask = batch.get("loss_mask")
        if mask is None:
            n_tokens = (labels != -100).sum()
        else:
            n_tokens = ((labels != -100) & mask[:, 1:].bool()).sum()
        return loss_sum, n_tokens.clamp(min=1)

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> float | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if max_batches and i >= max_batches:
                break
            loss_sum, n_tokens = self._forward(batch)
            total_loss += loss_sum.detach().item()
            total_tokens += n_tokens.detach().item()
        self.model.train()
        return total_loss / max(total_tokens, 1)

    def train(self) -> dict[str, float]:
        cfg = self.cfg
        total = cfg.max_steps
        warmup = cfg.warmup_steps if cfg.warmup_steps is not None else max(1, int(total * cfg.warmup_ratio))
        self.model.train()
        it = iter(self.train_loader)
        t0 = time.time()
        running = 0.0

        while self.state.step < total:
            lr = lr_at_step(
                self.state.step, total, cfg.learning_rate, warmup, cfg.min_lr_ratio, cfg.lr_schedule
            )
            for group in self.opt.param_groups:
                group["lr"] = lr

            self.opt.zero_grad(set_to_none=True)
            microbatches = []
            for _ in range(cfg.grad_accum_steps):
                try:
                    microbatches.append(next(it))
                except StopIteration:
                    it = iter(self.train_loader)
                    microbatches.append(next(it))
            # One denominator for the whole accumulation window makes every
            # supervised token contribute equally, including a short final batch.
            expected_tokens = sum(
                int(((b["labels"][:, 1:] != -100) & b.get("loss_mask", torch.ones_like(b["labels"]))[:, 1:].bool()).sum())
                for b in microbatches
            )
            expected_tokens = max(expected_tokens, 1)
            accum_loss_sum = 0.0
            for batch in microbatches:
                loss_sum, _ = self._forward(batch)
                (loss_sum / expected_tokens).backward()
                accum_loss_sum += loss_sum.detach().item()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.state.step += 1
            self.state.tokens_seen += expected_tokens
            running += accum_loss_sum / expected_tokens

            if self.state.step % cfg.log_every == 0:
                avg = running / cfg.log_every
                running = 0.0
                elapsed = time.time() - t0
                self.logger.log(
                    {
                        "train_loss": round(avg, 4),
                        "train_ppl": round(math.exp(min(avg, 20)), 3),
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "tokens": self.state.tokens_seen,
                        "tokens_per_s": round(self.state.tokens_seen / max(elapsed, 1e-6)),
                    },
                    step=self.state.step,
                )
                print(
                    f"step {self.state.step:>6}/{total} loss {avg:.4f} "
                    f"ppl {math.exp(min(avg, 20)):.2f} lr {lr:.2e}"
                )

            if cfg.eval_every and self.state.step % cfg.eval_every == 0:
                val = self.evaluate(cfg.eval_batches)
                if val is not None:
                    self.logger.log(
                        {"val_loss": round(val, 4), "val_ppl": round(math.exp(min(val, 20)), 3)},
                        step=self.state.step,
                    )
                    print(f"  eval  step {self.state.step} val_loss {val:.4f}")
                    if val < self.state.best_val:
                        self.state.best_val = val
                        self._save_checkpoint("best")

            if cfg.save_every and self.state.step % cfg.save_every == 0:
                self._save_checkpoint(f"step-{self.state.step}")

        final_val = self.evaluate(cfg.eval_batches)
        self._save_checkpoint("final")
        summary = {
            "params": self.n_params,
            "steps": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "final_val_loss": final_val,
            "best_val_loss": None if self.state.best_val == float("inf") else self.state.best_val,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        self.logger.close()
        print(f"[done] {json.dumps(summary, default=float)}")
        return summary

    # ---- checkpoints -------------------------------------------------------
    def _save_checkpoint(self, tag: str) -> Path:
        path = self.out_dir / tag
        model = getattr(self.model, "_orig_mod", self.model)
        model.save_pretrained(path)
        torch.save(
            {
                "optimizer": self.opt.state_dict(),
                "step": self.state.step,
                "best_val": self.state.best_val,
                "tokens_seen": self.state.tokens_seen,
                "train_config": self.cfg.to_dict(),
            },
            path / "trainer_state.pt",
        )
        # Ship the tokenizer alongside so a checkpoint is self-contained.
        tok_src = Path(self.cfg.tokenizer)
        tok_dst = path / "tokenizer"
        src_file = tok_src / "tokenizer.json" if tok_src.is_dir() else tok_src
        if src_file.exists():
            tok_dst.mkdir(exist_ok=True)
            shutil.copy(src_file, tok_dst / "tokenizer.json")
        self._prune_checkpoints()
        return path

    def _prune_checkpoints(self) -> None:
        keep = self.cfg.keep_last
        if keep <= 0:
            return
        steps = sorted(
            (p for p in self.out_dir.glob("step-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[1]),
        )
        for p in steps[:-keep]:
            shutil.rmtree(p, ignore_errors=True)

    def _load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        model = SmallTalkModel.from_pretrained(path, device="cpu")
        self.model.load_state_dict(model.state_dict())
        self.model.to(self.device)
        st = path / "trainer_state.pt"
        if st.exists():
            blob = torch.load(st, map_location="cpu", weights_only=False)
            self.opt.load_state_dict(blob["optimizer"])
            self.state.step = blob.get("step", 0)
            self.state.best_val = blob.get("best_val", float("inf"))
            self.state.tokens_seen = blob.get("tokens_seen", 0)
            print(f"[resume] step {self.state.step} from {path}")
