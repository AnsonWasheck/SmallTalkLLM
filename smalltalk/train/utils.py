"""Device/dtype resolution, seeding, LR schedule and local logging."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def is_rocm() -> bool:
    """True when this torch build targets AMD ROCm/HIP.

    On ROCm, AMD GPUs are addressed through the *same* `cuda` device type and the
    same `torch.cuda` API -- there is no separate 'rocm' device. So all training
    code paths here work unchanged; only the installed wheel differs.
    """
    return bool(getattr(torch.version, "hip", None))


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        # 'rocm'/'hip'/'amd' are accepted aliases for the cuda device type.
        if spec.lower() in ("rocm", "hip", "amd"):
            spec = "cuda"
        return torch.device(spec)
    if torch.cuda.is_available():  # covers NVIDIA CUDA and AMD ROCm alike
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    device = device or resolve_device("auto")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        backend = f"ROCm {torch.version.hip}" if is_rocm() else f"CUDA {torch.version.cuda}"
        bf16 = "bf16" if torch.cuda.is_bf16_supported() else "fp32 only"
        return f"{name} via {backend} ({bf16})"
    if device.type == "mps":
        return "Apple MPS (fp32; bf16 autocast unsupported)"
    return f"CPU ({torch.get_num_threads()} threads)"


def resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    if spec == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[spec]


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lr_at_step(step: int, total: int, peak: float, warmup: int, min_ratio: float, schedule: str) -> float:
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if schedule == "constant":
        return peak
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    floor = peak * min_ratio
    if schedule == "linear":
        return floor + (peak - floor) * (1.0 - progress)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model, lr: float, betas: tuple[float, float], weight_decay: float):
    """No weight decay on 1-D params (norms/embeddings biases) -- standard practice."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


@dataclass
class RunLogger:
    """Local JSONL + CSV logging; WandB is strictly optional."""

    out_dir: Path
    use_wandb: bool = False
    project: str = "smalltalk-ai"
    entity: str | None = None
    run_name: str = "run"
    config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = (self.out_dir / "log.jsonl").open("a", encoding="utf-8")
        self._csv_path = self.out_dir / "log.csv"
        # Adopt the existing header so a resumed run appends instead of clobbering.
        self._csv_fields: list[str] = []
        if self._csv_path.exists():
            with self._csv_path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            self._csv_fields = list(header)
        self._start = time.time()
        self._wandb = None
        if self.use_wandb:
            try:
                import wandb

                self._wandb = wandb.init(
                    project=self.project, entity=self.entity, name=self.run_name,
                    config=self.config or {}, reinit=True,
                )
            except Exception as exc:  # pragma: no cover
                print(f"[logger] wandb unavailable ({exc}); falling back to local logs")
                self._wandb = None
        if self.config:
            (self.out_dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))

    def log(self, metrics: dict[str, Any], step: int) -> None:
        row = {"step": step, "elapsed_s": round(time.time() - self._start, 2), **metrics}
        self.jsonl.write(json.dumps(row, default=float) + "\n")
        self.jsonl.flush()
        new_fields = [k for k in row if k not in self._csv_fields]
        if new_fields:
            # Metrics appear at different cadences (val_* only on eval steps), so
            # widen the header and rewrite, keeping every column already on disk.
            existing = []
            if self._csv_path.exists():
                with self._csv_path.open(newline="", encoding="utf-8") as f:
                    existing = list(csv.DictReader(f))
            self._csv_fields += new_fields
            for r in existing:
                for k in r:
                    if k not in self._csv_fields:
                        self._csv_fields.append(k)
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_fields, restval="")
                w.writeheader()
                w.writerows(existing)
                w.writerow(row)
        else:
            with self._csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._csv_fields, restval="").writerow(row)
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self) -> None:
        self.jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()
