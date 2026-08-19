"""Model / run configuration.

Design choice: plain dataclasses + YAML files, no external config framework.
Every experimental knob lives in `configs/` so no model size is hard-coded in code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def _load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


class _FromDictMixin:
    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
        return cls(**data)  # type: ignore[call-arg]

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(_load_mapping(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]


@dataclass
class ModelConfig(_FromDictMixin):
    """Llama-compatible decoder-only causal transformer."""

    name: str = "smalltalk-4m"
    vocab_size: int = 4096
    hidden_size: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 64
    intermediate_size: int = 704
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    initializer_range: float = 0.02
    # Documented target from the research plan; verified by smalltalk.params.
    expected_params: int | None = None
    param_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"({self.num_attention_heads} % {self.num_key_value_heads})"
            )
        if not self.tie_word_embeddings:
            # Untied embeddings double vocab cost; allowed but flagged loudly.
            pass

    @property
    def num_query_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass
class TrainConfig(_FromDictMixin):
    """Hyperparameters shared by stage 1 (CLM), stage 2 (SFT) and stage 3 (distill)."""

    run_name: str = "run"
    stage: str = "clm"  # clm | sft | distill
    model_config: str = "configs/model/smalltalk-4m.yaml"
    tokenizer: str = "artifacts/tokenizer-4096"
    train_data: str = "data/processed/train.jsonl"
    val_data: str = "data/processed/val.jsonl"
    output_dir: str = "artifacts/runs"
    init_from: str | None = None  # checkpoint dir to warm-start from (stage 2/3)
    resume: str | None = None  # checkpoint dir to resume optimizer+step state

    seq_len: int = 1024
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 2000
    # Optimizer (defaults straight from the research plan)
    learning_rate: float = 5e-4
    min_lr_ratio: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    warmup_steps: int | None = None
    lr_schedule: str = "cosine"  # cosine | linear | constant

    eval_every: int = 200
    eval_batches: int = 20
    save_every: int = 500
    log_every: int = 20
    keep_last: int = 2

    device: str = "auto"  # auto | cuda | mps | cpu
    dtype: str = "auto"  # auto -> bf16 on cuda, fp32 elsewhere
    compile: bool = False
    seed: int = 1337
    num_workers: int = 0

    wandb: bool = False
    wandb_project: str = "smalltalk-ai"
    wandb_entity: str | None = None

    # stage 2/3 only
    mask_non_assistant: bool = True
    # stage 3 only
    distill_weight: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


def load_model_config(path_or_name: str | Path) -> ModelConfig:
    """Accept either a path or a bare config name (e.g. ``smalltalk-5m``)."""
    p = Path(path_or_name)
    if p.exists():
        return ModelConfig.load(p)
    candidate = CONFIG_DIR / "model" / f"{path_or_name}.yaml"
    if candidate.exists():
        return ModelConfig.load(candidate)
    raise FileNotFoundError(f"no model config at {path_or_name} or {candidate}")


def all_model_configs() -> list[ModelConfig]:
    paths = sorted((CONFIG_DIR / "model").glob("*.yaml"))
    return [ModelConfig.load(p) for p in paths]
