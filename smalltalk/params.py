"""Analytic + empirical parameter counting.

We compute the count twice: from the formula (so the cost model is explicit and
auditable) and from the instantiated module. Any mismatch against the target in
the config is *reported*, never silently patched by changing the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig


@dataclass
class ParamBreakdown:
    embedding: int
    attention_per_layer: int
    mlp_per_layer: int
    norms_per_layer: int
    final_norm: int
    lm_head: int
    num_layers: int

    @property
    def per_layer(self) -> int:
        return self.attention_per_layer + self.mlp_per_layer + self.norms_per_layer

    @property
    def total(self) -> int:
        return (
            self.embedding
            + self.per_layer * self.num_layers
            + self.final_norm
            + self.lm_head
        )

    def as_rows(self) -> list[tuple[str, int]]:
        return [
            ("embed_tokens", self.embedding),
            ("attention (per layer)", self.attention_per_layer),
            ("mlp (per layer)", self.mlp_per_layer),
            ("norms (per layer)", self.norms_per_layer),
            (f"all {self.num_layers} layers", self.per_layer * self.num_layers),
            ("final norm", self.final_norm),
            ("lm_head", self.lm_head),
            ("TOTAL", self.total),
        ]


def analytic_breakdown(cfg: ModelConfig) -> ParamBreakdown:
    h = cfg.hidden_size
    attn = h * cfg.q_dim + 2 * (h * cfg.kv_dim) + cfg.q_dim * h
    mlp = 3 * (h * cfg.intermediate_size)
    return ParamBreakdown(
        embedding=cfg.vocab_size * h,
        attention_per_layer=attn,
        mlp_per_layer=mlp,
        norms_per_layer=2 * h,
        final_norm=h,
        lm_head=0 if cfg.tie_word_embeddings else cfg.vocab_size * h,
        num_layers=cfg.num_layers,
    )


def analytic_param_count(cfg: ModelConfig) -> int:
    return analytic_breakdown(cfg).total


def empirical_param_count(cfg: ModelConfig, trainable_only: bool = True) -> int:
    from .model import build_model

    model = build_model(cfg)
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


# Deployment byte budget. The research target is not really a parameter count --
# it is a *shipped artifact size*. Parameters are the proxy; bytes are the goal.
BITS_PER_WEIGHT = {"fp32": 32, "bf16": 16, "fp16": 16, "int8": 8, "int4": 4}

# Rough fixed overheads for a self-contained deployable artifact.
TOKENIZER_BYTES = 560_000     # measured: byte-level BPE json at 4096 vocab
METADATA_BYTES = 160_000      # safetensors headers + config.json


def deployed_bytes(cfg: ModelConfig, precision: str = "int8", include_tokenizer: bool = True) -> int:
    """Bytes needed to ship this model for inference (no optimizer state)."""
    if precision not in BITS_PER_WEIGHT:
        raise ValueError(f"precision must be one of {sorted(BITS_PER_WEIGHT)}")
    n = analytic_param_count(cfg)
    weights = n * BITS_PER_WEIGHT[precision] // 8
    extra = METADATA_BYTES + (TOKENIZER_BYTES if include_tokenizer else 0)
    return weights + extra


def fits_budget(cfg: ModelConfig, budget_mb: float, precision: str = "int8") -> bool:
    return deployed_bytes(cfg, precision) <= budget_mb * 1024 * 1024


def max_params_for_budget(budget_mb: float, precision: str = "int8",
                          include_tokenizer: bool = True) -> int:
    """How many parameters fit in a byte budget at a given precision."""
    extra = METADATA_BYTES + (TOKENIZER_BYTES if include_tokenizer else 0)
    usable = budget_mb * 1024 * 1024 - extra
    return max(0, int(usable * 8 // BITS_PER_WEIGHT[precision]))


@dataclass
class ParamCheck:
    name: str
    analytic: int
    empirical: int
    expected: int | None
    tolerance: float

    @property
    def internally_consistent(self) -> bool:
        return self.analytic == self.empirical

    @property
    def rel_error(self) -> float | None:
        if not self.expected:
            return None
        return (self.empirical - self.expected) / self.expected

    @property
    def ok(self) -> bool:
        if not self.internally_consistent:
            return False
        err = self.rel_error
        return err is None or abs(err) <= self.tolerance

    def report(self) -> str:
        lines = [
            f"{self.name:>16}: {self.empirical:>12,} params "
            f"({self.empirical / 1e6:.2f}M)"
        ]
        if not self.internally_consistent:
            lines.append(
                f"{'':>16}  MISMATCH formula={self.analytic:,} vs module={self.empirical:,}"
            )
        if self.expected:
            err = self.rel_error or 0.0
            flag = "ok" if abs(err) <= self.tolerance else "OFF-TARGET"
            lines.append(
                f"{'':>16}  target {self.expected:,} ({self.expected / 1e6:.2f}M) "
                f"delta {self.empirical - self.expected:+,} ({err:+.2%}) [{flag}]"
            )
        return "\n".join(lines)


def check_config(cfg: ModelConfig, empirical: bool = True) -> ParamCheck:
    return ParamCheck(
        name=cfg.name,
        analytic=analytic_param_count(cfg),
        empirical=empirical_param_count(cfg) if empirical else analytic_param_count(cfg),
        expected=cfg.expected_params,
        tolerance=cfg.param_tolerance,
    )
