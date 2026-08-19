"""Weight-only post-training quantization (int8 / int4) for deployment.

The research target is a *shipped artifact size*, not a parameter count, so this
module is a first-class part of the study: at a fixed byte budget, lower precision
buys more parameters, and more parameters buy more conversational ability. int4
lets `smalltalk-7m` ship in ~3.9 MB, where `smalltalk-4m` at int8 needs 4.4 MB.

Scheme (deliberately the simplest defensible one):
  * weight-only, symmetric, per-output-channel (per-row) affine quantization
  * activations stay bf16/fp32; we dequantize each weight matrix on use
  * int4 is packed two nibbles per byte
  * embeddings are quantized too (they are 27% of the 4M model, so exempting
    them would defeat the purpose), but `norm` weights and the RoPE tables are
    left in fp32 -- they are ~0.01% of the bytes and very sensitivity-heavy.

This is a size study, not a latency study: dequantizing on the fly is slower than
fp32 on CPU. `--fuse` materialises fp32 weights back for fast inference while
still reporting the quantized artifact size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

QUANT_BITS = {"int8": 8, "int4": 4}
# Layers we never quantize: tiny, and disproportionately sensitive.
SKIP_SUFFIXES = ("norm.weight", "_layernorm.weight")


def _skip(name: str) -> bool:
    return name.endswith(SKIP_SUFFIXES) or "rope" in name or name.endswith("cached")


# ---------------------------------------------------------------------------
# core quant / dequant
# ---------------------------------------------------------------------------
def quantize_tensor(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row quantization. Returns (codes, scales)."""
    assert w.ndim == 2, "expected a 2-D weight matrix"
    qmax = 2 ** (bits - 1) - 1          # int8 -> 127, int4 -> 7
    qmin = -(2 ** (bits - 1))           # int8 -> -128, int4 -> -8
    scale = w.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(torch.round(w / scale), qmin, qmax).to(torch.int8)
    return codes, scale.squeeze(1).to(torch.float32)


def dequantize_tensor(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.to(torch.float32) * scale.unsqueeze(1)


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack int4 values (-8..7) two-per-byte along the last dimension."""
    flat = codes.reshape(codes.shape[0], -1)
    if flat.shape[1] % 2:
        flat = torch.cat([flat, torch.zeros(flat.shape[0], 1, dtype=torch.int8)], dim=1)
    lo = (flat[:, 0::2].to(torch.uint8)) & 0x0F
    hi = (flat[:, 1::2].to(torch.uint8)) & 0x0F
    return (lo | (hi << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, cols: int) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)      # sign-extend the nibble
    hi = torch.where(hi > 7, hi - 16, hi)
    out = torch.stack([lo, hi], dim=2).reshape(packed.shape[0], -1)
    return out[:, :cols]


# ---------------------------------------------------------------------------
# model-level
# ---------------------------------------------------------------------------
@dataclass
class QuantReport:
    precision: str
    quantized_tensors: int
    skipped_tensors: int
    original_bytes: int
    quantized_bytes: int
    max_abs_error: float
    mean_rel_error: float

    @property
    def compression(self) -> float:
        return self.original_bytes / max(self.quantized_bytes, 1)

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "quantized_tensors": self.quantized_tensors,
            "skipped_tensors": self.skipped_tensors,
            "original_mb": round(self.original_bytes / 1048576, 3),
            "quantized_mb": round(self.quantized_bytes / 1048576, 3),
            "compression_x": round(self.compression, 2),
            "max_abs_error": round(self.max_abs_error, 6),
            "mean_rel_error": round(self.mean_rel_error, 6),
        }


def quantize_state_dict(
    state: dict[str, torch.Tensor], precision: str = "int4"
) -> tuple[dict[str, torch.Tensor], dict, QuantReport]:
    if precision not in QUANT_BITS:
        raise ValueError(f"precision must be one of {sorted(QUANT_BITS)}")
    bits = QUANT_BITS[precision]

    out: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    n_q = n_skip = 0
    orig_bytes = quant_bytes = 0
    max_err = 0.0
    rel_errs: list[float] = []

    for name, w in state.items():
        orig_bytes += w.numel() * w.element_size()
        if _skip(name) or w.ndim != 2:
            out[name] = w
            quant_bytes += w.numel() * w.element_size()
            n_skip += 1
            continue

        codes, scale = quantize_tensor(w.float(), bits)
        recon = dequantize_tensor(codes, scale)
        err = (recon - w.float()).abs()
        max_err = max(max_err, float(err.max()))
        denom = w.float().abs().mean().clamp(min=1e-8)
        rel_errs.append(float(err.mean() / denom))

        if bits == 4:
            payload = pack_int4(codes)
        else:
            payload = codes
        out[name + ".codes"] = payload
        out[name + ".scale"] = scale
        meta[name] = {"shape": list(w.shape), "bits": bits}
        quant_bytes += payload.numel() * payload.element_size() + scale.numel() * 4
        n_q += 1

    report = QuantReport(
        precision=precision, quantized_tensors=n_q, skipped_tensors=n_skip,
        original_bytes=orig_bytes, quantized_bytes=quant_bytes,
        max_abs_error=max_err,
        mean_rel_error=sum(rel_errs) / max(len(rel_errs), 1),
    )
    return out, meta, report


def dequantize_state_dict(
    state: dict[str, torch.Tensor], meta: dict[str, dict]
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, info in meta.items():
        codes = state[name + ".codes"]
        scale = state[name + ".scale"]
        rows, cols = info["shape"]
        if info["bits"] == 4:
            codes = unpack_int4(codes, cols)
        out[name] = dequantize_tensor(codes.reshape(rows, cols), scale)
    for k, v in state.items():
        if not k.endswith((".codes", ".scale")):
            out[k] = v
    return out


def save_quantized(model, out_dir: str | Path, precision: str = "int4") -> QuantReport:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    qstate, meta, report = quantize_state_dict(state, precision)
    save_file({k: v.contiguous() for k, v in qstate.items()}, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
    (out_dir / "quantization.json").write_text(
        json.dumps({"precision": precision, "tensors": meta, "report": report.to_dict()}, indent=2)
    )
    return report


def load_quantized(path: str | Path, device: str = "cpu"):
    """Load a quantized checkpoint, dequantizing to fp32 for inference."""
    from safetensors.torch import load_file

    from .config import ModelConfig
    from .model import SmallTalkModel

    path = Path(path)
    qinfo = json.loads((path / "quantization.json").read_text())
    cfg = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
    state = load_file(str(path / "model.safetensors"))
    full = dequantize_state_dict(state, qinfo["tensors"])
    model = SmallTalkModel(cfg)
    model.load_state_dict(full, strict=True)
    return model.to(device)


def artifact_bytes(path: str | Path) -> int:
    """Total on-disk size of a deployable checkpoint (excludes optimizer state)."""
    path = Path(path)
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and p.name != "trainer_state.pt":
            total += p.stat().st_size
    return total
