"""Llama-compatible micro decoder: RMSNorm, RoPE, SwiGLU, GQA, tied embeddings.

Deliberately written as one readable file so every architectural assumption in the
scaling study can be modified in place. No bias terms anywhere, no dropout
(we are data-limited, not overfitting-limited, at <=30M params -- see docs/DESIGN_CHOICES.md).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """Standard Llama RoPE with a cached cos/sin table (non-persistent buffers)."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self._build(max_seq_len, torch.device("cpu"), torch.float32)

    def _build(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim)
        )
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self.cached_len = seq_len

    def forward(self, seq_len: int, offset: int, device: torch.device, dtype: torch.dtype):
        need = offset + seq_len
        if need > self.cached_len or self.cos_cached.device != device:
            self._build(max(need, self.cached_len), device, torch.float32)
        cos = self.cos_cached[offset : offset + seq_len].to(dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, H, T, D); cos/sin: (T, D)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


@dataclass
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    """Growing KV cache with physical and absolute-position bookkeeping.

    ``length`` is the number of *physical* keys retained.  ``next_position`` is
    the absolute RoPE position for the next token.  Those deliberately diverge
    after :meth:`trim`: dropping old keys must never cause RoPE positions to be
    reused for newly generated tokens.
    """

    def __init__(self, num_layers: int):
        self.layers: list[LayerCache | None] = [None] * num_layers
        self.next_position = 0
        self.cache_start_position = 0

    @property
    def length(self) -> int:
        """Cached positions, measured on the LAST layer.

        Layer 0 is updated before the deeper layers run, so reading layer 0
        mid-forward would report a length that already includes the current
        chunk. `layer_length()` is what attention must use.
        """
        last = self.layers[-1]
        return 0 if last is None else last.key.shape[2]

    def layer_length(self, idx: int) -> int:
        lc = self.layers[idx]
        return 0 if lc is None else lc.key.shape[2]

    def update(self, idx: int, key: torch.Tensor, value: torch.Tensor):
        cur = self.layers[idx]
        if cur is None:
            self.layers[idx] = LayerCache(key, value)
        else:
            self.layers[idx] = LayerCache(
                torch.cat([cur.key, key], dim=2), torch.cat([cur.value, value], dim=2)
            )
        lc = self.layers[idx]
        return lc.key, lc.value

    def advance(self, tokens: int) -> None:
        """Advance the shared absolute position once per model forward call."""
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        self.next_position += tokens

    def trim(self, max_len: int) -> None:
        """Drop oldest positions so the cache never exceeds `max_len`."""
        if max_len < 0:
            raise ValueError("max_len must be non-negative")
        for i, lc in enumerate(self.layers):
            if lc is not None and lc.key.shape[2] > max_len:
                self.layers[i] = LayerCache(lc.key[:, :, -max_len:], lc.value[:, :, -max_len:])
        # All layers are advanced together.  This is diagnostic metadata rather
        # than an attention offset: retained keys already carry their RoPE phase.
        self.cache_start_position = self.next_position - self.length


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.n_rep = cfg.num_query_groups
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        attention_mask: torch.Tensor | None = None,
        position_offset: int | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        # Per-layer offset: deeper layers must not see layer 0's just-written chunk.
        offset = cache.layer_length(layer_idx) if cache is not None else 0

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)

        # ``offset`` is a physical key count for the causal mask.  RoPE needs an
        # absolute position, which remains monotonic when a sliding cache trims.
        rope_offset = (
            position_offset
            if position_offset is not None
            else (cache.next_position if cache is not None else offset)
        )
        cos, sin = self.rope(t, rope_offset, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # With a cache and t == 1, every cached position is a legal target.
        # With t > 1 *and* a non-empty cache (chunked prefill), SDPA's is_causal
        # aligns the mask top-left, which is wrong once queries are offset -- so
        # we build the shifted causal mask explicitly.
        is_causal = False
        if attention_mask is None and t > 1:
            if offset == 0:
                is_causal = True
            else:
                total = offset + t
                pos = torch.arange(offset, total, device=x.device)[:, None]
                key_pos = torch.arange(total, device=x.device)[None, :]
                attention_mask = (key_pos <= pos)[None, None]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.cfg.q_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, rope)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cache=None, attention_mask=None, position_offset=None):
        x = x + self.self_attn(
            self.input_layernorm(x), cache, self.layer_idx, attention_mask, position_offset
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class SmallTalkModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.layers = nn.ModuleList(
            [Block(cfg, self.rope, i) for i in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None  # weights reused from embed_tokens
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Scaled init on residual output projections (GPT-2 style depth stabilisation).
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(
                    p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.num_layers)
                )

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ---- forward -----------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        cache: KVCache | None = None,
        loss_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """Returns (logits, loss). `loss_mask` (B,T) selects which label positions count.

        `labels` are expected already shifted-by-caller? No: we shift internally, so
        pass `labels == input_ids` for plain causal LM. `loss_mask` aligns with
        `labels` (i.e. mask[t] == 1 means "predicting token t counts").
        """
        if segment_ids is not None and cache is not None:
            raise ValueError("segment_ids block masks are only supported without a KV cache")
        attention_mask = None
        if segment_ids is not None:
            if segment_ids.shape != input_ids.shape:
                raise ValueError("segment_ids must have shape (batch, sequence)")
            # Correctness-first packed-document isolation: a token can only see
            # earlier tokens from its own source conversation.  This is a full
            # mask, so callers should use it for packed CLM training only.
            t = input_ids.shape[1]
            causal = torch.ones((t, t), dtype=torch.bool, device=input_ids.device).tril()
            same_doc = segment_ids[:, :, None] == segment_ids[:, None, :]
            attention_mask = (same_doc & causal[None, :, :])[:, None, :, :]

        position_offset = cache.next_position if cache is not None else None
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cache=cache, attention_mask=attention_mask, position_offset=position_offset)
        if cache is not None:
            cache.advance(input_ids.shape[1])
        x = self.norm(x)
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(x, weight)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
            flat_labels = shift_labels.view(-1)
            if loss_mask is not None:
                mask = loss_mask[:, 1:].contiguous().reshape(-1).bool()
                flat_labels = flat_labels.masked_fill(~mask, -100)
            per_tok = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
            n = (flat_labels != -100).sum().clamp(min=1)
            loss = per_tok.sum() / n if reduction == "mean" else per_tok.sum()
        return logits, loss

    # ---- checkpointing -----------------------------------------------------
    def save_pretrained(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        try:
            from safetensors.torch import save_file

            save_file(state, str(path / "model.safetensors"))
        except Exception:  # pragma: no cover - fallback when safetensors absent
            torch.save(state, path / "model.pt")
        (path / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        return path

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device = "cpu"):
        path = Path(path)
        cfg = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
        model = cls(cfg)
        st = path / "model.safetensors"
        if st.exists():
            from safetensors.torch import load_file

            state = load_file(str(st))
        else:
            state = torch.load(path / "model.pt", map_location="cpu")
        model.load_state_dict(state)
        return model.to(device)

    # ---- helpers -----------------------------------------------------------
    def new_cache(self) -> KVCache:
        return KVCache(self.cfg.num_layers)

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)


def build_model(cfg: ModelConfig) -> SmallTalkModel:
    return SmallTalkModel(cfg)
