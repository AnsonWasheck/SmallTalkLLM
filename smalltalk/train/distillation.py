"""Token-level online knowledge distillation for a same-tokenizer teacher.

This is intentionally separate from rejection-selected SFT.  Candidate selection
trains on one chosen string; this module transfers the teacher's full 4096-way
next-token distribution without materialising logits for the corpus on disk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_positions(labels: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Boolean ``(B, T-1)`` mask aligned to next-token logits."""
    valid = labels[:, 1:] != -100
    return valid if loss_mask is None else valid & loss_mask[:, 1:].bool()


def causal_ce_and_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    *,
    alpha: float = 0.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return CE + temperature-scaled forward KL over supervised positions.

    ``alpha`` is the gold CE weight. Both models must share exact token IDs;
    callers should reject a tokenizer/config mismatch before invoking this.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = student_logits[:, :-1].float()
    t = teacher_logits[:, :-1].float()
    target = labels[:, 1:]
    mask = supervised_positions(labels, loss_mask)
    n = mask.sum().clamp(min=1)

    ce_each = F.cross_entropy(s.reshape(-1, s.size(-1)), target.reshape(-1),
                              ignore_index=-100, reduction="none").view_as(target)
    ce = (ce_each * mask).sum() / n
    log_s = F.log_softmax(s / temperature, dim=-1)
    prob_t = F.softmax(t / temperature, dim=-1)
    kl_each = (prob_t * (prob_t.clamp_min(1e-9).log() - log_s)).sum(dim=-1)
    kl = (kl_each * mask).sum() / n * (temperature * temperature)
    loss = alpha * ce + (1.0 - alpha) * kl
    return loss, {"ce": float(ce.detach()), "kl": float(kl.detach()), "tokens": int(n.detach())}
