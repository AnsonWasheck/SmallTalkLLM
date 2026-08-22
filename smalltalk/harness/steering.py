"""Policy-to-language control interfaces.

Phase 2 established that a 4,883-parameter linear probe recovers the
conversational policy from the frozen hidden state at 76.9% on held-out
benchmark surfaces, while the model's own LM head acts on that knowledge poorly.
Phase 3 asks how to transfer a KNOWN policy into generated text.

Three interfaces, in increasing order of invasiveness. All keep the base model
frozen and all leave the reply autoregressively generated:

  RESTRICT   the first token must come from the policy's mined opening set
  BIAS       the policy's opening tokens get a logit bonus for the first n steps
  HIDDEN     a policy centroid direction is added to the hidden state before the
             LM head, for the first n positions

RESTRICT is the bluntest and the most likely to collapse into templating, since
the corpus has one canonical target per intent. Its attractor concentration is
measured, not assumed. BIAS is the same signal applied softly, which is why both
exist: if BIAS captures most of RESTRICT's gain without the collapse, that is the
better mechanism even at equal score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class PrefixMap:
    """Mined policy -> opening-token statistics. A static harness asset."""
    table: dict[str, list[dict]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PrefixMap":
        p = Path(path)
        return cls(json.loads(p.read_text()) if p.exists() else {})

    def tokens(self, pid: str) -> list[int]:
        return [r["token"] for r in self.table.get(pid, {}).get("first", [])]

    def weights(self, pid: str) -> dict[int, float]:
        """Bias proportional to log-lift: a token three times more specific to a
        policy earns more push than one that is barely specific at all."""
        import math
        return {r["token"]: math.log(max(r["lift"], 1.0001))
                for r in self.table.get(pid, {}).get("first", [])}


@torch.no_grad()
def steered_generate(model, tokenizer, prompt_ids, gen, *, device=None,
                     allowed_first: list[int] | None = None,
                     bias: dict[int, float] | None = None,
                     bias_steps: int = 1, bias_scale: float = 1.0,
                     hidden_vec: torch.Tensor | None = None,
                     hidden_steps: int = 1, hidden_alpha: float = 0.0):
    """Greedy generation with optional steering on the opening tokens.

    Mirrors infer.generate for the unsteered path so that steering strength zero
    reproduces the baseline exactly; the harness must be able to turn every
    mechanism off and land back on A_RAW.
    """
    device = device or next(model.parameters()).device
    stop = {tokenizer.endofturn_id, tokenizer.eos_id}
    ids = list(prompt_ids)[-gen.max_context:]

    # Hidden-state steering is applied through a hook on the final norm, whose
    # output feeds the (tied) LM head. The base weights are never touched.
    handle = None
    step_ref = {"i": 0}
    if hidden_vec is not None and hidden_alpha:
        v = hidden_vec.to(device)

        def hook(_m, _i, out):
            if step_ref["i"] < hidden_steps:
                scale = hidden_alpha * out[:, -1:].norm(dim=-1, keepdim=True) / \
                    (v.norm() + 1e-6)
                out = out.clone()
                out[:, -1:] = out[:, -1:] + scale * v
            return out

        handle = model.norm.register_forward_hook(hook)

    try:
        cache = model.new_cache()
        logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device),
                          cache=cache)
        out: list[int] = []
        for step in range(gen.max_new_tokens):
            step_ref["i"] = step
            lg = logits[0, -1].float().clone()
            lg[tokenizer.pad_id] = float("-inf")
            if step == 0 and allowed_first:
                mask = torch.full_like(lg, float("-inf"))
                idx = torch.tensor(allowed_first, device=lg.device)
                mask[idx] = lg[idx]
                lg = mask
            if bias and step < bias_steps:
                for t, b in bias.items():
                    lg[t] += bias_scale * b
            nxt = int(torch.argmax(lg))
            if nxt in stop:
                break
            out.append(nxt)
            cache.trim(model.cfg.max_position_embeddings - 1)
            step_ref["i"] = step + 1
            logits, _ = model(torch.tensor([[nxt]], dtype=torch.long, device=device),
                              cache=cache)
        return out
    finally:
        if handle is not None:
            handle.remove()
