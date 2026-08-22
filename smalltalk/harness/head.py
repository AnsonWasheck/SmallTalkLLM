"""A tiny learned policy classifier over the frozen model's hidden state.

Justified by three independent measurements, not by the size budget having room:

  * ORACLE_POLICY scores 0.733 against A_RAW's 0.675, so a correct policy label
    is worth +5.8 points. The prize is real.
  * Same-model exemplar scoring identifies the policy 19.2% of the time (18.3%
    with PMI normalisation). The model cannot classify its own policy.
  * Reranking the model's top-k candidates by its own likelihood scores 0.09
    against greedy's 0.675. The model's probabilities cannot select either.

Together those say the headroom (top-4 = 75.8%, top-16 = 79.2%) is unreachable
from anything derived from the language-model head, and needs a small amount of
discrimination trained separately. The base model stays frozen; this reads its
final hidden state and nothing more.

Deliberately linear. A linear probe answers a sharper question than an MLP does:
is conversational policy LINEARLY encoded in the 256-dim representation? If it is
not, that is a finding about the curriculum, and the right response is to fix the
training data rather than to grow the classifier until it memorises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .policy import POLICIES

POLICY_IDS: list[str] = [p.pid for p in POLICIES]
PID_INDEX: dict[str, int] = {p: i for i, p in enumerate(POLICY_IDS)}


class PolicyHead(nn.Module):
    """hidden_state (+ optional deterministic features) -> policy logits."""

    def __init__(self, hidden_size: int = 256, n_features: int = 0,
                 n_classes: int = len(POLICY_IDS), hidden_units: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_features = n_features
        in_dim = hidden_size + n_features
        if hidden_units:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_units), nn.GELU(),
                nn.Linear(hidden_units, n_classes))
        else:
            self.net = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ---- size accounting ------------------------------------------------
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(),
                    "hidden_size": self.hidden_size,
                    "n_features": self.n_features,
                    "classes": POLICY_IDS}, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PolicyHead":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        head = cls(hidden_size=blob["hidden_size"], n_features=blob["n_features"])
        head.load_state_dict(blob["state_dict"])
        return head.eval()


# --- deterministic feature vector -------------------------------------
FEATURE_KEYS = [
    "n_words", "has_question_mark", "has_greeting", "has_goodbye", "has_thanks",
    "has_apology", "has_correction", "has_negation", "has_emoji", "all_caps",
    "repeated_punct", "prev_assistant_was_question",
    "consecutive_assistant_questions",
]


def feature_vector(f) -> torch.Tensor:
    d = f.as_dict()
    vals = []
    for k in FEATURE_KEYS:
        v = d.get(k, 0)
        if k == "n_words":
            v = min(float(v), 30.0) / 30.0        # bounded, roughly unit scale
        elif k == "consecutive_assistant_questions":
            v = min(float(v), 3.0) / 3.0
        vals.append(float(v))
    return torch.tensor(vals, dtype=torch.float32)


@torch.no_grad()
def hidden_state(model, input_ids: list[int], device=None) -> torch.Tensor:
    """Final pre-logit representation at the last prompt position.

    Captured with a forward hook on the model's final RMSNorm so the base model
    is read, never modified, and the hook is removed immediately afterwards.
    """
    device = device or next(model.parameters()).device
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, out):
        captured.append(out.detach())

    handle = model.norm.register_forward_hook(hook)
    try:
        model(torch.tensor([input_ids], dtype=torch.long, device=device))
    finally:
        handle.remove()
    return captured[0][0, -1].float().cpu()
