"""Sampling + a conversation engine that keeps the recent dialogue in context.

Defaults come from the research plan (temp 0.7, top_p 0.9, 48 max tokens, mild
repetition penalty) but every one is a tunable knob swept by evaluate.py -- they
are starting points, not truths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from ..model import KVCache, SmallTalkModel
from ..tokenizer import SmallTalkTokenizer


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0                # 0 = disabled
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    max_new_tokens: int = 48
    min_new_tokens: int = 1
    greedy: bool = False
    no_repeat_ngram_size: int = 0
    max_context: int = 1024       # tokens of dialogue history kept in context
    seed: int | None = None

    def with_(self, **kw) -> "GenerationConfig":
        d = dict(self.__dict__)
        d.update(kw)
        return GenerationConfig(**d)


def _apply_repetition_penalty(
    logits: torch.Tensor, generated: Sequence[int], penalty: float, presence: float
) -> torch.Tensor:
    if (penalty == 1.0 and presence == 0.0) or not generated:
        return logits
    idx = torch.tensor(sorted(set(generated)), device=logits.device, dtype=torch.long)
    if penalty != 1.0:
        vals = logits[idx]
        logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    if presence:
        logits[idx] -= presence
    return logits


def _filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative - probs > top_p     # always keep the top token
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    return logits


def _banned_by_ngram(generated: Sequence[int], n: int) -> set[int]:
    if n <= 0 or len(generated) < n:
        return set()
    prefix = tuple(generated[-(n - 1) :]) if n > 1 else ()
    banned = set()
    for i in range(len(generated) - n + 1):
        window = tuple(generated[i : i + n])
        if window[:-1] == prefix:
            banned.add(window[-1])
    return banned


@torch.no_grad()
def generate(
    model: SmallTalkModel,
    tokenizer: SmallTalkTokenizer,
    prompt_ids: Sequence[int],
    cfg: GenerationConfig | None = None,
    stop_ids: Sequence[int] | None = None,
    device: str | torch.device | None = None,
) -> list[int]:
    """Return only the newly generated token ids (stop token excluded)."""
    cfg = cfg or GenerationConfig()
    device = device or next(model.parameters()).device
    model.eval()
    gen_state = torch.Generator(device="cpu")
    if cfg.seed is not None:
        gen_state.manual_seed(cfg.seed)

    stop = set(stop_ids if stop_ids is not None else (tokenizer.endofturn_id, tokenizer.eos_id))
    ids = list(prompt_ids)[-cfg.max_context :]
    cache = model.new_cache()

    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x, cache=cache)
    out: list[int] = []

    for step in range(cfg.max_new_tokens):
        step_logits = logits[0, -1].float().clone()
        step_logits = _apply_repetition_penalty(
            step_logits, ids[-cfg.max_context :] + out, cfg.repetition_penalty, cfg.presence_penalty
        )
        for t in _banned_by_ngram(out, cfg.no_repeat_ngram_size):
            step_logits[t] = float("-inf")
        step_logits[tokenizer.pad_id] = float("-inf")
        if step < cfg.min_new_tokens:
            for s in stop:
                step_logits[s] = float("-inf")

        if cfg.greedy or cfg.temperature <= 0:
            nxt = int(torch.argmax(step_logits))
        else:
            step_logits = step_logits / cfg.temperature
            step_logits = _filter_top_k_top_p(step_logits, cfg.top_k, cfg.top_p)
            probs = F.softmax(step_logits, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=gen_state))

        if nxt in stop:
            break
        out.append(nxt)
        cache.trim(model.cfg.max_position_embeddings - 1)
        logits, _ = model(
            torch.tensor([[nxt]], dtype=torch.long, device=device), cache=cache
        )
    return out


# ---------------------------------------------------------------------------
# Conversation engine
# ---------------------------------------------------------------------------
@dataclass
class ConversationEngine:
    """Holds dialogue history and produces assistant replies.

    `use_state=True` enables the optional *zero-parameter* application-level
    conversation state (see state.py). It is off by default so pure-model
    performance can be measured independently -- that separation is the point.
    """

    model: SmallTalkModel
    tokenizer: SmallTalkTokenizer
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    system_prompt: str | None = None
    use_state: bool = False
    history: list[dict[str, str]] = field(default_factory=list)
    state: object | None = None

    def __post_init__(self) -> None:
        if self.use_state and self.state is None:
            from .state import ConversationState

            self.state = ConversationState()
        if self.system_prompt:
            self.history.insert(0, {"role": "system", "content": self.system_prompt})

    def reset(self) -> None:
        self.history = (
            [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        )
        if self.state is not None:
            self.state = type(self.state)()

    def _context(self) -> list[dict[str, str]]:
        msgs = list(self.history)
        if self.use_state and self.state is not None:
            hint = self.state.as_system_hint()  # type: ignore[attr-defined]
            if hint:
                msgs = [{"role": "system", "content": hint}] + [
                    m for m in msgs if m["role"] != "system"
                ]
        return msgs

    def reply(self, user_text: str, gen: GenerationConfig | None = None) -> str:
        cfg = gen or self.gen
        self.history.append({"role": "user", "content": user_text.strip()})
        if self.state is not None:
            self.state.observe("user", user_text)  # type: ignore[attr-defined]

        ids, _ = self.tokenizer.encode_conversation(
            self._context(), add_bos=True, add_generation_prompt=True
        )
        # Keep the whole recent dialogue; drop only the oldest tokens.
        ids = ids[-cfg.max_context :]
        new = generate(self.model, self.tokenizer, ids, cfg)
        text = self.tokenizer.decode(new).strip()
        if not text:
            text = "..."
        self.history.append({"role": "assistant", "content": text})
        if self.state is not None:
            self.state.observe("assistant", text)  # type: ignore[attr-defined]
        return text

    def run_scenario(
        self, user_turns: Sequence[str], gen: GenerationConfig | None = None
    ) -> list[dict[str, str]]:
        self.reset()
        for u in user_turns:
            self.reply(u, gen=gen)
        return [m for m in self.history if m["role"] != "system"]


def load_engine(
    checkpoint: str,
    tokenizer_path: str | None = None,
    device: str = "auto",
    gen: GenerationConfig | None = None,
    use_state: bool = False,
) -> ConversationEngine:
    from pathlib import Path

    from ..train.utils import resolve_device

    ckpt = Path(checkpoint)
    dev = resolve_device(device)
    model = SmallTalkModel.from_pretrained(ckpt, device=dev)
    tok_path = tokenizer_path
    if tok_path is None:
        for cand in (ckpt / "tokenizer", ckpt.parent / "tokenizer", ckpt):
            if (cand / "tokenizer.json").exists():
                tok_path = str(cand)
                break
    if tok_path is None:
        raise FileNotFoundError(
            f"no tokenizer.json found near {ckpt}; pass --tokenizer explicitly"
        )
    tokenizer = SmallTalkTokenizer.load(tok_path)
    return ConversationEngine(model, tokenizer, gen or GenerationConfig(), use_state=use_state)
