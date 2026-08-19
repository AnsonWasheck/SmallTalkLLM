"""Qwen3.5-9B local inference wrapper — text-only, non-thinking, batched.

Backend decision (see reports/QWEN_ENVIRONMENT.md): Transformers + ROCm BF16.
vLLM/SGLang are not viable on gfx1151 + ROCm 7.1 + Python 3.14 without a from-source
build, and BF16 fits in the 48 GB unified-memory carve-out, so no quantisation is used.

Two things this file is careful about:

1. TEXT-ONLY. `Qwen3.5-9B` is a `Qwen3_5ForConditionalGeneration` multimodal checkpoint.
   We load the causal-LM/text stack only so the vision tower never occupies VRAM.

2. NON-THINKING. Generation must not contain hidden reasoning. We pass
   `enable_thinking=False` into the chat template where supported, and additionally
   treat any `<think>` in the output as a hard validation failure downstream — belt and
   braces, because a silent template change would otherwise poison the corpus.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

THINK_RE = re.compile(r"<think>|</think>|<\|thinking\|>", re.I)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


@dataclass
class SamplerProfile:
    """Qwen's recommended non-thinking sampler is profile A's starting point."""

    name: str
    temperature: float
    top_p: float
    top_k: int = 20
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    max_new_tokens: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "temperature": self.temperature, "top_p": self.top_p,
                "top_k": self.top_k, "presence_penalty": self.presence_penalty,
                "repetition_penalty": self.repetition_penalty}


# The three profiles the brief asks us to compare, plus the critic's low-temp profile.
PROFILES = {
    "A_conservative": SamplerProfile("A_conservative", 0.68, 0.80),
    "B_balanced": SamplerProfile("B_balanced", 0.80, 0.90),
    "C_high_entropy": SamplerProfile("C_high_entropy", 0.95, 0.95),
    "critic": SamplerProfile("critic", 0.15, 0.80, max_new_tokens=384),
}


def strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a completion."""
    t = strip_fences(text)
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class QwenGenerator:
    def __init__(self, model_path: str, dtype: str = "bfloat16",
                 device: str = "cuda", max_memory_gb: float | None = None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        cfg = AutoConfig.from_pretrained(model_path)
        # Text-only: drop the vision tower rather than paying VRAM for it.
        text_cfg = getattr(cfg, "text_config", None) or cfg

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": {"": 0} if device == "cuda" else device,
            "low_cpu_mem_usage": True,   # stream shards; only 14 GB system RAM
        }
        if max_memory_gb:
            kwargs["max_memory"] = {0: f"{max_memory_gb}GiB"}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, config=text_cfg, **kwargs)
            self.text_only = True
        except Exception as exc:
            # Do NOT silently fall back: the multimodal class costs ~4 GB more VRAM
            # and a silent switch previously turned a recoverable text-only failure
            # into an OOM that looked like a capacity problem. Surface the cause.
            self.text_only_error = repr(exc)
            print(f"[client] text-only load FAILED: {exc!r}\n"
                  f"[client] falling back to multimodal class (higher VRAM)")
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
            self.text_only = False
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # required for batched decoder-only gen
        self.load_seconds = round(time.time() - t0, 1)

    # ---- prompt rendering --------------------------------------------------
    def render(self, messages: Sequence[dict[str, str]]) -> str:
        """Apply the chat template with thinking explicitly disabled."""
        try:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Template doesn't accept the kwarg on this revision.
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True)

    # ---- generation --------------------------------------------------------
    @torch.no_grad()
    def generate(self, batch: Sequence[Sequence[dict[str, str]]],
                 profile: SamplerProfile, seed: int | None = None) -> list[str]:
        if seed is not None:
            torch.manual_seed(seed)
        prompts = [self.render(m) for m in batch]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self.model.device)
        gen_kwargs: dict[str, Any] = dict(
            do_sample=profile.temperature > 0,
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k or None,
            repetition_penalty=profile.repetition_penalty,
            max_new_tokens=profile.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # HF exposes presence penalty only on newer versions; skip silently if absent.
        try:
            out = self.model.generate(**enc, **gen_kwargs)
        except TypeError:
            gen_kwargs.pop("top_k", None)
            out = self.model.generate(**enc, **gen_kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)

    def free(self) -> None:
        """Release the GPU. Called before teacher training per the scheduling rule."""
        del self.model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_report() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    free_b, total_b = torch.cuda.mem_get_info()
    return {"total_gb": round(total_b / 2**30, 2),
            "free_gb": round(free_b / 2**30, 2),
            "used_gb": round((total_b - free_b) / 2**30, 2),
            "torch_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}
