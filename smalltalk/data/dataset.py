"""Torch datasets for the three training stages.

Stage 1 (CLM)   : conversations packed into fixed-length blocks; loss on all tokens
                  so the model learns the shape of *both* sides of a dialogue.
Stage 2 (SFT)   : one example per conversation, loss masked to assistant tokens only.
Stage 3 (distill): same as SFT but the assistant target is the winning teacher
                  candidate for the final turn.

Design choice: stage 1 packs across conversation boundaries (with <|eos|> between)
because at 1024 tokens and ~120-token dialogues, per-example padding would waste
most of the compute budget. Attention is *not* reset at document boundaries -- an
explicit, documented simplification; the <|eos|> token is a strong enough signal at
this scale and it keeps the attention path a single fast SDPA call.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from ..tokenizer import SmallTalkTokenizer
from .schema import Conversation

DEFAULT_SYSTEM: str | None = None  # small talk needs no system prompt by default


@dataclass
class EncodedConversation:
    ids: list[int]
    assistant_mask: list[int]
    weight: float = 1.0


def encode_conversations(
    conversations: Iterable[Conversation],
    tokenizer: SmallTalkTokenizer,
    max_len: int,
    system_prompt: str | None = DEFAULT_SYSTEM,
    truncate: str = "left",
) -> list[EncodedConversation]:
    """Encode conversations, keeping the *most recent* turns when truncating.

    Truncating from the left preserves the generation target and its nearby
    context, which is what we actually evaluate.
    """
    out: list[EncodedConversation] = []
    for conv in conversations:
        msgs = [m.to_dict() for m in conv.messages]
        if system_prompt and not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        ids, mask = tokenizer.encode_conversation(msgs, add_bos=True, add_eos=True)
        if len(ids) > max_len:
            if truncate == "left":
                ids, mask = ids[-max_len:], mask[-max_len:]
                ids[0], mask[0] = tokenizer.bos_id, 0
            else:
                ids, mask = ids[:max_len], mask[:max_len]
        if sum(mask) == 0:
            continue
        out.append(
            EncodedConversation(ids, mask, float(conv.meta.get("weight", 1.0)))
        )
    return out


class PackedCLMDataset(Dataset):
    """Stage 1: contiguous token blocks. Loss on every token."""

    def __init__(
        self,
        conversations: Sequence[Conversation],
        tokenizer: SmallTalkTokenizer,
        seq_len: int,
        seed: int = 1337,
        shuffle_docs: bool = True,
    ):
        self.seq_len = seq_len
        encoded = encode_conversations(conversations, tokenizer, max_len=seq_len)
        order = list(range(len(encoded)))
        if shuffle_docs:
            random.Random(seed).shuffle(order)

        stream: list[int] = []
        asst: list[int] = []
        for i in order:
            e = encoded[i]
            reps = 2 if e.weight > 1.5 else 1  # integer upweighting only
            for _ in range(reps):
                stream.extend(e.ids)
                asst.extend(e.assistant_mask)
        n_blocks = max(len(stream) // (seq_len + 1), 0)
        if n_blocks == 0:
            raise ValueError(
                f"corpus is too small to fill one block of {seq_len + 1} tokens "
                f"(got {len(stream)}). Lower seq_len or add data."
            )
        usable = n_blocks * (seq_len + 1)
        self.tokens = torch.tensor(stream[:usable], dtype=torch.long).view(n_blocks, -1)
        self.assistant = torch.tensor(asst[:usable], dtype=torch.long).view(n_blocks, -1)
        self.num_tokens = usable

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        block = self.tokens[idx]
        return {
            "input_ids": block[:-1].clone(),
            "labels": block[:-1].clone(),
            "loss_mask": torch.ones(self.seq_len, dtype=torch.long),
            "assistant_mask": self.assistant[idx][:-1].clone(),
        }


class SFTDataset(Dataset):
    """Stage 2/3: per-conversation examples with assistant-only loss masking."""

    def __init__(
        self,
        conversations: Sequence[Conversation],
        tokenizer: SmallTalkTokenizer,
        seq_len: int,
        mask_non_assistant: bool = True,
        use_chosen_candidate: bool = False,
    ):
        self.pad_id = tokenizer.pad_id
        self.seq_len = seq_len
        self.mask_non_assistant = mask_non_assistant
        convs = list(conversations)
        if use_chosen_candidate:
            convs = [_apply_chosen(c) for c in convs]
            convs = [c for c in convs if c is not None]  # type: ignore[misc]
        self.examples = encode_conversations(convs, tokenizer, max_len=seq_len)
        if not self.examples:
            raise ValueError("no usable SFT examples (all had zero assistant tokens)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.examples[idx]
        n = len(e.ids)
        pad = self.seq_len - n
        ids = e.ids + [self.pad_id] * pad
        mask = (e.assistant_mask if self.mask_non_assistant else [1] * n) + [0] * pad
        t = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": t,
            "labels": t.clone(),
            "loss_mask": torch.tensor(mask, dtype=torch.long),
            "weight": torch.tensor(e.weight, dtype=torch.float),
        }


def _apply_chosen(conv: Conversation) -> Conversation | None:
    """Replace the final assistant turn with the winning teacher candidate."""
    if not conv.candidates:
        return conv
    idx = conv.chosen
    if idx is None:
        scored = [(c.score if c.score is not None else -1e9, i) for i, c in enumerate(conv.candidates)]
        idx = max(scored)[1]
    if idx >= len(conv.candidates):
        return None
    best = conv.candidates[idx].content.strip()
    if not best:
        return None
    msgs = list(conv.messages)
    if msgs and msgs[-1].role == "assistant":
        msgs = msgs[:-1]
    from .schema import Turn

    msgs.append(Turn("assistant", best))
    return Conversation(
        id=conv.id, messages=msgs, source=conv.source, meta=dict(conv.meta)
    )


def collate(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def build_dataset(
    stage: str,
    conversations: Sequence[Conversation],
    tokenizer: SmallTalkTokenizer,
    seq_len: int,
    mask_non_assistant: bool = True,
    seed: int = 1337,
) -> Dataset:
    if stage == "clm":
        return PackedCLMDataset(conversations, tokenizer, seq_len, seed=seed)
    if stage == "sft":
        return SFTDataset(conversations, tokenizer, seq_len, mask_non_assistant)
    if stage == "distill":
        return SFTDataset(
            conversations, tokenizer, seq_len, mask_non_assistant, use_chosen_candidate=True
        )
    raise ValueError(f"unknown stage {stage!r} (expected clm|sft|distill)")
