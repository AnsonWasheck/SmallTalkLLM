"""Conversational byte-level BPE tokenizer + chat template.

Design choices (see docs/DESIGN_CHOICES.md):
  * byte-level BPE => no UNK, emoji/typos survive, and we can keep the vocab tiny.
  * 4096 / 6144 vocab only. At 256-dim hidden, a 32k vocab would be 8.4M params --
    more than the entire 8M model. Vocabulary is the single most expensive thing
    we can spend parameters on, and our domain is narrow English small talk.
  * Special tokens are *turn-structural*, not instruction-ish: <|system|>, <|user|>,
    <|assistant|>, <|endofturn|>, plus <|bos|>/<|eos|>/<|pad|>.
  * A tiny curated `ALWAYS_KEEP` list forces single tokens for high-frequency
    conversational atoms (contractions, interjections, fillers) so a 4-layer model
    doesn't burn depth re-assembling "don't" from three pieces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
ENDOFTURN = "<|endofturn|>"

# Response-length policy tokens (v0.2-Core). The assistant emits one of these
# first, turning "how long should this reply be" into an explicit decision rather
# than something a 6.7M model must infer from an ambient style prior. They are
# declared as specials so the BPE trainer allocates them inside the 4096 budget:
# vocab_size is unchanged, so the released model stays at exactly 6,689,024
# parameters. They cost four displaced merges, which is the intended trade.
LEN_REACTION = "<|len_reaction|>"
LEN_VSHORT = "<|len_vshort|>"
LEN_SHORT = "<|len_short|>"
LEN_MEDIUM = "<|len_medium|>"
LENGTH_TOKENS = [LEN_REACTION, LEN_VSHORT, LEN_SHORT, LEN_MEDIUM]

# Referent placeholder. The model marks WHERE the user's word belongs; the
# harness renders WHICH word it is.
#
# Measured justification: elaboration succeeded on 11% of one-token referents and
# 0 of 29 multi-token ones -- copying a span like "bricklayer" is beyond eight
# layers with one KV head. Restricting the curriculum to copyable nouns is not an
# option either, since only 28 of 402 bank nouns are single-token and no place
# name is. Emitting a single placeholder token is trivially learnable, and it
# generalises to any noun including ones never seen, because the model never
# needs to know the word at all.
#
# Declared as a special so the BPE trainer allocates it inside the 4096 budget:
# vocabulary size is unchanged and the model stays at exactly 6,689,024
# parameters. It costs one displaced merge.
REF = "<|ref|>"

SPECIAL_TOKENS = [PAD, BOS, EOS, SYSTEM, USER, ASSISTANT, ENDOFTURN,
                  *LENGTH_TOKENS, REF]

ROLE_TOKENS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}

# Conversational atoms we insist on having as single tokens.
ALWAYS_KEEP: list[str] = [
    # contractions
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "n't",
    "don't", "didn't", "doesn't", "can't", "won't", "isn't", "wasn't", "aren't",
    "haven't", "hasn't", "couldn't", "wouldn't", "shouldn't", "ain't",
    "i'm", "i've", "i'll", "i'd", "it's", "that's", "you're", "we're", "they're",
    "what's", "how's", "there's", "let's", "gonna", "wanna", "gotta", "kinda",
    "sorta", "dunno", "y'all",
    # interjections / reactions
    " lol", " lmao", " haha", " hahaha", " oof", " ugh", " hmm", " huh", " oh",
    " ah", " aw", " eh", " yeah", " yep", " yup", " nah", " nope", " ok", " okay",
    " damn", " dang", " wow", " nice", " sweet", " oops", " hey", " hi", " hello",
    " bye", " yikes", " welp", " meh", " woah", " whoa", " ouch", " phew",
    # small-talk phrases
    " what's up", " how are you", " how's it going", " what happened",
    " that sucks", " that's rough", " for real", " no way", " same here",
    " good luck", " take care", " talk later", " see you", " thanks",
    " honestly", " actually", " basically", " literally", " probably",
    " i guess", " i mean", " you know", " right?", " really?",
    " tell me more", " no idea", " not sure", " makes sense", " fair enough",
    " sounds good", " sounds fun", " long day", " tired", " busy", " work",
    " school", " sleep", " tomorrow", " tonight", " weekend", " today",
    # punctuation / typography
    "...", "!!", "??", "?!", " :)", " :(", " <3", "--", "’", "…",
]


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class SmallTalkTokenizer:
    """Thin wrapper over a HuggingFace `tokenizers` ByteLevel BPE model."""

    def __init__(self, backend):
        self._tk = backend
        self.specials = {t: self._tk.token_to_id(t) for t in SPECIAL_TOKENS}
        missing = [t for t, i in self.specials.items() if i is None]
        if missing:
            raise ValueError(f"tokenizer is missing special tokens: {missing}")
        self.pad_id = self.specials[PAD]
        self.bos_id = self.specials[BOS]
        self.eos_id = self.specials[EOS]
        self.system_id = self.specials[SYSTEM]
        self.user_id = self.specials[USER]
        self.assistant_id = self.specials[ASSISTANT]
        self.endofturn_id = self.specials[ENDOFTURN]

    # ---- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "SmallTalkTokenizer":
        from tokenizers import Tokenizer

        p = Path(path)
        f = p / "tokenizer.json" if p.is_dir() else p
        return cls(Tokenizer.from_file(str(f)))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._tk.save(str(p / "tokenizer.json"))
        (p / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "vocab_size": self.vocab_size,
                    "special_tokens": self.specials,
                    "model_type": "byte_level_bpe",
                    "chat_template": CHAT_TEMPLATE_DOC,
                },
                indent=2,
            )
        )
        return p

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    # ---- core encode/decode ------------------------------------------------
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = self._tk.encode(text).ids
        return [self.bos_id] + ids if add_special else ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return self._tk.decode(list(ids), skip_special_tokens=skip_special)

    def id_to_token(self, i: int) -> str:
        return self._tk.id_to_token(i)

    # ---- chat template -----------------------------------------------------
    def encode_message(self, msg: Message | dict) -> list[int]:
        m = msg if isinstance(msg, Message) else Message(msg["role"], msg["content"])
        if m.role not in ROLE_TOKENS:
            raise ValueError(f"unknown role {m.role!r}")
        head = self.specials[ROLE_TOKENS[m.role]]
        return [head] + self.encode(m.content.strip()) + [self.endofturn_id]

    def encode_conversation(
        self,
        messages: Iterable[Message | dict],
        add_bos: bool = True,
        add_eos: bool = False,
        add_generation_prompt: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Returns (ids, assistant_mask).

        `assistant_mask[i] == 1` exactly for assistant *content* tokens and the
        assistant's <|endofturn|>. Role headers, system and user tokens get 0 --
        this is the stage-2 SFT loss mask.
        """
        ids: list[int] = []
        mask: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
            mask.append(0)
        for msg in messages:
            m = msg if isinstance(msg, Message) else Message(msg["role"], msg["content"])
            toks = self.encode_message(m)
            is_asst = m.role == "assistant"
            ids.extend(toks)
            # header token is context (predicting it is not the assistant's job)
            mask.extend([0] + [1 if is_asst else 0] * (len(toks) - 1))
        if add_generation_prompt:
            ids.append(self.assistant_id)
            mask.append(0)
        if add_eos:
            ids.append(self.eos_id)
            mask.append(1 if ids else 0)
        return ids, mask

    def render(self, messages: Iterable[Message | dict], add_generation_prompt: bool = False) -> str:
        """Human-readable rendering of the chat template (debugging / docs)."""
        out = []
        for msg in messages:
            m = msg if isinstance(msg, Message) else Message(msg["role"], msg["content"])
            out.append(f"{ROLE_TOKENS[m.role]}{m.content.strip()}{ENDOFTURN}")
        if add_generation_prompt:
            out.append(ASSISTANT)
        return "".join(out)


CHAT_TEMPLATE_DOC = (
    "<|bos|>" "{<|system|>|<|user|>|<|assistant|>}content<|endofturn|> ... "
    "generation prompt = trailing <|assistant|>; stop on <|endofturn|> or <|eos|>"
)


# ---- training ---------------------------------------------------------------
def iter_training_text(conversations: Iterable[Sequence[dict]]) -> Iterator[str]:
    """Yield raw utterance text (no special tokens) for BPE training."""
    for conv in conversations:
        for m in conv:
            content = (m.get("content") or "").strip()
            if content:
                yield content


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    out_dir: str | Path,
    min_frequency: int = 2,
    always_keep: Sequence[str] | None = None,
    pad_to_vocab_size: bool = True,
) -> SmallTalkTokenizer:
    """Train a byte-level BPE tokenizer of exactly `vocab_size` entries.

    `pad_to_vocab_size` appends unused `<|reserved_N|>` tokens when the corpus
    cannot fill the target. Vocabulary size is an *architectural budget* in this
    study (it sets the embedding parameter cost), so it must not silently shrink
    with corpus size -- otherwise parameter counts stop matching their targets and
    the scaling comparison is no longer matched. Reserved rows cost embedding
    params but are never emitted, which is exactly the cost we intend to measure.
    """
    from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
    from tokenizers.models import BPE

    tk = Tokenizer(BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()

    keep = list(ALWAYS_KEEP if always_keep is None else always_keep)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    # Repeat the curated atoms so the BPE merge search reliably forms them.
    corpus = list(texts) + keep * 64
    tk.train_from_iterator(corpus, trainer=trainer)

    if pad_to_vocab_size:
        from tokenizers import AddedToken

        shortfall = vocab_size - tk.get_vocab_size()
        if shortfall > 0:
            tk.add_tokens(
                [
                    AddedToken(f"<|reserved_{i}|>", special=True, normalized=False)
                    for i in range(shortfall)
                ]
            )

    tokenizer = SmallTalkTokenizer(tk)
    tokenizer.save(out_dir)
    return tokenizer
