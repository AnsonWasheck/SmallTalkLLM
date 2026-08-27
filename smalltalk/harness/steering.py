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


# Words that carry no referent. Kept here rather than imported from the
# benchmark so the harness never depends on a test module.
_STOP = {
    "i", "you", "it", "that", "this", "the", "a", "an", "is", "was", "are", "were",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "so", "my", "your",
    "me", "we", "they", "he", "she", "them", "him", "her", "do", "did", "does",
    "have", "has", "had", "be", "been", "got", "get", "no", "not", "yeah", "yes",
    "oh", "ah", "well", "how", "what", "when", "where", "who", "why", "up", "out",
    "all", "just", "really", "very", "with", "about", "like", "if", "there",
    "here", "again", "still", "too", "then", "now", "one", "new", "much",
}
_WORD_RE = __import__("re").compile(r"[a-z']+")
REF_TOKEN = "<|ref|>"


def referent_tokens(text: str, tokenizer, max_words: int = 3) -> dict[int, float]:
    """Token ids for the content words the user just said.

    Pulling the noun out of "i got a new dog" is exact string work, and the
    founding principle of this project is that the 6.7M model should not spend
    capacity on operations ordinary software performs perfectly. The harness
    extracts the referent; the model still decides whether and how to use it.

    Returns a flat bias map -- later words are weighted slightly higher, since
    the thing being talked about tends to arrive at the end of a short turn.
    """
    ws = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2]
    if not ws:
        return {}
    ws = ws[-max_words:]
    bias: dict[int, float] = {}
    for i, w in enumerate(ws):
        weight = 0.6 + 0.4 * (i + 1) / len(ws)
        for form in (" " + w, w):
            for tid in tokenizer.encode(form):
                bias[tid] = max(bias.get(tid, 0.0), weight)
    return bias


def render_ref(reply: str, user_text: str, fallback: str = "") -> tuple[str, bool]:
    """Replace the model's <|ref|> placeholder with the user's actual words.

    The model has marked WHERE the referent belongs; extracting WHICH word that
    is, is exact string work. This is the same division as everywhere else in the
    harness, made explicit in the vocabulary instead of guessed at afterwards.

    Unlike substitute_referent, this never has to decide WHETHER to fire: the
    model asked for a referent. If the user's turn contains no usable content
    word, the reply is discarded rather than rendered with a wrong noun, because
    a well-formed question about nothing is worse than a generic reaction.
    """
    if REF_TOKEN not in reply:
        return reply, False
    ws = [w for w in _WORD_RE.findall(user_text.lower())
          if w not in _STOP and len(w) > 2]
    if not ws:
        return fallback, False

    # Prefer a known bank noun over positional guessing. Taking the last content
    # word gave "my sister is visiting" -> "how's your visiting doing?": the
    # referent is the subject, not the final word. Bank membership identifies it
    # exactly when the subject is one we know; position is only the fallback.
    from ..core.frame_gen import BANKS

    known = {w for bank in BANKS.values() for w in bank}
    picked = next((w for w in ws if w in known), ws[-1])

    out = reply.replace(REF_TOKEN, picked)
    # "a axolotl" -> "an axolotl". The model cannot know the article in advance
    # because it does not know the word; the harness does.
    out = __import__("re").sub(r"\ba (?=[aeiou])", "an ", out)
    return out, True


def substitute_referent(reply: str, user_text: str, banks: dict[str, list[str]]
                        ) -> tuple[str, bool]:
    """Replace a hallucinated bank noun with the one the user actually said.

    Measured on r009: the model produces the right FRAME almost every time and
    then fills the slot with a different member of the same bank --
    "i'm a nurse" -> "do you like being a bricklayer?",
    "we went to italy" -> "how was iceland?".

    It cannot do better on its own: elaboration succeeded on 11% of one-token
    referents and 0 of 29 multi-token ones, and only 28 of 402 nouns in the banks
    are single-token. Copying a multi-token span is beyond eight layers with one
    KV head, and restricting the curriculum to copyable nouns would shrink the
    bank back to a memorisable size.

    So the model decides WHETHER to elaborate and in WHAT shape; the harness
    supplies WHICH word, exactly. That is the division this project is built on,
    and unlike the earlier biasing attempt there is now a slot to put the word
    into.

    Returns (reply, changed). Conservative by construction: it fires only when
    the reply names a bank noun the user did not say AND the user's turn contains
    a noun from that same bank, so it can substitute like for like.
    """
    r_words = _WORD_RE.findall(reply.lower())
    u_words = _WORD_RE.findall(user_text.lower())
    if not r_words or not u_words:
        return reply, False

    for slot, bank in banks.items():
        bank_set = set(bank)
        said = [w for w in u_words if w in bank_set]
        if not said:
            continue
        wrong = [w for w in r_words if w in bank_set and w not in said]
        if not wrong:
            continue
        # Preserve the reply's own casing and punctuation; swap the word only.
        pattern = __import__("re").compile(rf"\b{__import__('re').escape(wrong[0])}\b",
                                           __import__("re").IGNORECASE)
        return pattern.sub(said[-1], reply, count=1), True
    return reply, False


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
