"""Cleaning, normalisation, filtering, deduplication and splitting.

The filters here *are* the core scientific instrument of this project: they define
what "conversationally focused training" means, which is the independent variable
in research question 5. Every rule is a named, individually toggleable predicate
and every drop is counted so the corpus report is auditable.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

from .schema import Conversation, Turn

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
_WS = re.compile(r"[ \t ]+")
_NEWLINES = re.compile(r"\n{2,}")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:%])")
_SPACE_IN_CONTRACTION = re.compile(r"\b(\w+)\s+'\s*(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_DUP_PUNCT = re.compile(r"([!?.,])\1{3,}")
_SPEAKER_PREFIX = re.compile(r"^\s*(?:user|assistant|human|ai|bot|a|b)\s*[:>]\s*", re.IGNORECASE)
_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}


def normalize_text(text: str, strip_urls: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    for a, b in _QUOTES.items():
        text = text.replace(a, b)
    if strip_urls:
        text = _URL.sub("<link>", text)
        text = _EMAIL.sub("<email>", text)
    text = _SPEAKER_PREFIX.sub("", text)
    text = _SPACE_IN_CONTRACTION.sub(r"\1'\2", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _DUP_PUNCT.sub(r"\1\1\1", text)
    text = _NEWLINES.sub("\n", text)
    text = _WS.sub(" ", text)
    return text.strip()


def normalize_conversation(conv: Conversation, strip_urls: bool = True) -> Conversation:
    msgs = []
    for m in conv.messages:
        content = normalize_text(m.content, strip_urls=strip_urls)
        if content:
            msgs.append(Turn(m.role, content))
    conv.messages = merge_consecutive(msgs)
    return conv


def merge_consecutive(messages: Sequence[Turn]) -> list[Turn]:
    """Preserve dialogue boundaries by collapsing same-role runs into one turn."""
    out: list[Turn] = []
    for m in messages:
        if out and out[-1].role == m.role and m.role != "system":
            sep = " " if out[-1].content.endswith((".", "!", "?", ",")) else ". "
            out[-1] = Turn(m.role, f"{out[-1].content}{sep}{m.content}")
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Filter vocabulary: what we are deliberately removing
# ---------------------------------------------------------------------------
AI_ASSISTANT_PHRASES = [
    "as an ai", "as a language model", "i'm an ai", "i am an ai",
    "i don't have personal", "i do not have personal", "i cannot provide",
    "i can't provide", "it's important to note", "it is important to note",
    "it's worth noting", "here are some", "here's a list", "in conclusion",
    "i hope this helps", "let me know if you have any", "feel free to ask",
    "certainly!", "of course!", "great question", "i'd be happy to help",
    "i'm here to help", "as your assistant", "please note that",
    "there are several", "firstly,", "secondly,", "furthermore,",
    "in summary", "to summarize", "keep in mind that", "disclaimer",
    "consult a professional", "consult a doctor", "seek professional",
    "i apologize for any confusion", "step 1", "step-by-step",
]

FACTUAL_QA_MARKERS = [
    "the capital of", "was invented by", "is defined as", "refers to the",
    "according to the", "in the year", "the formula for", "the theory of",
    "is a type of", "consists of the following", "the process by which",
    "photosynthesis", "mitochondria", "world war", "the periodic table",
]

# Code detection must not fire on ordinary small talk. "class was fun today" and
# "i had to print my ticket" are on-domain; naive substrings like "class " or
# "print(" without context silently delete school/errand conversations.
CODE_PATTERNS = [
    r"```",                                  # fenced block
    r"\bdef\s+\w+\s*\(",                     # def foo(
    r"\bclass\s+[A-Z]\w*\s*[:(]",            # class Foo:
    r"\b(?:import|from)\s+\w+\s+(?:import|as)\b",
    r"\b(?:console\.log|System\.out|printf|puts)\s*\(",
    r"\bprint\s*\([\"'\w]",                  # print("x")  not "print my ticket"
    r"</\w+>|<\w+\s*/>",                     # markup tags
    r"\bSELECT\b.+\bFROM\b",
    r"#include\s*<",
    r"\b(?:npm|pip|apt|brew|cargo)\s+install\b",
    r"\bgit\s+(?:commit|clone|push|checkout)\b",
    r"=>\s*[\{\(]|\bfunction\s*\(",
    r"\}\s*\)\s*;|\{\s*$",
    r"^\s*(?:\$|>>>|sudo)\s+\S",
    r"\bfor\s*\(\s*\w+\s*=|\bwhile\s*\(",
]
_CODE_RE = re.compile("|".join(CODE_PATTERNS), re.MULTILINE)

# Kept for backwards compatibility / documentation of intent.
CODE_MARKERS = CODE_PATTERNS

# Same care for maths: "i have to calculate my taxes" is small talk; "solve for x"
# is not. Require the textbook framing, not a bare arithmetic verb.
MATH_PATTERNS = [
    r"\bsolve for\b", r"\bderivative of\b", r"\bintegral of\b",
    r"\bsquare root of\b", r"\b\w+'?s? theorem\b", r"\bcalculate the \w+ of\b",
    r"\b[a-z]\s*=\s*-?\d", r"\bsum of the\b", r"\bequation\b",
    r"\b\d+\s*[\+\-\*/x]\s*\d+\s*=", r"\bmultiply(?: \w+)? by\b",
    r"\bdivide(?:d)? by\b",
]
_MATH_RE = re.compile("|".join(MATH_PATTERNS))
MATH_MARKERS = MATH_PATTERNS

INSTRUCTION_MARKERS = [
    "write an essay", "summarize the following", "translate the following",
    "rewrite the", "generate a list", "explain in detail", "act as a",
    "you are a helpful", "follow these instructions", "output the following",
    "in bullet points", "in json", "step by step",
]

LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)


def _has_any(text: str, needles: Iterable[str]) -> bool:
    low = text.lower()
    return any(n in low for n in needles)


# ---------------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------------
@dataclass
class FilterConfig:
    min_turns: int = 2
    max_turns: int = 40
    min_chars: int = 1
    max_turn_words: int = 60          # a 100-word monologue is not small talk
    max_assistant_words: int = 45
    target_assistant_words: tuple[int, int] = (1, 25)
    max_mean_assistant_words: float = 30.0
    require_assistant_turn: bool = True
    require_alternating: bool = False
    drop_ai_assistant_style: bool = True
    drop_factual_qa: bool = True
    drop_code: bool = True
    drop_math: bool = True
    drop_instructions: bool = True
    drop_lists: bool = True
    max_list_lines: int = 2
    max_non_ascii_ratio: float = 0.05  # English-focused corpus
    min_alpha_ratio: float = 0.5
    downweight_instead_of_drop: bool = False  # keep-with-weight mode for ablations
    dedup: bool = True
    near_dedup_ngram: int = 8

    @classmethod
    def permissive(cls) -> "FilterConfig":
        """Ablation control: generic-LM style filtering (research question 5)."""
        return cls(
            max_turn_words=1000,
            max_assistant_words=1000,
            max_mean_assistant_words=1e9,
            drop_ai_assistant_style=False,
            drop_factual_qa=False,
            drop_code=False,
            drop_math=False,
            drop_instructions=False,
            drop_lists=False,
            max_non_ascii_ratio=1.0,
            min_alpha_ratio=0.0,
        )


@dataclass
class FilterStats:
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    weights: list[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.kept + sum(self.dropped.values())

    def report(self) -> str:
        lines = [f"kept {self.kept:,} / {self.total:,} conversations"]
        for reason, n in self.dropped.most_common():
            lines.append(f"  dropped {n:>8,}  {reason}")
        return "\n".join(lines)


def _reject_reason(conv: Conversation, cfg: FilterConfig) -> str | None:
    msgs = [m for m in conv.messages if m.role != "system"]
    if len(msgs) < cfg.min_turns:
        return "too_few_turns"
    if len(msgs) > cfg.max_turns:
        return "too_many_turns"
    if cfg.require_assistant_turn and not any(m.role == "assistant" for m in msgs):
        return "no_assistant_turn"
    if cfg.require_alternating and not conv.alternates():
        return "not_alternating"

    full = " ".join(m.content for m in msgs)
    if not full.strip():
        return "empty"
    if len(full) < cfg.min_chars:
        return "too_short"

    alpha = sum(c.isalpha() or c.isspace() or c in ".,!?'" for c in full)
    if alpha / max(len(full), 1) < cfg.min_alpha_ratio:
        return "low_alpha_ratio"
    non_ascii = sum(ord(c) > 127 and not _is_emoji(c) for c in full)
    if non_ascii / max(len(full), 1) > cfg.max_non_ascii_ratio:
        return "non_english"

    asst_words = [len(m.content.split()) for m in msgs if m.role == "assistant"]
    for m in msgs:
        n = len(m.content.split())
        # Report the assistant-specific reason first: assistant verbosity is the
        # failure mode we care about, so it should not be masked by the generic cap.
        if m.role == "assistant" and n > cfg.max_assistant_words:
            return "assistant_turn_too_long"
        if n > cfg.max_turn_words:
            return "turn_too_long"
    if asst_words and sum(asst_words) / len(asst_words) > cfg.max_mean_assistant_words:
        return "assistant_too_verbose"

    # Style/knowledge filters judge the ASSISTANT's turns only. A user who asks
    # "what's the capital of kyrgyzstan" is precisely the context we want the model
    # to answer socially ("no idea lol") -- dropping it would delete the graceful-
    # ignorance training signal we are explicitly trying to create. Only the
    # assistant *answering* factually is off-target.
    assistant_text = " ".join(m.content for m in msgs if m.role == "assistant")
    if cfg.drop_ai_assistant_style and _has_any(assistant_text, AI_ASSISTANT_PHRASES):
        return "ai_assistant_style"
    if cfg.drop_code and _CODE_RE.search(full):
        return "code"          # code from either side is off-domain
    if cfg.drop_math and _MATH_RE.search(assistant_text.lower()):
        return "math"
    if cfg.drop_instructions and _has_any(full, INSTRUCTION_MARKERS):
        return "instruction_following"
    if cfg.drop_factual_qa and _has_any(assistant_text, FACTUAL_QA_MARKERS):
        return "factual_qa"
    if cfg.drop_lists and len(LIST_LINE.findall(full)) > cfg.max_list_lines:
        return "list_formatting"
    return None


def _is_emoji(c: str) -> bool:
    return unicodedata.category(c) == "So" or 0x1F300 <= ord(c) <= 0x1FAFF


def conversation_weight(conv: Conversation, cfg: FilterConfig) -> float:
    """Soft downweighting for conversations that are on-domain but off-style."""
    w = 1.0
    asst = [len(m.content.split()) for m in conv.assistant_turns]
    if asst:
        lo, hi = cfg.target_assistant_words
        in_band = sum(lo <= n <= hi for n in asst) / len(asst)
        w *= 0.35 + 0.65 * in_band
    reason = _reject_reason(conv, cfg)
    if reason:
        w *= 0.1
    return round(w, 4)


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _shingles(text: str, n: int) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def clean_conversations(
    conversations: Iterable[Conversation],
    cfg: FilterConfig | None = None,
    extra_predicates: Sequence[Callable[[Conversation], str | None]] = (),
) -> tuple[list[Conversation], FilterStats]:
    cfg = cfg or FilterConfig()
    stats = FilterStats()
    seen_exact: set[str] = set()
    seen_shingles: set[str] = set()
    kept: list[Conversation] = []

    for conv in conversations:
        conv = normalize_conversation(conv)
        reason = _reject_reason(conv, cfg)
        if reason is None:
            for pred in extra_predicates:
                reason = pred(conv)
                if reason:
                    break

        if reason and not cfg.downweight_instead_of_drop:
            stats.dropped[reason] += 1
            continue

        body = conv.text()
        if cfg.dedup:
            h = _hash(re.sub(r"\W+", " ", body.lower()).strip())
            if h in seen_exact:
                stats.dropped["exact_duplicate"] += 1
                continue
            seen_exact.add(h)
            # Cheap near-dup: drop if *every* long shingle was already seen.
            sh = _shingles(body, cfg.near_dedup_ngram)
            if sh and len(sh) > 3 and sh <= seen_shingles:
                stats.dropped["near_duplicate"] += 1
                continue
            seen_shingles |= sh

        weight = conversation_weight(conv, cfg) if cfg.downweight_instead_of_drop else 1.0
        if weight != 1.0:
            conv.meta["weight"] = weight
        if reason:
            conv.meta["filter_flag"] = reason
        stats.weights.append(weight)
        kept.append(conv)
        stats.kept += 1

    return kept, stats


def split_train_val(
    conversations: Sequence[Conversation],
    val_ratio: float = 0.02,
    seed: int = 1337,
    min_val: int = 1,
    group_by_source: bool = True,
) -> tuple[list[Conversation], list[Conversation]]:
    """Deterministic hash-based split. Stratified by source so every corpus is
    represented in validation, and stable when new data is appended."""
    rng = random.Random(seed)
    buckets: dict[str, list[Conversation]] = {}
    for c in conversations:
        buckets.setdefault(c.source if group_by_source else "all", []).append(c)

    train: list[Conversation] = []
    val: list[Conversation] = []
    for _, items in sorted(buckets.items()):
        items = sorted(items, key=lambda c: _hash(c.id))
        k = max(min_val, int(round(len(items) * val_ratio))) if len(items) > 2 else 0
        val.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def corpus_stats(conversations: Sequence[Conversation]) -> dict[str, float | int | dict]:
    turns = [c.num_turns for c in conversations]
    asst = [len(m.content.split()) for c in conversations for m in c.assistant_turns]
    sources = Counter(c.source for c in conversations)
    asst_sorted = sorted(asst)

    def pct(p: float) -> float:
        if not asst_sorted:
            return 0.0
        return float(asst_sorted[min(len(asst_sorted) - 1, int(p * len(asst_sorted)))])

    return {
        "conversations": len(conversations),
        "sources": dict(sources),
        "mean_turns": round(sum(turns) / max(len(turns), 1), 2),
        "assistant_turns": len(asst),
        "assistant_words_mean": round(sum(asst) / max(len(asst), 1), 2),
        "assistant_words_p50": pct(0.50),
        "assistant_words_p90": pct(0.90),
        "assistant_words_in_3_25": round(
            sum(3 <= n <= 25 for n in asst) / max(len(asst), 1), 4
        ),
    }


# ---------------------------------------------------------------------------
# Family-level splitting
# ---------------------------------------------------------------------------
def split_by_family(
    conversations: "Sequence[Conversation]",
    val_families: float = 0.15,
    test_families: float = 0.15,
    seed: int = 1337,
    family_key: str = "family",
) -> tuple[list, list, list]:
    """Split so that TRAIN / VAL / TEST never share a generator grammar.

    Measured failure this fixes: with example-level splitting, 83.5% of validation
    assistant utterances appeared VERBATIM in training, because both splits were
    drawn from the same template family. `val_loss` was then measuring
    memorisation of a grammar, not generalisation -- it sat at 0.205 for five
    consecutive rounds while behaviour visibly changed.

    Conversations lacking a family tag fall back to their `source`, so real
    corpora (DailyDialog etc.) are split as one family each.
    """
    by_family: dict[str, list] = {}
    for c in conversations:
        fam = str(c.meta.get(family_key) or c.source or "unknown")
        by_family.setdefault(fam, []).append(c)

    # Stable hash partitioning is append-safe: adding a new family never moves
    # an old one between train/validation/test.  This is critical for experiment
    # comparability as generated corpora grow.
    if not 0 <= val_families < 1 or not 0 <= test_families < 1 or val_families + test_families >= 1:
        raise ValueError("family split ratios must be non-negative and sum to < 1")
    val_f, test_f = set(), set()
    for fam in by_family:
        bucket = int(_hash(f"{seed}:{fam}")[:16], 16) / float(16**16)
        if bucket < test_families:
            test_f.add(fam)
        elif bucket < test_families + val_families:
            val_f.add(fam)

    train, val, test = [], [], []
    for fam, items in by_family.items():
        (val if fam in val_f else test if fam in test_f else train).extend(items)
    # Stable ordering prevents a new append from perturbing an existing split.
    for part in (train, val, test):
        part.sort(key=lambda c: _hash(f"{seed}:{c.id}"))
    return train, val, test


def family_report(conversations: "Sequence[Conversation]", family_key: str = "family") -> dict:
    fams: dict[str, int] = {}
    for c in conversations:
        fam = str(c.meta.get(family_key) or c.source or "unknown")
        fams[fam] = fams.get(fam, 0) + 1
    return dict(sorted(fams.items(), key=lambda kv: -kv[1]))
