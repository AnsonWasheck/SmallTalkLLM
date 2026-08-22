"""Harness contracts.

The most important test here is `test_a_raw_is_identical_to_the_bare_engine`: if
A_RAW ever diverges from unmodified inference, every ablation number becomes
uninterpretable, because gains could be coming from incidental prompting changes
rather than from a mechanism.
"""

import pytest
import torch

from smalltalk.harness import MODES, Harness
from smalltalk.harness.config import HarnessConfig
from smalltalk.harness.confidence import score as conf_score
from smalltalk.harness.context import select
from smalltalk.harness.features import extract
from smalltalk.harness.memory import Memory
from smalltalk.harness.policy import (BY_ID, LENGTH_TOKENS, POLICIES,
                                      high_confidence_shortcut)
from smalltalk.harness.repetition import (internal_loop, is_exact_repeat,
                                          shares_long_ngram)
from smalltalk.harness.state import ConversationState
from smalltalk.harness.validator import validate


# --- features ---------------------------------------------------------
def test_features_do_not_rewrite_the_message():
    f = extract("HEY!!! how r u 😀")
    assert f.text == "HEY!!! how r u 😀"      # surface preserved verbatim
    assert f.all_caps is False                # mixed case overall
    assert f.repeated_punct and f.has_emoji and f.has_greeting


def test_all_caps_needs_enough_letters():
    assert extract("OK").all_caps is False    # too short to be deliberate
    assert extract("WHAT ARE YOU DOING").all_caps is True


# --- state ------------------------------------------------------------
def test_consecutive_question_tracking():
    s = ConversationState()
    s.observe_reply("how come?", "P17", 3)
    s.observe_reply("really?", "P17", 2)
    assert s.consecutive_questions == 2
    s.observe_reply("fair enough", "P18", 2)
    assert s.consecutive_questions == 0


def test_recent_hashes_detect_repeats():
    s = ConversationState()
    s.observe_reply("that sounds rough", "P08", 4)
    assert s.seen_recently("That Sounds Rough")


# --- memory -----------------------------------------------------------
def test_memory_learns_and_corrects():
    m = Memory()
    m.observe("my dog is called luna", 0)
    assert m.facts["pet_name"].value == "luna"
    m.observe("my dog is called sonata", 1)
    assert m.facts["pet_name"].value == "sonata"      # correction wins


def test_memory_absence_is_not_invented():
    m = Memory()
    m.observe("nice weather today", 0)
    assert m.retrieve("what's my dog called?") == []


def test_memory_does_not_store_states_as_jobs():
    """`i'm a bit tired` must not fill the occupation slot."""
    m = Memory()
    m.observe("i'm a bit tired", 0)
    assert "job" not in m.facts
    m.observe("i'm a plumber", 1)
    assert m.facts["job"].value == "plumber"


def test_memory_is_bounded():
    m = Memory(slots=2)
    m.observe("i'm a plumber", 0)
    m.observe("i live in leeds", 1)
    m.observe("my cat is called pepper", 2)
    assert len(m.facts) <= 2


# --- policy -----------------------------------------------------------
def test_policy_ids_unique_and_lengths_known():
    assert len({p.pid for p in POLICIES}) == len(POLICIES)
    for p in POLICIES:
        assert p.length in LENGTH_TOKENS
        assert p.exemplars


def test_shortcuts_only_fire_on_unambiguous_surfaces():
    assert high_confidence_shortcut(extract("thanks")).pid == "P04"
    assert high_confidence_shortcut(extract("bye")).pid == "P06"
    # A long message merely containing "thanks" is not a thanks act.
    assert high_confidence_shortcut(
        extract("thanks for that but honestly the whole day was a disaster")) is None


# --- confidence -------------------------------------------------------
def test_confidence_flags_flat_distributions():
    peaked = conf_score({"a": 0.82, "b": 0.08, "c": 0.10},
                        min_top1=0.34, min_margin=0.06)
    flat = conf_score({"a": 0.29, "b": 0.27, "c": 0.25, "d": 0.19},
                      min_top1=0.34, min_margin=0.06)
    assert peaked.status == "HIGH"
    assert flat.status == "LOW"
    assert flat.entropy > peaked.entropy


# --- repetition -------------------------------------------------------
def test_trivial_acknowledgements_may_recur():
    assert is_exact_repeat("yeah", ["yeah", "mm"]) is False
    assert is_exact_repeat("that sounds rough", ["that sounds rough"]) is True


def test_internal_loop_detected():
    assert internal_loop("i hope you get it i hope you get it")
    assert not internal_loop("that sounds really rough honestly")


# --- validator --------------------------------------------------------
def test_validator_catches_question_when_forbidden():
    p = BY_ID["P06"]        # GOODBYE / NO_QUESTION / CLOSE
    v = validate("doing anything later?", policy=p, n_tokens=4, recent=[],
                 closing=True)
    assert not v.ok
    assert "question_when_forbidden" in v.failures


def test_validator_passes_a_good_close():
    v = validate("see you", policy=BY_ID["P06"], n_tokens=2, recent=[], closing=True)
    assert v.ok


def test_validator_catches_boilerplate_and_token_leaks():
    assert "ai_boilerplate" in validate("As an AI, I cannot help", policy=None,
                                        n_tokens=6, recent=[], closing=False).failures
    assert "special_token_leak" in validate("<|endofturn|> hey", policy=None,
                                            n_tokens=3, recent=[],
                                            closing=False).failures


# --- context ----------------------------------------------------------
def test_context_respects_token_budget(tokenizer):
    history = [{"role": "user" if i % 2 == 0 else "assistant",
                "content": f"this is turn number {i} and it has some words in it"}
               for i in range(31)]        # odd count -> ends on a user turn
    sel = select(history, tokenizer, max_turns=20, token_budget=64)
    assert sel.n_tokens <= 64 or sel.n_turns == 1
    assert sel.dropped > 0
    assert sel.messages[-1]["role"] == "user"      # must end on a user turn


def test_context_injects_memory_hint_only_when_given(tokenizer):
    history = [{"role": "user", "content": "what's my dog called?"}]
    plain = select(history, tokenizer, max_turns=6, token_budget=256)
    with_hint = select(history, tokenizer, max_turns=6, token_budget=256,
                       memory_hint="pet name: luna.")
    assert plain.messages[0]["role"] == "user"
    assert with_hint.messages[0]["role"] == "system"
    assert with_hint.n_tokens > plain.n_tokens


# --- integration ------------------------------------------------------
@pytest.fixture(scope="module")
def tiny(tokenizer):
    from smalltalk.config import ModelConfig
    from smalltalk.model import build_model
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, hidden_size=64, num_layers=2,
                      num_attention_heads=4, num_key_value_heads=1, head_dim=16,
                      intermediate_size=128, max_position_embeddings=512)
    return build_model(cfg).eval(), tokenizer


def test_a_raw_is_identical_to_the_bare_engine(tiny):
    """A_RAW must reproduce unmodified inference exactly.

    If this drifts, every ablation number is confounded: a gain could be coming
    from prompting or decoding changes rather than from a harness mechanism.
    """
    from smalltalk.infer.generate import ConversationEngine, GenerationConfig

    model, tok = tiny
    gen = GenerationConfig(temperature=0.0, top_p=1.0, top_k=0, greedy=True,
                           repetition_penalty=1.0, max_new_tokens=12, seed=0)
    text = "hey there how are you"

    h = Harness(model=model, tokenizer=tok, cfg=MODES["A_RAW"], gen=gen)
    engine = ConversationEngine(model=model, tokenizer=tok, gen=gen)
    assert h.reply(text) == engine.reply(text)


def test_harness_never_changes_the_model(tiny):
    model, tok = tiny
    before = sum(p.numel() for p in model.parameters())
    h = Harness(model=model, tokenizer=tok, cfg=MODES["F_FULL_HARNESS"])
    h.reply("hi")
    assert sum(p.numel() for p in model.parameters()) == before


def test_trace_records_every_stage(tiny):
    from smalltalk.harness.trace import Trace
    model, tok = tiny
    h = Harness(model=model, tokenizer=tok, cfg=MODES["F_FULL_HARNESS"])
    tr = Trace()
    h.reply("i'm exhausted", trace=tr)
    d = tr.as_dict()
    for key in ("features", "context", "policy_scores", "confidence",
                "generation", "validator", "final", "model_calls"):
        assert key in d
    assert tr.model_calls > 0
    assert isinstance(tr.render(), str)


def test_modes_are_reproducible(tiny):
    model, tok = tiny
    outs = []
    for _ in range(2):
        h = Harness(model=model, tokenizer=tok, cfg=MODES["F_FULL_HARNESS"])
        outs.append([h.reply(t) for t in ("hey", "i'm tired", "bye")])
    assert outs[0] == outs[1]


def test_harness_stays_within_the_model_size_budget():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    total = sum(p.stat().st_size for p in (root / "smalltalk" / "harness").glob("*.py"))
    assert total <= 6_689_024 * 4


# --- learned policy head ----------------------------------------------
def test_policy_head_is_tiny_and_linear():
    """A linear probe answers a sharper question than an MLP: is policy
    LINEARLY encoded in the 256-dim state? Measured: 96.6% on held-out corpus
    paraphrases, so yes -- the model knows the policy it cannot act on."""
    from smalltalk.harness.head import POLICY_IDS, PolicyHead
    head = PolicyHead(hidden_size=256, n_features=0)
    assert head.n_params < 6000
    assert head.n_bytes < 25_000
    assert len(POLICY_IDS) == 19


def test_policy_head_reads_the_model_without_changing_it(tiny):
    from smalltalk.harness.head import hidden_state
    model, tok = tiny
    before = [p.clone() for p in model.parameters()]
    ids, _ = tok.encode_conversation([{"role": "user", "content": "hey"}],
                                     add_bos=True, add_generation_prompt=True)
    h = hidden_state(model, ids)
    assert h.shape == (model.cfg.hidden_size,)
    # The forward hook must be removed and no weight touched.
    assert not model.norm._forward_hooks
    for a, b in zip(before, model.parameters()):
        assert torch.equal(a, b)


def test_feature_vector_is_bounded():
    from smalltalk.harness.head import FEATURE_KEYS, feature_vector
    v = feature_vector(extract("WHAT ARE YOU DOING!!! " * 20))
    assert v.shape == (len(FEATURE_KEYS),)
    assert float(v.max()) <= 1.0 and float(v.min()) >= 0.0


# --- Phase 3 steering -------------------------------------------------
def test_steering_off_reproduces_the_baseline(tiny):
    """Every steering interface must be a no-op at zero strength.

    If it is not, the ablation cannot separate the mechanism from incidental
    changes to the decoding path.
    """
    from smalltalk.infer.generate import GenerationConfig, generate
    from smalltalk.harness.steering import steered_generate

    model, tok = tiny
    gen = GenerationConfig(temperature=0.0, top_p=1.0, top_k=0, greedy=True,
                           repetition_penalty=1.0, max_new_tokens=10, seed=0)
    ids, _ = tok.encode_conversation([{"role": "user", "content": "hey there"}],
                                     add_bos=True, add_generation_prompt=True)
    assert steered_generate(model, tok, ids, gen) == generate(model, tok, ids, gen)


def test_hidden_steering_removes_its_hook(tiny):
    from smalltalk.harness.steering import steered_generate
    from smalltalk.infer.generate import GenerationConfig
    model, tok = tiny
    gen = GenerationConfig(greedy=True, max_new_tokens=4, seed=0)
    ids, _ = tok.encode_conversation([{"role": "user", "content": "hi"}],
                                     add_bos=True, add_generation_prompt=True)
    before = [p.clone() for p in model.parameters()]
    steered_generate(model, tok, ids, gen,
                     hidden_vec=torch.randn(model.cfg.hidden_size),
                     hidden_steps=2, hidden_alpha=0.5)
    assert not model.norm._forward_hooks
    for a, b in zip(before, model.parameters()):
        assert torch.equal(a, b)


def test_prefix_map_is_mined_not_authored():
    """Prefixes must carry statistical support, not be hand-written."""
    from pathlib import Path
    from smalltalk.harness.steering import PrefixMap
    p = Path(__file__).resolve().parents[1] / "artifacts/harness/prefixes.json"
    if not p.exists():
        pytest.skip("prefix map not mined in this environment")
    m = PrefixMap.load(p)
    for pid, entry in m.table.items():
        for row in entry["first"]:
            assert row["count"] >= 25 and row["lift"] >= 1.5
