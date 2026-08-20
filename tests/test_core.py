"""v0.2-Core: the curriculum must be narrow on output and broad on input, and
the benchmark must be a generalisation test rather than a lookup test."""

import pytest

from smalltalk.core import bench_core, core_gen
from smalltalk.core.intents import INTENTS, LEN_TOKENS, normalise
from smalltalk.tokenizer import LENGTH_TOKENS


def test_held_out_paraphrases_are_never_generated():
    """The whole point: Core-Bench prompts must be unseen surfaces."""
    trained = {normalise(m.content)
               for c in core_gen.generate(core_gen.CoreConfig(n=4000))
               for m in c.messages if m.role == "user"}
    for s in bench_core.build_scenarios():
        assert normalise(s.prompt) not in trained, s.prompt


def test_output_entropy_is_low():
    """Each intent must map to essentially one reply; diversity here is a bug."""
    from collections import defaultdict
    targets = defaultdict(set)
    for c in core_gen.generate(core_gen.CoreConfig(n=6000)):
        targets[c.meta["skill"]].add(c.messages[-1].content)
    for skill, ts in targets.items():
        assert len(ts) <= 2, f"{skill} has {len(ts)} distinct targets"


def test_every_target_carries_a_length_token():
    for c in core_gen.generate(core_gen.CoreConfig(n=500)):
        assert c.messages[-1].content.startswith(tuple(LEN_TOKENS))


def test_length_tokens_come_from_the_tokenizer():
    assert list(LEN_TOKENS) == LENGTH_TOKENS


def test_input_surface_is_broad():
    seen = {m.content for c in core_gen.generate(core_gen.CoreConfig(n=4000))
            for m in c.messages if m.role == "user"}
    assert len(seen) > 600


def test_scoring_ignores_the_length_prefix():
    assert bench_core.score("greeting", "<|len_vshort|> hey")
    assert bench_core.score("greeting", "hey")
    assert not bench_core.score("greeting", "<|len_vshort|> what do you mean?")


def test_frozen_checksum_detects_drift(tmp_path):
    p = tmp_path / "frozen.json"
    bench_core.freeze(p)
    bench_core.verify_frozen(p)
    p.write_text(p.read_text().replace(bench_core.checksum(), "deadbeefdeadbeef"))
    with pytest.raises(RuntimeError):
        bench_core.verify_frozen(p)


def test_tier1_intents_are_the_conversational_openers():
    tier1 = {i.name for i in INTENTS if i.tier == 1}
    assert {"greeting", "greeting_how_are_you", "how_are_you", "thanks",
            "goodbye"} <= tier1


def test_benchmark_measures_context_not_just_turn_one():
    """A reflex that only fires on turn 1 is not reliable.

    Until v0.2.2 every scenario ran on a freshly reset engine, so the benchmark
    could not see the continuity axis that a manual side-by-side test found to
    separate two checkpoints it scored as close.
    """
    scenarios = bench_core.build_scenarios()
    with_ctx = [s for s in scenarios if s.context]
    assert with_ctx, "benchmark is single-turn only"
    assert len(with_ctx) / len(scenarios) > 0.25


def test_context_preambles_carry_no_core_cue():
    """Preambles must not themselves be answerable as a Core intent, or a model
    could score by responding to the preamble instead of the probe."""
    surfaces = {normalise(p) for i in INTENTS for p in i.train + i.held_out}
    for ctx in bench_core.CONTEXTS:
        for turn in ctx:
            assert normalise(turn) not in surfaces, turn


def test_topic_statement_covers_the_common_case():
    """The class exists because 99.3% of real user turns matched no intent."""
    it = next(i for i in INTENTS if i.name == "topic_statement")
    assert len(it.train) >= 20
    assert len(it.targets) == 1          # narrow output, per the design rule
