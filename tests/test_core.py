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


def _fake_real_pool(n=300):
    """Context requires a real-dialogue pool; without one the generator emits
    turn-1 examples by necessity, which is a property of the input, not a bug."""
    from smalltalk.data.schema import Conversation, Turn
    return [Conversation(id=f"f{i}", source="fake",
                         messages=[Turn("user", f"just some filler line {i} here"),
                                   Turn("assistant", f"mm yeah line {i} indeed")])
            for i in range(n)]


def test_curriculum_trains_reflexes_in_context_not_just_as_openers():
    """Measured: goodbye scored 76.2% bare but 35.7% with a preamble.

    The curriculum was 65% turn-1 examples, so reflexes were being learned in a
    position they rarely occupy in real use.
    """
    convs = list(core_gen.generate(core_gen.CoreConfig(n=4000),
                                   real_convs=_fake_real_pool()))
    multi = sum(1 for c in convs if len(c.messages) > 2)
    assert multi / len(convs) > 0.5, "most Core examples are still turn-1 only"


def test_inherently_mid_conversation_intents_get_context():
    convs = [c for c in core_gen.generate(core_gen.CoreConfig(n=6000),
                                          real_convs=_fake_real_pool())
             if c.meta["skill"] == "goodbye"]
    assert convs
    with_ctx = sum(1 for c in convs if len(c.messages) > 2) / len(convs)
    assert with_ctx > 0.75, f"goodbye trained with context only {with_ctx:.0%} of the time"


def test_statebench_probes_carry_no_valence():
    """The probe must be uninformative alone, or the pair stops being a test of
    state and becomes a test of last-turn classification."""
    from smalltalk.core import statebench as sb
    for a, b in sb.pairs():
        assert a.probe == b.probe, f"{a.pair_id} probes differ"
        assert sb.classify(a.probe) in ("neutral", "other"), a.probe


def test_statebench_pairs_differ_only_in_valence():
    from smalltalk.core import statebench as sb
    for a, b in sb.pairs():
        assert a.topic == b.topic
        assert len(a.turns) == len(b.turns)
        differing = sum(x != y for x, y in zip(a.turns, b.turns))
        assert differing <= 2, f"{a.pair_id} differs in {differing} turns"


def test_state_curriculum_emits_minimal_counterfactual_pairs():
    """The whole v0.3 mechanism: identical probes, opposite targets. If pairs
    are not minimal, the model can separate them without tracking state."""
    from smalltalk.core.state_gen import StateConfig, generate
    fams = {}
    for c in generate(StateConfig(n=400)):
        fams.setdefault(c.meta["family"], []).append(c)
    complete = [g for g in fams.values() if len(g) == 2]
    assert len(complete) > 150
    for a, b in complete:
        ua = [m.content for m in a.messages if m.role == "user"]
        ub = [m.content for m in b.messages if m.role == "user"]
        ta = [m.content for m in a.messages if m.role == "assistant"]
        tb = [m.content for m in b.messages if m.role == "assistant"]
        assert len(ua) == len(ub)
        assert sum(x != y for x, y in zip(ua, ub)) <= 2, "pair is not minimal"
        assert ta != tb, "counterfactual pair has identical targets"


def test_state_generator_never_emits_a_statebench_opener():
    from smalltalk.core.state_gen import BENCH_SURFACES, StateConfig, generate
    import re
    def n(t):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", t.lower())).strip()
    for c in generate(StateConfig(n=600)):
        for m in c.messages:
            if m.role == "user":
                assert n(m.content) not in BENCH_SURFACES, m.content


def test_varietybench_penalises_a_broken_record():
    """A model repeating one phrase must score badly even if the phrase is right."""
    from smalltalk.core import varietybench as vb
    stuck = {n: ["that sounds rough"] * len(t) for n, t in vb.CONVERSATIONS}
    varied = {n: [f"reply number {i} here" for i, _ in enumerate(t)]
              for n, t in vb.CONVERSATIONS}
    assert vb.score(stuck)["repeat_rate"] > 0.7
    assert vb.score(varied)["repeat_rate"] == 0.0
    assert vb.score(stuck)["top1_share"] > vb.score(varied)["top1_share"]


def test_varietybench_forgives_trivial_acknowledgements():
    """"yeah" twice in a conversation is human; a repeated reaction is not."""
    from smalltalk.core import varietybench as vb
    trivial = {n: ["yeah"] * len(t) for n, t in vb.CONVERSATIONS}
    assert vb.score(trivial)["repeat_rate"] == 0.0


def test_state_curriculum_does_not_repeat_within_a_conversation():
    from smalltalk.core.state_gen import StateConfig, generate
    dup = tot = 0
    for c in generate(StateConfig(n=800)):
        rs = [m.content for m in c.messages if m.role == "assistant"]
        dup += len(rs) - len(set(rs))
        tot += len(rs)
    assert dup / tot < 0.05, "training conversations repeat themselves"


def test_every_frame_reply_reuses_the_referent():
    """The whole point: an elaboration that drops the noun is a hedge."""
    from smalltalk.core.frame_gen import FRAMES
    for f in FRAMES:
        for t in f.reply:
            assert "{n}" in t, f"{f.name}: reply {t!r} has no referent slot"
        for t in f.user:
            assert "{n}" in t, f"{f.name}: user {t!r} has no referent slot"


def test_frame_nouns_are_split_and_bench_words_blocked():
    from smalltalk.core.frame_gen import BENCH_WORDS, noun_split
    train, held = noun_split()
    for slot in train:
        assert not (set(train[slot]) & set(held[slot])), "train/held-out overlap"
        assert not (set(train[slot]) & BENCH_WORDS), "benchmark subject in training"
        assert len(held[slot]) >= 1


def test_frame_generator_reuses_the_noun_it_was_given():
    from smalltalk.core.frame_gen import FrameConfig, generate
    convs = list(generate(FrameConfig(n=400)))
    reuse = sum(1 for c in convs for m in c.messages
                if m.role == "assistant" and c.meta["noun"] in m.content)
    total = sum(1 for c in convs for m in c.messages if m.role == "assistant")
    assert reuse / total > 0.5, "most elaborations should name the referent"


def test_held_out_nouns_never_appear_in_training_output():
    """If held-out nouns leaked, generalisation could not be measured."""
    from smalltalk.core.frame_gen import FrameConfig, generate, noun_split
    _, held = noun_split()
    banned = {w for ws in held.values() for w in ws}
    for c in generate(FrameConfig(n=600)):
        for m in c.messages:
            assert not (set(m.content.lower().split()) & banned), m.content
