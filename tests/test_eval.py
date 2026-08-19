"""Generation, metrics, bench, judge formats, distillation scoring, state."""

import json

import torch

from smalltalk.distill.scoring import (
    HeuristicScorer,
    SelectionConfig,
    rejection_sample,
    weighted_score,
)
from smalltalk.eval.bench import CATEGORIES, default_scenarios
from smalltalk.eval.judge import (
    CRITERIA,
    aggregate_judge_scores,
    build_judge_requests,
    build_pairwise,
    score_pairwise,
    write_pairwise,
)
from smalltalk.eval.metrics import (
    aggregate,
    context_copy_ratio,
    distinct_n,
    evaluate_transcript,
    looks_ungrammatical,
    loop_detected,
    repeated_ngram_ratio,
)
from smalltalk.infer.generate import ConversationEngine, GenerationConfig, generate
from smalltalk.infer.state import ConversationState
from smalltalk.model import build_model


# ---- generation ------------------------------------------------------------
def test_generate_respects_limits(tiny_cfg, tokenizer):
    torch.manual_seed(0)
    m = build_model(tiny_cfg)
    ids, _ = tokenizer.encode_conversation(
        [{"role": "user", "content": "hey"}], add_generation_prompt=True
    )
    out = generate(m, tokenizer, ids, GenerationConfig(max_new_tokens=12, seed=1))
    assert 0 < len(out) <= 12
    assert tokenizer.pad_id not in out
    assert isinstance(tokenizer.decode(out), str)


def test_generation_is_reproducible_with_a_seed(tiny_cfg, tokenizer):
    torch.manual_seed(0)
    m = build_model(tiny_cfg)
    ids = tokenizer.encode_conversation([{"role": "user", "content": "hey"}],
                                        add_generation_prompt=True)[0]
    cfg = GenerationConfig(max_new_tokens=10, seed=42)
    assert generate(m, tokenizer, ids, cfg) == generate(m, tokenizer, ids, cfg)


def test_greedy_is_deterministic(tiny_cfg, tokenizer):
    torch.manual_seed(0)
    m = build_model(tiny_cfg)
    ids = tokenizer.encode_conversation([{"role": "user", "content": "hey"}],
                                        add_generation_prompt=True)[0]
    cfg = GenerationConfig(greedy=True, max_new_tokens=8)
    assert generate(m, tokenizer, ids, cfg) == generate(m, tokenizer, ids, cfg)


def test_stops_on_endofturn(tiny_cfg, tokenizer):
    """A dominant <|endofturn|> logit must terminate generation."""
    m = build_model(tiny_cfg).eval()
    stop = tokenizer.endofturn_id

    class ForcedStop(torch.nn.Module):
        """Wraps the model and always makes <|endofturn|> the argmax."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.cfg = inner.cfg

        def forward(self, *a, **kw):
            logits, loss = self.inner(*a, **kw)
            logits = logits.clone()
            logits[..., stop] = logits.max() + 10.0
            return logits, loss

        def new_cache(self):
            return self.inner.new_cache()

    forced = ForcedStop(m)
    ids = tokenizer.encode_conversation([{"role": "user", "content": "hey"}],
                                        add_generation_prompt=True)[0]
    # min_new_tokens suppresses the stop token for the first 3 steps...
    out = generate(forced, tokenizer, ids,
                   GenerationConfig(greedy=True, max_new_tokens=40, min_new_tokens=3))
    assert len(out) == 3, out
    # ...and with no minimum it stops immediately.
    out0 = generate(forced, tokenizer, ids,
                    GenerationConfig(greedy=True, max_new_tokens=40, min_new_tokens=0))
    assert out0 == []


def test_no_repeat_ngram_blocks_repeats(tiny_cfg, tokenizer):
    m = build_model(tiny_cfg)
    ids = tokenizer.encode_conversation([{"role": "user", "content": "hey"}],
                                        add_generation_prompt=True)[0]
    out = generate(m, tokenizer, ids,
                   GenerationConfig(greedy=True, max_new_tokens=24, no_repeat_ngram_size=2))
    bigrams = [tuple(out[i : i + 2]) for i in range(len(out) - 1)]
    assert len(bigrams) == len(set(bigrams))


def test_engine_keeps_history_and_truncates_context(tiny_cfg, tokenizer):
    m = build_model(tiny_cfg)
    eng = ConversationEngine(m, tokenizer, GenerationConfig(max_new_tokens=6, max_context=48))
    for turn in ["hey", "tired", "work was rough", "yeah"]:
        eng.reply(turn)
    assert len(eng.history) == 8
    assert eng.history[0]["role"] == "user" and eng.history[-1]["role"] == "assistant"
    transcript = eng.run_scenario(["hey", "yeah"])
    assert len(transcript) == 4


def test_engine_state_is_optional_and_off_by_default(tiny_cfg, tokenizer):
    m = build_model(tiny_cfg)
    assert ConversationEngine(m, tokenizer).state is None
    eng = ConversationEngine(m, tokenizer, GenerationConfig(max_new_tokens=4), use_state=True)
    eng.reply("hey im dave, work was brutal today")
    assert eng.state.user_name == "Dave"
    assert eng.state.current_topic == "work"


# ---- zero-parameter state --------------------------------------------------
def test_state_tracks_topic_mood_and_details():
    s = ConversationState()
    s.observe("user", "hey im Sam")
    s.observe("user", "work was brutal, im exhausted")
    s.observe("user", "i have a beagle named emma")
    assert s.user_name == "Sam"
    assert s.current_topic in ("work", "family", "mood")
    assert s.user_mood in ("tired", "stressed")
    assert any("beagle" in d for d in s.user_details)
    assert "Sam" in s.as_system_hint()
    assert len(s.as_system_hint()) < 200


# ---- metrics ---------------------------------------------------------------
def test_repetition_and_loop_detection():
    assert repeated_ngram_ratio("i am tired i am tired i am tired") > 0.3
    assert repeated_ngram_ratio("hey what's up today") == 0.0
    assert loop_detected(["yeah totally", "yeah totally", "ok"])
    assert loop_detected(["ok", "sure", "ok", "ok"])
    assert not loop_detected(["hey", "long day?", "damn, what happened?"])


def test_ungrammatical_heuristics():
    assert looks_ungrammatical("the the the the")
    assert looks_ungrammatical("hhhhhhhh")
    assert looks_ungrammatical("")
    assert not looks_ungrammatical("hey, what's up?")
    assert not looks_ungrammatical("that's the worst kind of tired lol")


def test_context_copy_and_diversity():
    assert context_copy_ratio("work was brutal", "work was brutal today") == 1.0
    assert context_copy_ratio("long day?", "work was brutal") == 0.0
    assert distinct_n(["hey there", "hey there"], 1) == 0.5


def test_evaluate_transcript_flags_broken_turns():
    good = [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "hey, what's up?"},
        {"role": "user", "content": "kinda tired"},
        {"role": "assistant", "content": "long day?"},
    ]
    ev = evaluate_transcript(good)
    assert ev.broken_turns == 0 and ev.completed_clean

    bad = [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "the the the the the"},
        {"role": "user", "content": "kinda tired"},
        {"role": "assistant", "content": "As an AI language model, I cannot be tired."},
    ]
    ev2 = evaluate_transcript(bad)
    assert ev2.broken_turns == 2
    assert "ungrammatical" in ev2.turns[0].broken_reasons
    assert "ai_assistant_style" in ev2.turns[1].broken_reasons
    assert not ev2.completed_clean


def test_context_copy_is_flagged():
    msgs = [
        {"role": "user", "content": "work was really brutal today honestly"},
        {"role": "assistant", "content": "work was really brutal today honestly"},
    ]
    assert "context_copy" in evaluate_transcript(msgs).turns[0].broken_reasons


def test_probes_run():
    msgs = [
        {"role": "user", "content": "my dog is emma"},
        {"role": "assistant", "content": "cute name!"},
        {"role": "user", "content": "whats my dog's name?"},
        {"role": "assistant", "content": "emma right?"},
    ]
    from smalltalk.eval.bench import Probe

    ev = evaluate_transcript(msgs, probes=[Probe(turn=2, type="memory", expect_any=["emma"])])
    assert ev.probe_results[0]["passed"]

    ev2 = evaluate_transcript(msgs, probes=[Probe(turn=1, type="memory", expect_any=["emma"])])
    assert not ev2.probe_results[0]["passed"]
    assert "probe_memory" in ev2.turns[0].broken_reasons


def test_unknown_fact_probe_rejects_fabrication():
    from smalltalk.eval.bench import Probe

    fab = [{"role": "user", "content": "who invented it"},
           {"role": "assistant", "content": "It was invented by James Hargreaves in 1764."}]
    graceful = [{"role": "user", "content": "who invented it"},
                {"role": "assistant", "content": "honestly no idea lol. what is it?"}]
    p = [Probe(turn=1, type="unknown_fact", forbid_confident=True)]
    assert not evaluate_transcript(fab, probes=p).probe_results[0]["passed"]
    assert evaluate_transcript(graceful, probes=p).probe_results[0]["passed"]


def test_aggregate_primary_metric():
    clean = [{"role": "user", "content": f"turn {i}"} for i in range(1)]
    msgs = []
    replies = ["hey", "long day?", "oof", "damn", "yeah?", "for real",
               "haha", "nice", "same", "later!"]
    for i, r in enumerate(replies):
        msgs.append({"role": "user", "content": f"thing number {i} happened"})
        msgs.append({"role": "assistant", "content": r})
    ev = evaluate_transcript(msgs, scenario_id="s", category="greeting")
    agg = aggregate([ev], min_turns=10)
    assert agg["scenarios"] == 1
    assert agg["clean_10turn_rate"] == 1.0
    assert agg["clean_5turn_rate"] == 1.0
    assert agg["grammatical_rate"] == 1.0
    assert "greeting" in agg["by_category"]
    assert clean is not None


# ---- bench -----------------------------------------------------------------
def test_bench_scenarios_are_wellformed():
    scenarios = default_scenarios()
    assert len(scenarios) >= 12
    ids = [s.id for s in scenarios]
    assert len(set(ids)) == len(ids)
    covered = {s.category for s in scenarios}
    for required in ("greeting", "bad_day", "good_day", "boredom", "excitement",
                     "story", "topic_change", "emotional_disclosure", "joke",
                     "short_answers", "ambiguous", "memory", "unknown_fact"):
        assert required in covered, required
    assert all(c in CATEGORIES for c in covered)
    # the primary metric needs 10-turn scenarios
    assert all(s.num_turns >= 10 for s in scenarios)


def test_bench_roundtrip(tmp_path):
    from smalltalk.eval.bench import load_scenarios, save_scenarios

    p = save_scenarios(tmp_path / "b.jsonl")
    back = load_scenarios(p)
    assert [s.id for s in back] == [s.id for s in default_scenarios()]
    assert back[1].probes[0].expect_any


def test_run_bench_end_to_end(tiny_cfg, tokenizer):
    from smalltalk.eval.runner import run_bench

    m = build_model(tiny_cfg)
    eng = ConversationEngine(m, tokenizer, GenerationConfig(max_new_tokens=6, max_context=96))
    evals, transcripts = run_bench(eng, default_scenarios()[:3])
    assert len(evals) == 3 and len(transcripts) == 3
    assert all(e.metrics["num_turns"] == 10 for e in evals)
    agg = aggregate(evals)
    assert 0.0 <= agg["clean_10turn_rate"] <= 1.0


# ---- judge -----------------------------------------------------------------
def test_judge_request_format():
    transcripts = [{"scenario_id": "greet-01", "category": "greeting",
                    "messages": [{"role": "user", "content": "hey"},
                                 {"role": "assistant", "content": "hey!"}]}]
    reqs = build_judge_requests(transcripts, "smalltalk-4m")
    d = reqs[0].to_dict()
    assert set(d["criteria"]) == set(CRITERIA)
    assert len(CRITERIA) == 8
    assert d["rubric"]["scale"] == "1-5 integers"
    assert set(d["response_format"]["scores"]) == set(CRITERIA)
    assert "not an assistant" in d["system_prompt"].lower()


def test_judge_score_ingestion(tmp_path):
    p = tmp_path / "scores.jsonl"
    p.write_text("\n".join(
        json.dumps({"id": f"x{i}", "scores": {k: 4 for k in CRITERIA}}) for i in range(3)
    ), encoding="utf-8")
    from smalltalk.eval.judge import read_judge_scores

    scores = read_judge_scores(p)
    agg = aggregate_judge_scores(scores)
    assert len(scores) == 3
    assert agg["mean_all_criteria"] == 4.0


def test_pairwise_is_blind_and_scorable(tmp_path):
    def mk(name, reply):
        return [{"scenario_id": "s1", "category": "greeting",
                 "messages": [{"role": "user", "content": "hey"},
                              {"role": "assistant", "content": reply},
                              {"role": "user", "content": "tired"},
                              {"role": "assistant", "content": f"{reply} 2"}]}]

    items, key = build_pairwise(mk("a", "hey!"), mk("b", "greetings."), "model_a", "model_b")
    assert items
    d = items[0].to_dict()
    assert set(d["options"]) == {"A", "B"}
    assert "model" not in json.dumps(d["options"])  # identities hidden
    p, kp = write_pairwise(tmp_path / "pw.jsonl", items, key)

    votes = tmp_path / "votes.jsonl"
    votes.write_text("\n".join(
        json.dumps({"id": i.id, "choice": "A"}) for i in items
    ), encoding="utf-8")
    res = score_pairwise(votes, kp)
    assert res["comparisons"] == len(items)
    assert sum(res["wins"].values()) == len(items)


# ---- distillation ----------------------------------------------------------
def test_heuristic_scorer_prefers_natural_short_replies():
    s = HeuristicScorer()
    ctx = [{"role": "user", "content": "work was brutal today, im exhausted"}]
    good = s.score(ctx, "oof, long day?")
    verbose = s.score(ctx, "I'm sorry to hear that. Here are some tips: 1. rest well. "
                           "It's important to note that sleep is essential for recovery.")
    parrot = s.score(ctx, "work was brutal today, im exhausted")
    w = SelectionConfig().weights
    assert weighted_score(good, w) > weighted_score(verbose, w)
    assert weighted_score(good, w) > weighted_score(parrot, w)
    assert good["emotional_appropriateness"] > 4.0
    assert verbose["non_assistant_style"] < 3.0
    assert parrot["relevance"] < 2.0


def test_scorer_rewards_graceful_ignorance():
    s = HeuristicScorer()
    ctx = [{"role": "user", "content": "who invented the spinning jenny?"}]
    honest = s.score(ctx, "honestly no idea lol. what is it?")
    fabricated = s.score(ctx, "it was invented by James Hargreaves in 1764.")
    assert honest["no_unnecessary_facts"] > fabricated["no_unnecessary_facts"]


def test_scorer_penalises_valence_mismatch():
    s = HeuristicScorer()
    sad = [{"role": "user", "content": "my dog died today, im so sad"}]
    assert s.score(sad, "congrats! amazing!")["emotional_appropriateness"] < 2.0
    assert s.score(sad, "oh no, im so sorry.")["emotional_appropriateness"] > 4.0


def test_all_eight_dimensions_scored():
    from smalltalk.distill.scoring import DIMENSIONS

    s = HeuristicScorer().score([{"role": "user", "content": "hey"}], "hey, what's up?")
    assert set(s) == set(DIMENSIONS)
    assert len(DIMENSIONS) == 8
    assert all(1.0 <= v <= 5.0 for v in s.values())


def test_rejection_sampling_selects_and_rejects():
    from smalltalk.data.schema import Candidate, Conversation, Turn

    conv = Conversation(
        id="c1",
        messages=[Turn("user", "work was brutal today, im exhausted"),
                  Turn("assistant", "placeholder")],
        candidates=[
            Candidate("As an AI language model, I cannot get tired. Here are some tips: 1. rest."),
            Candidate("oof. long day?"),
            Candidate("the the the the the the"),
        ],
    )
    kept, stats = rejection_sample([conv])
    assert stats["kept"] == 1
    assert kept[0].chosen == 1
    assert kept[0].candidates[1].score > kept[0].candidates[0].score

    hopeless = Conversation(
        id="c2", messages=[Turn("user", "hey"), Turn("assistant", "x")],
        candidates=[Candidate("As an AI language model I must note that furthermore")],
    )
    kept2, stats2 = rejection_sample([hopeless])
    assert kept2 == []
    assert stats2["rejected_low_score"] + stats2["rejected_floor"] == 1


def test_precomputed_scorer_uses_external_scores(tmp_path):
    from smalltalk.data.schema import Candidate, Conversation, Turn
    from smalltalk.distill.scoring import DIMENSIONS, PrecomputedScorer, score_conversation

    p = tmp_path / "human.jsonl"
    p.write_text(json.dumps({
        "id": "c1", "candidate": 1, "scores": {d: 5 for d in DIMENSIONS}
    }) + "\n", encoding="utf-8")
    conv = Conversation(id="c1", messages=[Turn("user", "hey"), Turn("assistant", "x")],
                        candidates=[Candidate("greetings, human."), Candidate("hey!")])
    scored = score_conversation(conv, PrecomputedScorer.from_jsonl(p, HeuristicScorer()))
    assert scored.chosen == 1
    assert scored.candidates[1].score == 5.0


def test_llm_judge_packets_shape():
    from smalltalk.data.schema import Candidate, Conversation, Turn
    from smalltalk.distill.scoring import llm_judge_packets

    conv = Conversation(id="c1", messages=[Turn("user", "hey"), Turn("assistant", "x")],
                        candidates=[Candidate("hey!"), Candidate("hi.")])
    packets = llm_judge_packets([conv])
    assert len(packets) == 2
    assert packets[0]["context"][-1]["role"] == "user"
    assert len(packets[0]["dimensions"]) == 8
