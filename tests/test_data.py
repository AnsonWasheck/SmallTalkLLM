"""Data pipeline: schema, adapters, cleaning, dedup, splitting, SFT masking."""

import json

from smalltalk.data.adapters import (
    load_dailydialog,
    load_empathetic_dialogues,
    load_jsonl_conversations,
    normalize_record,
    parse_transcript,
)
from smalltalk.data.clean import (
    FilterConfig,
    clean_conversations,
    corpus_stats,
    merge_consecutive,
    normalize_text,
    split_train_val,
    split_by_family,
)
from smalltalk.data.dataset import PackedCLMDataset, SFTDataset, build_dataset
from smalltalk.data.schema import Conversation, Turn, load_conversations, write_jsonl


def conv(*pairs, cid="c", source="test"):
    msgs = []
    for i, text in enumerate(pairs):
        msgs.append(Turn("user" if i % 2 == 0 else "assistant", text))
    return Conversation(id=cid, messages=msgs, source=source)


# ---- schema ---------------------------------------------------------------
def test_jsonl_roundtrip(tmp_path):
    c = conv("hey", "hey, what's up?", "tired", "long day?")
    p = tmp_path / "x.jsonl"
    assert write_jsonl(p, [c]) == 1
    back = load_conversations(p)[0]
    assert back.id == c.id
    assert [m.to_dict() for m in back.messages] == [m.to_dict() for m in c.messages]
    assert back.num_turns == 4


def test_candidates_roundtrip(tmp_path):
    from smalltalk.data.schema import Candidate

    c = conv("hey", "hi")
    c.candidates = [Candidate("hey!", {"naturalness": 5.0}, 4.5), Candidate("greetings.", score=2.0)]
    c.chosen = 0
    p = tmp_path / "c.jsonl"
    write_jsonl(p, [c])
    back = load_conversations(p)[0]
    assert len(back.candidates) == 2 and back.chosen == 0
    assert back.candidates[0].scores["naturalness"] == 5.0


# ---- adapters -------------------------------------------------------------
def test_dailydialog_raw_format(tmp_path):
    d = tmp_path / "dd"
    d.mkdir()
    (d / "dialogues_text.txt").write_text(
        "Hey there . __eou__ Hi ! How are you ? __eou__ Good thanks . __eou__\n"
        "What's up ? __eou__ Not much . __eou__\n",
        encoding="utf-8",
    )
    convs = list(load_dailydialog(d))
    assert len(convs) == 2
    assert convs[0].messages[0].role == "user"
    assert convs[0].messages[1].role == "assistant"
    assert convs[0].num_turns == 3


def test_empathetic_dialogues_csv_regrouping(tmp_path):
    csv = tmp_path / "train.csv"
    csv.write_text(
        "conv_id,utterance_idx,context,prompt,utterance\n"
        "hit:0,1,sad,I was lonely,I felt so alone_comma_ honestly\n"
        "hit:0,2,sad,I was lonely,That sounds really hard\n"
        "hit:1,1,proud,I won,I won the race!\n"
        "hit:1,2,proud,I won,Congrats!\n",
        encoding="utf-8",
    )
    convs = sorted(load_empathetic_dialogues(csv), key=lambda c: c.id)
    assert len(convs) == 2
    assert "_comma_" not in convs[0].messages[0].content
    assert convs[0].meta["emotion"] == "sad"


def test_transcript_parsing():
    turns = parse_transcript("User: hey\nAssistant: hey, what's up?\nUser: tired")
    assert [t.role for t in turns] == ["user", "assistant", "user"]
    assert turns[1].content == "hey, what's up?"


def test_normalize_record_accepts_three_shapes():
    a = normalize_record({"messages": [{"role": "user", "content": "hey"},
                                       {"role": "assistant", "content": "hi"}]}, "a")
    b = normalize_record({"conversation": "User: hey\nAssistant: hi"}, "b")
    c = normalize_record({"dialog": ["hey", "hi"]}, "c")
    for x in (a, b, c):
        assert x is not None and x.num_turns == 2


def test_teacher_candidates_ingested(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({
        "messages": [{"role": "user", "content": "hey"}, {"role": "user", "content": "tired"}],
        "candidates": ["long day?", {"content": "oof", "score": 4.0}],
    }) + "\n", encoding="utf-8")
    c = list(load_jsonl_conversations(p))[0]
    assert len(c.candidates) == 2
    assert c.candidates[1].score == 4.0


# ---- cleaning -------------------------------------------------------------
def test_normalization():
    assert normalize_text("hey   there  !") == "hey there!"
    assert normalize_text("don ' t  worry") == "don't worry"
    assert normalize_text("“quoted”") == '"quoted"'
    assert "<link>" in normalize_text("see https://example.com/x now")
    assert normalize_text("User: hey") == "hey"
    assert normalize_text("wow!!!!!!!") == "wow!!!"


def test_merge_consecutive_preserves_boundaries():
    turns = [Turn("user", "hey"), Turn("user", "you there"), Turn("assistant", "yeah")]
    merged = merge_consecutive(turns)
    assert len(merged) == 2
    assert merged[0].role == "user" and "you there" in merged[0].content


def test_filters_drop_off_domain_content():
    cases = {
        "ai_assistant_style": conv("hi", "As an AI language model, I cannot help with that."),
        "code": conv("hi", "sure: def foo(): return 1"),
        "factual_qa": conv("what is paris", "Paris is the capital of France."),
        "instruction_following": conv("hey", "Write an essay about the ocean please."),
        "assistant_turn_too_long": conv("hey", " ".join(["word"] * 200)),
    }
    for expected, c in cases.items():
        kept, stats = clean_conversations([c], FilterConfig())
        assert kept == [], f"{expected} was not filtered"
        assert expected in stats.dropped, (expected, dict(stats.dropped))


def test_good_smalltalk_survives():
    c = conv("hey", "hey, what's up?", "honestly kinda tired today", "Long day?",
             "yeah work was brutal", "Damn. What happened?")
    kept, stats = clean_conversations([c], FilterConfig())
    assert len(kept) == 1, stats.report()


def test_permissive_config_is_a_real_control():
    c = conv("what is paris", "Paris is the capital of France. " + " ".join(["x"] * 100))
    assert clean_conversations([c], FilterConfig())[0] == []
    assert len(clean_conversations([c], FilterConfig.permissive())[0]) == 1


def test_exact_and_near_dedup():
    a = conv("hey", "hey, what's up?", cid="a")
    b = conv("hey", "hey, what's up?", cid="b")
    kept, stats = clean_conversations([a, b], FilterConfig())
    assert len(kept) == 1
    assert stats.dropped["exact_duplicate"] == 1


def test_downweight_mode_keeps_with_weight():
    c = conv("hey", " ".join(["word"] * 100))
    cfg = FilterConfig()
    cfg.downweight_instead_of_drop = True
    kept, _ = clean_conversations([c], cfg)
    assert len(kept) == 1
    assert kept[0].meta["weight"] < 0.5
    assert kept[0].meta["filter_flag"]


def test_split_is_deterministic_and_stratified():
    convs = [conv("hey", f"reply {i}", cid=f"x{i}", source="s" if i % 2 else "t")
             for i in range(50)]
    t1, v1 = split_train_val(convs, val_ratio=0.2, seed=1)
    t2, v2 = split_train_val(convs, val_ratio=0.2, seed=1)
    assert {c.id for c in v1} == {c.id for c in v2}
    assert not ({c.id for c in t1} & {c.id for c in v1})
    assert len(v1) + len(t1) == 50
    assert {c.source for c in v1} == {"s", "t"}


def test_family_split_is_disjoint_and_append_stable():
    original = [
        Conversation(id=f"c{i}", messages=conv("hey", "hi", cid=f"c{i}").messages,
                     source="qwen", meta={"family": f"family-{i}"})
        for i in range(40)
    ]
    expanded = original + [
        Conversation(id=f"new{i}", messages=conv("hey", "hi", cid=f"new{i}").messages,
                     source="qwen", meta={"family": f"new-family-{i}"})
        for i in range(40)
    ]
    a = split_by_family(original, seed=7)
    b = split_by_family(expanded, seed=7)
    memberships_a = {c.meta["family"]: split for split, part in zip(("train", "val", "test"), a) for c in part}
    memberships_b = {c.meta["family"]: split for split, part in zip(("train", "val", "test"), b) for c in part}
    assert memberships_a == {k: memberships_b[k] for k in memberships_a}
    assert len(set().union(*(set(c.meta["family"] for c in part) for part in a))) == 40


def test_corpus_stats(tiny_corpus):
    s = corpus_stats(tiny_corpus)
    assert s["conversations"] == len(tiny_corpus)
    assert 1 <= s["assistant_words_mean"] <= 30
    assert s["assistant_words_in_3_25"] > 0.5


# ---- datasets -------------------------------------------------------------
def test_packed_clm_dataset_shapes(tiny_corpus, tokenizer):
    ds = PackedCLMDataset(tiny_corpus, tokenizer, seq_len=64, seed=0)
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["labels"].shape == (64,)
    assert item["loss_mask"].sum() == 64  # stage 1 trains on everything
    assert item["segment_ids"].shape == (64,)
    assert len(ds) > 1


def test_sft_dataset_masks_only_assistant(tiny_corpus, tokenizer):
    ds = SFTDataset(tiny_corpus, tokenizer, seq_len=256, mask_non_assistant=True)
    item = ds[0]
    ids, mask = item["input_ids"].tolist(), item["loss_mask"].tolist()
    assert 0 < sum(mask) < len(mask)
    for i, m in zip(ids, mask):
        if m:
            assert i not in (tokenizer.user_id, tokenizer.system_id,
                             tokenizer.bos_id, tokenizer.pad_id, tokenizer.assistant_id)
    # padding is never supervised
    for i, m in zip(ids, mask):
        if i == tokenizer.pad_id:
            assert m == 0


def test_sft_mask_ablation_trains_on_everything(tiny_corpus, tokenizer):
    masked = SFTDataset(tiny_corpus, tokenizer, 256, mask_non_assistant=True)[0]
    unmasked = SFTDataset(tiny_corpus, tokenizer, 256, mask_non_assistant=False)[0]
    assert unmasked["loss_mask"].sum() > masked["loss_mask"].sum()


def test_distill_dataset_uses_chosen_candidate(tokenizer):
    from smalltalk.data.schema import Candidate

    c = conv("hey", "placeholder reply here")
    c.candidates = [Candidate("bad one", score=1.0), Candidate("hey, what's up?", score=4.9)]
    ds = build_dataset("distill", [c], tokenizer, seq_len=64)
    ids = ds[0]["input_ids"].tolist()
    mask = ds[0]["loss_mask"].tolist()
    text = tokenizer.decode([i for i, m in zip(ids, mask) if m])
    assert "what's up" in text
    assert "placeholder" not in text
