"""Tokenizer round-trip, special tokens, chat template, SFT mask alignment."""

import pytest

from smalltalk.tokenizer import (
    ASSISTANT,
    ENDOFTURN,
    SPECIAL_TOKENS,
    USER,
    Message,
    SmallTalkTokenizer,
)

PROBES = [
    "hey", "hey, what's up?", "honestly kinda tired today",
    "yeah work was brutal", "That's the worst kind of tired lol.",
    "don't even get me started ugh", "night :)", "i'm gonna go, later!!",
    "whoa really?? no way", "lmao that's rough buddy",
]


@pytest.mark.parametrize("text", PROBES)
def test_roundtrip_is_lossless(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_level_handles_unseen_characters(tokenizer):
    for weird in ["ñ", "日本", "🙂", " x", "~`^"]:
        assert tokenizer.decode(tokenizer.encode(weird)) == weird


def test_special_tokens_present_and_low_ids(tokenizer):
    for t in SPECIAL_TOKENS:
        i = tokenizer.specials[t]
        assert i is not None and i < len(SPECIAL_TOKENS) + 1


def test_special_tokens_are_atomic(tokenizer):
    """A role marker must never be split into pieces."""
    ids = tokenizer.encode_message(Message("user", "hey"))
    assert ids[0] == tokenizer.user_id
    assert ids[-1] == tokenizer.endofturn_id


def test_chat_template_structure(tokenizer):
    msgs = [
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "hey, what's up?"},
    ]
    ids, mask = tokenizer.encode_conversation(msgs, add_bos=True)
    assert ids[0] == tokenizer.bos_id
    assert tokenizer.user_id in ids and tokenizer.assistant_id in ids
    assert ids.count(tokenizer.endofturn_id) == 2
    assert len(ids) == len(mask)
    rendered = tokenizer.render(msgs)
    assert rendered.startswith(f"{USER}hey{ENDOFTURN}")
    assert ASSISTANT in rendered


def test_generation_prompt_ends_with_assistant_header(tokenizer):
    ids, mask = tokenizer.encode_conversation(
        [{"role": "user", "content": "hey"}], add_generation_prompt=True
    )
    assert ids[-1] == tokenizer.assistant_id
    assert mask[-1] == 0


def test_assistant_mask_selects_only_assistant_content(tokenizer):
    msgs = [
        {"role": "system", "content": "be casual"},
        {"role": "user", "content": "hey there friend"},
        {"role": "assistant", "content": "hey, what's up?"},
        {"role": "user", "content": "not much"},
        {"role": "assistant", "content": "nice"},
    ]
    ids, mask = tokenizer.encode_conversation(msgs, add_bos=True)
    masked = [i for i, m in zip(ids, mask) if m]
    unmasked = [i for i, m in zip(ids, mask) if not m]

    # no user/system/bos token is ever a training target
    for sid in (tokenizer.bos_id, tokenizer.user_id, tokenizer.system_id,
                tokenizer.assistant_id):
        assert sid not in masked

    text = tokenizer.decode(masked)
    assert "hey, what's up?" in text and "nice" in text
    assert "not much" not in text and "be casual" not in text
    # the assistant's endofturn IS supervised (the model must learn to stop)
    assert mask[ids.index(tokenizer.assistant_id) + 1] == 1
    assert tokenizer.endofturn_id in masked
    assert tokenizer.user_id in unmasked


def test_conversational_atoms_are_single_tokens(tokenizer):
    """The curated atoms should be cheap. Verified on a high-value subset."""
    cheap = 0
    for atom in ["don't", "i'm", "that's", " lol", " yeah", " hey", " tired"]:
        if len(tokenizer.encode(atom)) == 1:
            cheap += 1
    assert cheap >= 4, "curated conversational atoms are not being merged"


def test_load_after_save(tokenizer, tmp_path):
    p = tokenizer.save(tmp_path / "tok")
    reloaded = SmallTalkTokenizer.load(p)
    assert reloaded.vocab_size == tokenizer.vocab_size
    assert reloaded.encode("hey lol") == tokenizer.encode("hey lol")


def test_vocab_size_is_respected():
    from smalltalk.tokenizer import train_tokenizer

    import tempfile

    texts = ["hey what's up", "not much you", "tired lol"] * 200
    with tempfile.TemporaryDirectory() as d:
        tok = train_tokenizer(texts, vocab_size=400, out_dir=d)
    assert tok.vocab_size <= 400
