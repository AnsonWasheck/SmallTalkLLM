import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smalltalk.config import ModelConfig, all_model_configs  # noqa: E402
from smalltalk.data.synthetic import OfflineConfig, generate_offline_corpus  # noqa: E402
from smalltalk.tokenizer import train_tokenizer  # noqa: E402


@pytest.fixture(scope="session")
def tiny_corpus():
    return list(generate_offline_corpus(OfflineConfig(num_conversations=300, seed=3)))


@pytest.fixture(scope="session")
def tokenizer(tiny_corpus, tmp_path_factory):
    texts = [m.content for c in tiny_corpus for m in c.messages]
    out = tmp_path_factory.mktemp("tok")
    return train_tokenizer(texts, vocab_size=512, out_dir=out)


@pytest.fixture
def tiny_cfg(tokenizer):
    """A 2-layer toy model for fast behavioural tests."""
    return ModelConfig(
        name="toy", vocab_size=tokenizer.vocab_size, hidden_size=64, num_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, max_position_embeddings=128,
    )


@pytest.fixture(scope="session")
def experiment_configs():
    return all_model_configs()
