"""Smoke check: parameter counts match the documented targets exactly."""

import pytest

from smalltalk.params import analytic_param_count, check_config, empirical_param_count

EXPECTED = {
    "smalltalk-4m": 3_868_928,
    "smalltalk-5m": 5_278_976,
    "smalltalk-7m": 6_689_024,
    "smalltalk-8m": 8_099_072,
    "smalltalk-15m": 14_948_736,
}


def test_all_configs_match_their_targets(experiment_configs):
    for cfg in experiment_configs:
        chk = check_config(cfg)
        assert chk.internally_consistent, chk.report()
        assert chk.ok, chk.report()


def test_formula_matches_module(experiment_configs):
    for cfg in experiment_configs:
        assert analytic_param_count(cfg) == empirical_param_count(cfg), cfg.name


@pytest.mark.parametrize("name,expected", EXPECTED.items())
def test_documented_targets(experiment_configs, name, expected):
    cfg = next(c for c in experiment_configs if c.name == name)
    assert empirical_param_count(cfg) == expected
    # within 1% of the value stated in the research plan
    plan = {"smalltalk-4m": 3.87e6, "smalltalk-5m": 5.28e6, "smalltalk-7m": 6.69e6,
            "smalltalk-8m": 8.10e6, "smalltalk-15m": 14.95e6}[name]
    assert abs(expected - plan) / plan < 0.01


def test_embeddings_are_tied(experiment_configs):
    from smalltalk.model import build_model

    for cfg in experiment_configs:
        assert cfg.tie_word_embeddings
        m = build_model(cfg)
        assert m.lm_head is None, f"{cfg.name} should reuse embed_tokens for the head"


def test_no_bias_parameters(experiment_configs):
    from smalltalk.model import build_model

    m = build_model(experiment_configs[0])
    assert not [n for n, _ in m.named_parameters() if n.endswith("bias")]


def test_gqa_is_actually_grouped(experiment_configs):
    for cfg in experiment_configs:
        assert cfg.num_key_value_heads < cfg.num_attention_heads
        assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
