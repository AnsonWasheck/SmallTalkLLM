"""Architecture correctness: causal masking, KV-cache equivalence, save/reload,
forward+backward for every experimental configuration."""

import torch

from smalltalk.model import SmallTalkModel, build_model


def test_forward_shapes(tiny_cfg):
    m = build_model(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    logits, loss = m(x, labels=x)
    assert logits.shape == (2, 16, tiny_cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_causal_masking_future_tokens_do_not_leak(tiny_cfg):
    """Changing token t must not change logits at positions < t."""
    torch.manual_seed(0)
    m = build_model(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base, _ = m(x)
        for t in (5, 8, 11):
            y = x.clone()
            y[0, t] = (y[0, t] + 7) % tiny_cfg.vocab_size
            alt, _ = m(y)
            assert torch.allclose(base[:, :t], alt[:, :t], atol=1e-5), f"leak at t={t}"
            assert not torch.allclose(base[:, t], alt[:, t], atol=1e-5)


def test_kv_cache_matches_full_forward(tiny_cfg):
    torch.manual_seed(0)
    m = build_model(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 20))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        # prefill then decode one token at a time
        inc, _ = m(x[:, :10], cache=cache)
        outs = [inc]
        for t in range(10, 20):
            step, _ = m(x[:, t : t + 1], cache=cache)
            outs.append(step)
        incremental = torch.cat(outs, dim=1)
    assert incremental.shape == full.shape
    assert torch.allclose(full, incremental, atol=1e-4), (
        (full - incremental).abs().max().item()
    )


def test_cache_trim_keeps_recent_positions(tiny_cfg):
    m = build_model(tiny_cfg).eval()
    cache = m.new_cache()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 30))
    with torch.no_grad():
        m(x, cache=cache)
    assert cache.length == 30
    cache.trim(10)
    assert cache.length == 10


def test_rope_makes_the_model_order_sensitive(tiny_cfg):
    """Word order must matter.

    Note we cannot test this with a run of *identical* tokens: attention over
    identical value vectors returns that value regardless of the weights, so a
    correct RoPE implementation is genuinely position-invariant there.
    """
    torch.manual_seed(0)
    m = build_model(tiny_cfg).eval()
    a = torch.tensor([[3, 9, 14, 2, 7, 5]])
    b = torch.tensor([[3, 9, 14, 2, 5, 7]])  # last two swapped
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert torch.allclose(la[:, :4], lb[:, :4], atol=1e-5)      # prefix unaffected
    assert not torch.allclose(la[:, 5], lb[:, 5], atol=1e-4)    # suffix differs


def test_rope_offset_is_per_layer(tiny_cfg):
    """Regression: layer 0's cache write must not shift deeper layers' positions."""
    torch.manual_seed(0)
    m = build_model(tiny_cfg).eval()
    assert m.cfg.num_layers > 1
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        parts = [m(x[:, :6], cache=cache)[0]]
        for t in range(6, 12):
            parts.append(m(x[:, t : t + 1], cache=cache)[0])
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4)


def test_chunked_prefill_matches_full_forward(tiny_cfg):
    """t>1 against a non-empty cache needs the shifted causal mask."""
    torch.manual_seed(0)
    m = build_model(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 15))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        chunks = torch.cat(
            [m(x[:, a:b], cache=cache)[0] for a, b in ((0, 5), (5, 10), (10, 15))], dim=1
        )
    assert torch.allclose(full, chunks, atol=1e-4)


def test_save_and_reload_roundtrip(tiny_cfg, tmp_path):
    torch.manual_seed(0)
    m = build_model(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    with torch.no_grad():
        before, _ = m(x)
    path = m.save_pretrained(tmp_path / "ckpt")
    assert (path / "config.json").exists()
    assert (path / "model.safetensors").exists()

    m2 = SmallTalkModel.from_pretrained(path).eval()
    with torch.no_grad():
        after, _ = m2(x)
    assert torch.allclose(before, after, atol=1e-6)
    assert m2.num_parameters() == m.num_parameters()


def test_every_experiment_config_forward_backward(experiment_configs):
    for cfg in experiment_configs:
        m = build_model(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        _, loss = m(x, labels=x)
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads, cfg.name
        assert all(torch.isfinite(g).all() for g in grads), cfg.name
        assert torch.isfinite(loss), cfg.name


def test_initial_loss_is_near_uniform(experiment_configs):
    """A correctly initialised model starts around ln(vocab_size)."""
    import math

    cfg = experiment_configs[0]
    torch.manual_seed(0)
    m = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 64))
    with torch.no_grad():
        _, loss = m(x, labels=x)
    assert abs(float(loss) - math.log(cfg.vocab_size)) < 1.0


def test_long_context_up_to_max_positions():
    from smalltalk.config import ModelConfig

    cfg = ModelConfig(vocab_size=256, hidden_size=64, num_layers=2,
                      num_attention_heads=4, num_key_value_heads=1, head_dim=16,
                      intermediate_size=128, max_position_embeddings=1024)
    m = build_model(cfg).eval()
    for seq in (512, 1024):
        x = torch.randint(0, cfg.vocab_size, (1, seq))
        with torch.no_grad():
            logits, _ = m(x)
        assert logits.shape == (1, seq, cfg.vocab_size)
        assert torch.isfinite(logits).all()
