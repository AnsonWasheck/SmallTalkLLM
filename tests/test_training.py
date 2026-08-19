"""Loss goes down on a tiny overfitting test; checkpoints resume; masking works
end-to-end through the real trainer."""

import json
import math

import torch

from smalltalk.config import ModelConfig, TrainConfig
from smalltalk.data.schema import write_jsonl
from smalltalk.data.synthetic import OfflineConfig, generate_offline_corpus
from smalltalk.model import build_model
from smalltalk.train.utils import build_optimizer, lr_at_step, resolve_device
from smalltalk.train.distillation import causal_ce_and_kl


def test_tiny_overfit_drives_loss_down(tiny_cfg):
    """The classic sanity check: one batch, memorise it."""
    torch.manual_seed(0)
    m = build_model(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 24))
    opt = build_optimizer(m, 3e-3, (0.9, 0.95), 0.0)

    _, first = m(x, labels=x)
    for _ in range(120):
        opt.zero_grad()
        _, loss = m(x, labels=x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    first_v, last_v = first.detach().item(), loss.detach().item()
    assert last_v < first_v * 0.3, (first_v, last_v)
    assert last_v < 1.0


def test_masked_loss_only_sees_masked_positions(tiny_cfg):
    torch.manual_seed(0)
    m = build_model(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    mask = torch.zeros(1, 12, dtype=torch.long)
    mask[0, 6:] = 1

    _, masked_loss = m(x, labels=x, loss_mask=mask)
    masked_loss.backward()
    g1 = m.embed_tokens.weight.grad.clone()

    m.zero_grad()
    # changing an unsupervised *label* must not change the masked loss
    y = x.clone()
    y[0, 3] = (y[0, 3] + 1) % tiny_cfg.vocab_size
    # position 3 is both input and label; use labels-only variation instead
    labels = x.clone()
    labels[0, 2] = (labels[0, 2] + 1) % tiny_cfg.vocab_size
    _, loss2 = m(x, labels=labels, loss_mask=mask)
    assert abs(float(loss2) - float(masked_loss)) < 1e-6
    assert g1.abs().sum() > 0


def test_zero_mask_is_safe(tiny_cfg):
    m = build_model(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    _, loss = m(x, labels=x, loss_mask=torch.zeros(1, 8, dtype=torch.long))
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_token_level_distillation_masks_and_backpropagates(tiny_cfg):
    torch.manual_seed(0)
    student = build_model(tiny_cfg)
    teacher = build_model(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    mask = torch.zeros_like(x)
    mask[:, 6:] = 1
    student_logits, _ = student(x)
    with torch.no_grad():
        teacher_logits, _ = teacher(x)
    loss, metrics = causal_ce_and_kl(student_logits, teacher_logits, x, mask,
                                     alpha=0.5, temperature=2.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["tokens"] == int(mask[:, 1:].sum())
    assert student.embed_tokens.weight.grad is not None


def test_lr_schedule_shape():
    peak, total, warm = 5e-4, 100, 10
    assert lr_at_step(0, total, peak, warm, 0.1, "cosine") < peak
    assert abs(lr_at_step(warm - 1, total, peak, warm, 0.1, "cosine") - peak) < 1e-12
    mid = lr_at_step(55, total, peak, warm, 0.1, "cosine")
    end = lr_at_step(total - 1, total, peak, warm, 0.1, "cosine")
    assert peak > mid > end >= peak * 0.1 - 1e-12
    assert lr_at_step(50, total, peak, warm, 0.1, "constant") == peak


def test_no_weight_decay_on_1d_params(tiny_cfg):
    m = build_model(tiny_cfg)
    opt = build_optimizer(m, 1e-3, (0.9, 0.95), 0.1)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])


def _write_tiny_data(tmp_path, tokenizer, n=200):
    convs = list(generate_offline_corpus(OfflineConfig(num_conversations=n, seed=5)))
    write_jsonl(tmp_path / "train.jsonl", convs[: int(n * 0.9)])
    write_jsonl(tmp_path / "val.jsonl", convs[int(n * 0.9) :])
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    return tok_dir


def _train_cfg(tmp_path, tok_dir, stage, tokenizer, **kw):
    mc = ModelConfig(name="toy", vocab_size=tokenizer.vocab_size, hidden_size=64,
                     num_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, intermediate_size=128, max_position_embeddings=128)
    (tmp_path / "model.yaml").write_text(json.dumps(mc.to_dict()))
    cfg = TrainConfig(
        run_name=kw.pop("run_name", f"t-{stage}"), stage=stage,
        model_config=str(tmp_path / "model.json"),
        tokenizer=str(tok_dir),
        train_data=str(tmp_path / "train.jsonl"),
        val_data=str(tmp_path / "val.jsonl"),
        output_dir=str(tmp_path / "runs"),
        seq_len=64, batch_size=4, max_steps=8, eval_every=4, eval_batches=2,
        save_every=4, log_every=4, warmup_steps=2, device="cpu", dtype="fp32",
        **kw,
    )
    (tmp_path / "model.json").write_text(json.dumps(mc.to_dict()))
    return cfg, mc


def test_trainer_runs_all_stages_and_saves(tmp_path, tokenizer):
    from smalltalk.train.trainer import Trainer

    tok_dir = _write_tiny_data(tmp_path, tokenizer)
    for stage in ("clm", "sft"):
        cfg, _ = _train_cfg(tmp_path, tok_dir, stage, tokenizer)
        summary = Trainer(cfg).train()
        run = tmp_path / "runs" / cfg.run_name
        assert (run / "final" / "model.safetensors").exists()
        assert (run / "final" / "tokenizer" / "tokenizer.json").exists()
        assert (run / "log.jsonl").exists() and (run / "log.csv").exists()
        assert summary["steps"] == 8
        assert math.isfinite(summary["final_val_loss"])


def test_checkpoint_resume_restores_step(tmp_path, tokenizer):
    from smalltalk.train.trainer import Trainer

    tok_dir = _write_tiny_data(tmp_path, tokenizer)
    cfg, _ = _train_cfg(tmp_path, tok_dir, "clm", tokenizer, run_name="resume")
    Trainer(cfg).train()
    ckpt = tmp_path / "runs" / "resume" / "final"

    cfg2, _ = _train_cfg(tmp_path, tok_dir, "clm", tokenizer, run_name="resume")
    cfg2.max_steps = 12
    cfg2.resume = str(ckpt)
    t2 = Trainer(cfg2)
    assert t2.state.step == 8
    assert t2.train().get("steps") == 12


def test_determinism_same_seed_same_loss(tmp_path, tokenizer):
    from smalltalk.train.trainer import Trainer

    tok_dir = _write_tiny_data(tmp_path, tokenizer)
    losses = []
    for i in range(2):
        cfg, _ = _train_cfg(tmp_path, tok_dir, "clm", tokenizer, run_name=f"det{i}")
        losses.append(Trainer(cfg).train()["final_val_loss"])
    assert abs(losses[0] - losses[1]) < 1e-6


def test_tokenizer_vocab_mismatch_is_rejected(tmp_path, tokenizer):
    from smalltalk.train.trainer import Trainer

    tok_dir = _write_tiny_data(tmp_path, tokenizer)
    cfg, mc = _train_cfg(tmp_path, tok_dir, "clm", tokenizer)
    mc.vocab_size = tokenizer.vocab_size + 64
    (tmp_path / "model.json").write_text(json.dumps(mc.to_dict()))
    try:
        Trainer(cfg)
    except ValueError as exc:
        assert "vocab" in str(exc)
    else:
        raise AssertionError("expected a vocab-mismatch error")


def test_device_resolution_never_crashes():
    d = resolve_device("auto")
    assert d.type in ("cuda", "mps", "cpu")
    assert resolve_device("cpu").type == "cpu"
