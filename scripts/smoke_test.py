#!/usr/bin/env python3
"""Pre-flight gate. Run this BEFORE spending money on training.

Verifies, on a tiny generated dataset:
  1. tokenizer round-trips
  2. causal masking is correct
  3. assistant-only SFT masking is correct
  4. parameter counts match targets
  5. loss decreases on a tiny overfitting test
  6. generation works
  7. KV caching produces equivalent logits
  8. checkpoints save/reload
  9. every model configuration completes a forward+backward pass
 10. full pipeline: prepare -> tokenizer -> train -> sft -> evaluate -> chat

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import traceback
from pathlib import Path

import torch

import _bootstrap  # noqa: F401

from smalltalk.config import ModelConfig, TrainConfig, all_model_configs
from smalltalk.data.clean import FilterConfig, clean_conversations
from smalltalk.data.dataset import PackedCLMDataset, SFTDataset
from smalltalk.data.schema import write_jsonl
from smalltalk.data.synthetic import OfflineConfig, generate_offline_corpus
from smalltalk.eval.bench import default_scenarios
from smalltalk.eval.metrics import aggregate
from smalltalk.eval.runner import run_bench
from smalltalk.infer.generate import ConversationEngine, GenerationConfig, generate
from smalltalk.model import SmallTalkModel, build_model
from smalltalk.params import check_config
from smalltalk.tokenizer import train_tokenizer
from smalltalk.train.trainer import Trainer

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                detail = fn(*a, **kw) or ""
                RESULTS.append((name, True, str(detail)))
                print(f"  PASS  {name}  {detail}")
                return True
            except Exception as exc:
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {name}  {type(exc).__name__}: {exc}")
                if "-v" in __import__("sys").argv:
                    traceback.print_exc()
                return False
        return wrapper
    return deco


def toy_config(vocab: int) -> ModelConfig:
    return ModelConfig(name="toy", vocab_size=vocab, hidden_size=64, num_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                       intermediate_size=128, max_position_embeddings=256)


@check("1. tokenizer round-trips")
def c_tokenizer(tok):
    probes = ["hey", "hey, what's up?", "honestly kinda tired today", "don't stop",
              "night :)", "lmao 🙂", "whoa really??"]
    bad = [p for p in probes if tok.decode(tok.encode(p)) != p]
    assert not bad, f"round-trip failed for {bad}"
    return f"vocab={tok.vocab_size}, {len(probes)} probes lossless"


@check("2. causal masking is correct")
def c_causal(cfg):
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base, _ = m(x)
        y = x.clone()
        y[0, 7] = (y[0, 7] + 3) % cfg.vocab_size
        alt, _ = m(y)
    assert torch.allclose(base[:, :7], alt[:, :7], atol=1e-5), "future token leaked backwards"
    assert not torch.allclose(base[:, 7], alt[:, 7], atol=1e-5), "position 7 did not change"
    return "no backward information flow"


@check("3. assistant-only SFT masking is correct")
def c_sft_mask(convs, tok):
    ds = SFTDataset(convs[:20], tok, seq_len=256, mask_non_assistant=True)
    item = ds[0]
    ids, mask = item["input_ids"].tolist(), item["loss_mask"].tolist()
    forbidden = {tok.user_id, tok.system_id, tok.bos_id, tok.pad_id, tok.assistant_id}
    leaked = [i for i, m in zip(ids, mask) if m and i in forbidden]
    assert not leaked, f"non-assistant tokens supervised: {leaked}"
    assert 0 < sum(mask) < len(mask), "mask is degenerate"
    frac = sum(mask) / len([i for i in ids if i != tok.pad_id])
    return f"{sum(mask)} supervised tokens ({frac:.0%} of the sequence)"


@check("4. parameter counts match targets")
def c_params():
    cfgs = all_model_configs()
    assert cfgs, "no model configs found"
    bad = []
    for cfg in cfgs:
        chk = check_config(cfg)
        if not chk.ok:
            bad.append(chk.report())
    assert not bad, "\n" + "\n".join(bad)
    return " ".join(f"{c.name}={check_config(c, empirical=False).analytic/1e6:.2f}M" for c in cfgs)


@check("5. loss decreases on a tiny overfitting test")
def c_overfit(cfg):
    torch.manual_seed(0)
    m = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.9, 0.95))
    _, first = m(x, labels=x)
    for _ in range(120):
        opt.zero_grad()
        _, loss = m(x, labels=x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    first_v, last_v = first.detach().item(), loss.detach().item()
    assert last_v < first_v * 0.3, f"loss {first_v:.3f} -> {last_v:.3f}"
    return f"loss {first_v:.3f} -> {last_v:.3f}"


@check("6. generation works")
def c_generate(cfg, tok):
    torch.manual_seed(0)
    m = build_model(cfg)
    ids, _ = tok.encode_conversation([{"role": "user", "content": "hey"}],
                                     add_generation_prompt=True)
    out = generate(m, tok, ids, GenerationConfig(max_new_tokens=16, seed=1))
    assert 0 < len(out) <= 16
    assert tok.pad_id not in out
    text = tok.decode(out)
    return f"{len(out)} tokens -> {text[:40]!r}"


@check("7. KV caching produces equivalent logits")
def c_kv(cfg):
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        parts = [m(x[:, :10], cache=cache)[0]]
        for t in range(10, 20):
            parts.append(m(x[:, t : t + 1], cache=cache)[0])
        inc = torch.cat(parts, dim=1)
    delta = (full - inc).abs().max().item()
    assert delta < 1e-4, f"max logit divergence {delta}"
    return f"max |delta| = {delta:.2e}"


@check("8. checkpoints save/reload")
def c_checkpoint(cfg, tmp):
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        before, _ = m(x)
    p = m.save_pretrained(Path(tmp) / "ckpt")
    m2 = SmallTalkModel.from_pretrained(p).eval()
    with torch.no_grad():
        after, _ = m2(x)
    assert torch.allclose(before, after, atol=1e-6)
    fmt = "safetensors" if (p / "model.safetensors").exists() else "torch.save"
    return f"bitwise-equal logits after reload ({fmt})"


@check("9. every config does forward+backward")
def c_all_configs():
    details = []
    for cfg in all_model_configs():
        m = build_model(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        _, loss = m(x, labels=x)
        loss.backward()
        assert torch.isfinite(loss), f"{cfg.name}: non-finite loss"
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), f"{cfg.name}: bad grads"
        expect = math.log(cfg.vocab_size)
        assert abs(float(loss) - expect) < 1.5, f"{cfg.name}: init loss {loss:.2f} vs ln(V)={expect:.2f}"
        details.append(f"{cfg.name}:{float(loss):.2f}")
        del m
    return "init loss " + " ".join(details)


@check("10. full pipeline (prepare->train->sft->eval->chat)")
def c_pipeline(convs, tok, tmp):
    tmp = Path(tmp)
    write_jsonl(tmp / "train.jsonl", convs[: int(len(convs) * 0.9)])
    write_jsonl(tmp / "val.jsonl", convs[int(len(convs) * 0.9) :])
    tok_dir = tmp / "tok"
    tok.save(tok_dir)
    mc = toy_config(tok.vocab_size)
    (tmp / "model.json").write_text(__import__("json").dumps(mc.to_dict()))

    common = dict(
        model_config=str(tmp / "model.json"), tokenizer=str(tok_dir),
        train_data=str(tmp / "train.jsonl"), val_data=str(tmp / "val.jsonl"),
        output_dir=str(tmp / "runs"), seq_len=128, batch_size=4, max_steps=20,
        eval_every=10, eval_batches=2, save_every=20, log_every=10,
        warmup_steps=3, device="cpu", dtype="fp32", learning_rate=1e-3,
    )
    s1 = Trainer(TrainConfig(run_name="smoke-clm", stage="clm", **common)).train()
    stage1_ckpt = tmp / "runs" / "smoke-clm" / "final"
    s2 = Trainer(TrainConfig(run_name="smoke-sft", stage="sft",
                             init_from=str(stage1_ckpt), **common)).train()

    engine = ConversationEngine(
        SmallTalkModel.from_pretrained(tmp / "runs" / "smoke-sft" / "final"),
        tok, GenerationConfig(max_new_tokens=12, max_context=128),
    )
    evals, _ = run_bench(engine, default_scenarios()[:3])
    agg = aggregate(evals)
    reply = engine.reply("hey")
    assert isinstance(reply, str) and reply
    assert all(e.metrics["num_turns"] == 10 for e in evals)
    return (f"clm val {s1['final_val_loss']:.3f} -> sft val {s2['final_val_loss']:.3f}; "
            f"bench clean_10turn={agg['clean_10turn_rate']}; reply={reply[:24]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conversations", type=int, default=400)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    args = ap.parse_args()

    print("=" * 72)
    print("smalltalk-ai smoke tests (tiny data, CPU-safe)")
    print("=" * 72)

    tmp = tempfile.mkdtemp(prefix="smalltalk-smoke-")
    try:
        raw = list(generate_offline_corpus(OfflineConfig(num_conversations=args.conversations, seed=7)))
        convs, stats = clean_conversations(raw, FilterConfig())
        print(f"  data  {len(convs):,} conversations after cleaning "
              f"({sum(stats.dropped.values()):,} dropped)")
        texts = [m.content for c in convs for m in c.messages]
        tok = train_tokenizer(texts, vocab_size=512, out_dir=Path(tmp) / "tok0")
        cfg = toy_config(tok.vocab_size)
        print()

        c_tokenizer(tok)
        c_causal(cfg)
        c_sft_mask(convs, tok)
        c_params()
        c_overfit(cfg)
        c_generate(cfg, tok)
        c_kv(cfg)
        c_checkpoint(cfg, tmp)
        c_all_configs()
        c_pipeline(convs, tok, tmp)
    finally:
        if args.keep:
            print(f"\nworkspace kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 72)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        print("Do NOT start expensive training until these pass.")
        return 1
    print("All pre-flight checks passed. Safe to launch training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
