"""Quantization: round-trip fidelity, packing correctness, real size reduction."""

import torch

from smalltalk.model import build_model
from smalltalk.params import deployed_bytes, fits_budget, max_params_for_budget
from smalltalk.quantize import (
    artifact_bytes,
    dequantize_tensor,
    load_quantized,
    pack_int4,
    quantize_tensor,
    save_quantized,
    unpack_int4,
)


def test_int4_packing_roundtrip():
    codes = torch.randint(-8, 8, (7, 10), dtype=torch.int8)
    packed = pack_int4(codes)
    assert packed.numel() <= (codes.numel() + 1) // 2 + codes.shape[0]
    assert torch.equal(unpack_int4(packed, codes.shape[1]), codes)


def test_int4_packing_handles_odd_widths():
    codes = torch.randint(-8, 8, (3, 5), dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(codes), 5), codes)


def test_quantize_preserves_magnitude():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    for bits, tol in ((8, 0.02), (4, 0.20)):
        codes, scale = quantize_tensor(w, bits)
        recon = dequantize_tensor(codes, scale)
        rel = (recon - w).abs().mean() / w.abs().mean()
        assert rel < tol, (bits, float(rel))
        assert codes.min() >= -(2 ** (bits - 1)) and codes.max() <= 2 ** (bits - 1) - 1


def test_zero_rows_do_not_produce_nans():
    w = torch.zeros(4, 8)
    codes, scale = quantize_tensor(w, 4)
    assert torch.isfinite(dequantize_tensor(codes, scale)).all()


def test_save_load_quantized_model(tiny_cfg, tmp_path):
    torch.manual_seed(0)
    m = build_model(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 12))
    with torch.no_grad():
        ref, _ = m(x)

    for precision, tol in (("int8", 0.15), ("int4", 1.5)):
        out = tmp_path / precision
        report = save_quantized(m, out, precision)
        assert report.compression > (3.0 if precision == "int8" else 6.0)
        m2 = load_quantized(out).eval()
        with torch.no_grad():
            got, _ = m2(x)
        # logits shift, but must stay correlated and finite
        assert torch.isfinite(got).all()
        rel = (got - ref).abs().mean() / ref.abs().mean()
        assert rel < tol, (precision, float(rel))
        corr = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1]
        assert corr > 0.9, (precision, float(corr))


def test_quantized_artifact_is_smaller_on_disk(tiny_cfg, tmp_path):
    m = build_model(tiny_cfg).eval()
    fp32 = m.save_pretrained(tmp_path / "fp32")
    save_quantized(m, tmp_path / "int4", "int4")
    assert artifact_bytes(tmp_path / "int4") < artifact_bytes(fp32) / 5


def test_budget_helpers():
    from smalltalk.config import load_model_config

    cfg7 = load_model_config("smalltalk-7m")
    assert fits_budget(cfg7, 4.0, "int4")
    assert not fits_budget(cfg7, 4.0, "int8")
    assert deployed_bytes(cfg7, "int4") < deployed_bytes(cfg7, "int8")
    assert max_params_for_budget(4.0, "int4") > max_params_for_budget(4.0, "int8")
