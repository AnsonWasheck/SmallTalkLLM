#!/usr/bin/env python3
"""Report the training backend actually available, and prove it can train.

Works for NVIDIA CUDA, AMD ROCm/HIP, Apple MPS and CPU. On AMD, PyTorch exposes
the GPU as device type `cuda` (that is not a bug -- ROCm reuses the CUDA API), so
`--device cuda` and `--device auto` both select the AMD GPU.
"""

from __future__ import annotations

import torch

import _bootstrap  # noqa: F401

from smalltalk.config import load_model_config
from smalltalk.model import build_model
from smalltalk.train.utils import describe_device, is_rocm, resolve_device


def main() -> int:
    print(f"torch            {torch.__version__}")
    print(f"cuda build       {torch.version.cuda}")
    print(f"hip/rocm build   {torch.version.hip}")
    print(f"cuda.is_available {torch.cuda.is_available()}")
    print(f"device count     {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    mps = getattr(torch.backends, "mps", None)
    print(f"mps available    {bool(mps and mps.is_available())}")

    dev = resolve_device("auto")
    print(f"\nselected device  {dev}  ->  {describe_device(dev)}")

    if is_rocm():
        print("backend          AMD ROCm (addressed as device type 'cuda')")
    elif dev.type == "cuda":
        print("backend          NVIDIA CUDA")
    else:
        print("backend          " + dev.type)
        if not torch.cuda.is_available():
            print("\nNOTE: this is a CPU-only PyTorch build. To train on an AMD GPU install\n"
                  "the ROCm wheel, e.g.:\n"
                  "  pip install --index-url https://download.pytorch.org/whl/rocm6.2 \\\n"
                  "      torch --force-reinstall\n"
                  "then re-run this script; cuda.is_available should become True and\n"
                  "hip/rocm build should be non-None.")

    # real forward+backward on the selected device
    cfg = load_model_config("smalltalk-4m")
    model = build_model(cfg).to(dev)
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    x = torch.randint(0, cfg.vocab_size, (2, 128), device=dev)
    with torch.autocast(device_type=dev.type, dtype=dtype, enabled=dtype != torch.float32):
        _, loss = model(x, labels=x)
    loss.backward()
    print(f"\nsmoke train step ok: {cfg.name} loss={loss.detach().item():.4f} dtype={dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
