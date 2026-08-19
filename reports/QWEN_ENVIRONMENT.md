# QWEN_ENVIRONMENT — local inference audit

Phase 1 audit for replacing the blocked Claude-API corpus phase with a local
open-weight upstream teacher (`Qwen/Qwen3.5-9B`, post-trained).

## Hardware

| item | value |
|---|---|
| APU | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S |
| GPU arch | **gfx1151** (Strix Halo), 40 CU |
| GPU VRAM pool | **48.0 GB** (51,539,607,552 B) — BIOS carve-out from unified memory |
| VRAM in use at audit | 0.87 GB → **~47 GB free** |
| System RAM (OS-visible) | **14 GB total, ~8.1 GB available** |
| Disk free (after cleanup) | **39 GB** |

**The binding constraints are system RAM and disk, not VRAM.** This is a unified-memory
APU: the 48 GB "VRAM" is carved out of the same physical DRAM, leaving the OS only 14 GB.
Model loading must stream weights disk→GPU (safetensors mmap + `device_map`) rather than
materialising a 19 GB state dict in CPU RAM, which would OOM.

## Software

| item | value |
|---|---|
| ROCm (system) | 7.1.0 |
| PyTorch | 2.10.0+rocm7.0 (HIP 7.0.51831), `cuda.is_available() == True` |
| Python | 3.14.4 |
| transformers | **5.15.1** |
| accelerate | installed |
| huggingface_hub | 1.28.0 |
| tokenizers / safetensors | 0.23.1 / 0.8.0 |
| vLLM | **not installed** |
| llama.cpp | **not installed** |

Note: PyTorch here is the ROCm wheel installed into `.venv-rocm`. It requires
`LD_PRELOAD` of the *system* `libhsa-runtime64.so` — the wheel's bundled ROCm 7.0 HSA
runtime segfaults against the 7.1 driver stack. `scripts/rocm.sh` applies this.

## Model: `Qwen/Qwen3.5-9B`

Pinned revision **`c202236235762e1c871ad0ccb60c8ee5ba337b9a`**. 19.33 GB across 4
safetensors shards.

| field | value |
|---|---|
| architecture | `Qwen3_5ForConditionalGeneration` (**multimodal wrapper**) |
| model_type | `qwen3_5` |
| hidden / layers | 4096 / 32 |
| attention | **hybrid: 24 `linear_attention` + 8 `full_attention`** (`full_attention_interval: 4`) |
| heads (Q/KV) | 16 / 4, head_dim 256 |
| intermediate | 12288 |
| vocab | **248,320** (incompatible with our 4096 student vocab — hence the 50M bridge) |
| max positions | 262,144 |
| dtype | bfloat16 |

## Backend decision

**Selected: Transformers + ROCm, BF16, text-only.**

Rationale, in the priority order given (quality first):

1. **BF16 fits comfortably.** ~19 GB weights against ~47 GB free VRAM. No quantisation
   is needed, so we take option 1 of the stated preference order and avoid any
   quantisation quality loss.
2. **vLLM rejected — not installed, and gfx1151 is not a supported vLLM target.**
   Building it against ROCm 7.1 + Python 3.14 would be the "hours forcing an
   incompatible configuration" the brief warns against. SGLang (the upstream README's
   recommendation) has the same problem and additionally requires a from-source main-branch
   build.
3. **The linear-attention path has a pure-PyTorch fallback.** `modeling_qwen3_5` decorates
   its kernels with `@use_kernel_func_from_hub_with_fallback(...)` over `fla` and
   `causal_conv1d`, falling back to `torch_chunk_gated_delta_rule` /
   `torch_recurrent_gated_delta_rule` / `causal_conv1d_fn`. Those Triton/CUDA kernels are
   not available for gfx1151, so we run the torch path: **correct, just slower**. Given
   corpus quality outranks throughput, this is the right trade.

Text-only loading uses the `qwen3_5_text` sub-config so the vision tower is never
placed on the GPU (the weights still download — they are interleaved across the shards —
but are not resident).

## Risks carried into Phase 2+

| risk | mitigation |
|---|---|
| Torch-fallback linear attention is slow | Batch aggressively; measure tok/s in the smoke test before sizing the pilot. Generation is a one-off cost. |
| 14 GB system RAM during load | `device_map={"":0}` + safetensors mmap; never `.to()` a CPU-materialised model. |
| 39 GB disk, 19.3 GB model | Corpus is small (JSONL). Qwen must be **unloaded before teacher training**, per the GPU-scheduling rule. |
| Python 3.14 is very new | transformers 5.15.1 imports and registers `qwen3_5` cleanly; verified before download. |
| Qwen style fingerprints | Latent planner supplies structure; sampler study + critic gate on assistant-speak. |
