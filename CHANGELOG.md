# Changelog

All notable changes to this project will be documented here. The project uses
semantic versioning for source releases; checkpoints and datasets have separate
versioned manifests.

## [Unreleased]

### v0.2 development

- Fixed sliding KV-cache absolute RoPE position tracking; trimming now cannot
  reuse positional phases during long generation.
- Made packed CLM conversation boundaries causal-isolated with segment block masks.
- Replaced example-level generated-family splitting with append-stable family hash
  partitions and made trainer losses/token accounting token-weighted.
- Extended the fail-closed benchmark leakage gate to frozen SmallTalkBench-v2.
- Added the verified 50,021,888-parameter native-tokenizer teacher configuration
  and online CE+KL student-distillation implementation.
- Guarded the legacy automatic loop from silently selecting against a frozen bench.

### v0.1 public release

- Added public repository policies, CI, issue forms, model/data cards, and
  repository hygiene checks.
- Corrected public documentation to describe SmallTalkBench-v2 (396 scenarios,
  18 skills) and the frozen 6,689,024-parameter student.

## [0.1.0] - 2026-08-19

- Initial research repository snapshot.
- Llama-compatible tiny decoder implementations and exact parameter contracts.
- Tokenizer, data adapters, cleaning, training, SFT, distillation, inference,
  evaluation, and quantization infrastructure.
