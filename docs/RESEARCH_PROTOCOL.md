# Research protocol

The primary study asks how small a generative model can be while sustaining
natural casual conversation. The student architecture and tokenizer are frozen;
data entropy, context length, teacher distillation, failure mining, and
preference optimization are experimental factors.

The current immutable benchmark is SmallTalkBench-v2 (`396` scenarios, `18`
skills, checksum `a2ce68928e780ce5`). It must never be used as training data or
edited to improve a score. Family-level data splitting and fail-closed leakage
checks are required because example-level validation previously produced
83.5%-verbatim train/validation overlap.

See `docs/TEACHER_SPEC_V2.md` for upstream-data specification and the root
`EXPERIMENTS.md` for provenance and release requirements.
