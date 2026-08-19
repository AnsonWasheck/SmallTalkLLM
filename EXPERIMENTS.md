# Experiment and release protocol

## Required experiment record

Each run should have a machine-readable record containing:

- git commit and source release;
- model config and analytical/instantiated parameter counts;
- tokenizer path, vocabulary size, and checksum;
- dataset manifest/checksum, family split, and leakage report;
- initialization checkpoint and random seed;
- optimizer, learning rate, schedule, batch, accumulation, dtype, and sequence length;
- token count, steps, loss masking, distillation/DPO settings;
- checkpoint path and raw evaluation output;
- hardware and software environment.

## Comparisons

Architecture, tokenizer, and SmallTalkBench-v2 are frozen for the primary study.
Compare one variable at a time where practical: scratch vs warm start, dense CE
vs distillation, 512 vs 1024 context, offline vs on-policy distillation, and
pre-DPO vs post-DPO.

## Branching

Use `exp/<name>` for a controlled experiment. Merge only reproducible
infrastructure or documented findings into `develop`; promote to `main` through
a release PR. Never force-push `main` or rewrite an experiment's provenance.

## Release checklist

1. `make test`, `make smoke`, `make params`, and `make audit` pass.
2. Frozen benchmark checksum is unchanged.
3. No checkpoints, raw/private data, credentials, or local environments are tracked.
4. Model/data cards and limitations are updated.
5. Raw evaluation outputs and manifest checksums are archived outside git or attached to the release.
6. Tag the source release and record the exact checkpoint artifact separately.
