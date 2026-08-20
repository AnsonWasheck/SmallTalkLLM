# smalltalk-ai

**A research repository for finding the smallest language model that can sustain natural, casual one-on-one conversation.**

`smalltalk-ai` is an open, reproducible attempt to study social fluency at very small parameter counts. The project deliberately does not optimize for factual knowledge, mathematics, coding, tool use, long-form writing, or general assistant behavior. The target is ordinary conversation: acknowledgement, emotional reaction, nearby memory, follow-up questions, topic drift, repair, light humor, and natural goodbyes.

> The released student model is frozen at exactly **6,689,024 trainable parameters**.

## Current line: v0.2-Core

The project has shifted objective. Earlier rounds optimized for conversational
*diversity*, which at 6.7M parameters spread probability mass across a dozen
equally-plausible replies and made greedy decoding a coin flip. v0.2-Core
optimizes for *reliability* instead:

> Given an ordinary conversational input, choose an appropriate short response
> almost every time.

The design rule is **broad input paraphrases, narrow output behaviour**. Each
intent has one canonical target; a wider accept set is used only for scoring.
Conditional entropy is driven down on purpose.

**Core-Bench** (`smalltalk/core/bench_core.py`) measures this: 342 scenarios
built from held-out paraphrases the generator never emits, crossed with fixed
surface transforms, scored at temperature 0. No RNG anywhere, so two runs of one
checkpoint are byte-identical. Checksum `33a35a049b672226`.

| round | overall | tier 1 | tier 2 | tier 3 |
| --- | --- | --- | --- | --- |
| baseline | 0.550 | 0.767 | 0.488 | 0.196 |
| r001 | 0.594 | 0.760 | 0.506 | 0.451 |
| **r002 (current)** | **0.629** | 0.775 | 0.568 | 0.451 |
| target | — | 0.99 | 0.95 | 0.90 |

The current primary checkpoint is recorded in
[CURRENT_MODEL.json](CURRENT_MODEL.json), which is verified by tests against the
frozen benchmark checksum and against the archive refs, so a stale number cannot
silently look authoritative. Superseded models are archived, never deleted.

r001 added an `out_of_scope` class after a measured failure: with 20 intents and
no way to decline, *"what is the square root of nine"* and *"my hovercraft is
full of eels"* both returned *"that sounds rough"*. That class went 0.00 → 0.556
without costing tiers 1 or 2.

Training runs unattended via `scripts/core_loop.py`, which verifies the benchmark
checksum every round and halts rather than report an incomparable number. It may
change the curriculum but never the test, and never promotes a regression.

```bash
python scripts/core_loop.py --status      # read the experiment ledger
```

## Project status

This repository is **research alpha**. The architecture, tokenizer, benchmark, training code, and reproducibility scaffolding are public. Checkpoints and downloaded/private datasets are intentionally excluded from git. Results are versioned in reports and experiment manifests rather than presented as production claims.

## Quick start

```bash
git clone https://github.com/AnsonWasheck/SmallTalkLLM.git
cd SmallTalkLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

make params
make test
make smoke
```

The smoke gate is CPU-safe and should pass before expensive training. It checks tokenizer round trips, causal masking, assistant-only masking, exact parameter counts, tiny-set loss descent, generation, KV-cache equivalence, checkpoint reload, and forward/backward coverage for every documented model.

## Try a checkpoint

Checkpoints are not stored in this source repository. Download a compatible release asset or point the command at a local checkpoint directory:

```bash
python scripts/chat.py \
  --checkpoint /path/to/smalltalk-7m/best \
  --tokenizer /path/to/tokenizer-4096 \
  --device auto
```

For the browser demo:

```bash
python scripts/chat.py --checkpoint /path/to/checkpoint \
  --tokenizer /path/to/tokenizer-4096 --web --port 8000
```

Decoding controls include temperature, top-p, top-k, repetition penalty, maximum response tokens, and an optional zero-parameter application state store. State is off by default so naked-model and production-system results remain separate.

## Frozen student architecture

The public research target is `smalltalk-7m`; its architecture must not change across experiments:

| property | value |
|---|---:|
| trainable parameters | **6,689,024** |
| vocabulary | 4,096 byte-level BPE tokens |
| layers | 8 |
| hidden size | 256 |
| attention heads | 4 query / 1 key-value |
| head dimension | 64 |
| SwiGLU intermediate size | 704 |
| maximum context | 1,024 tokens |
| normalization | pre-norm RMSNorm |
| positional encoding | RoPE, theta 10,000 |
| embeddings | tied input/output |
| linear layers | bias-free |
| dropout | none |

Verify all contracts with:

```bash
python scripts/count_params.py --breakdown
```

The command checks analytical and instantiated module counts and exits nonzero on any mismatch. It never silently changes a configuration.

## Research pipeline

```text
latent scenario planner → high-entropy dialogue writer → independent critic
→ deterministic gates → family-disjoint corpus → dense causal pretraining
→ assistant SFT → teacher distillation → student failure mining → DPO → int4
```

The repository includes adapters and infrastructure for DailyDialog, EmpatheticDialogues, arbitrary JSONL conversations, scenario planning, Qwen upstream generation, family-level splitting, leakage detection, distillation, evaluation, and quantization. See [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md), [docs/DESIGN_CHOICES.md](docs/DESIGN_CHOICES.md), and [EXPERIMENTS.md](EXPERIMENTS.md).

## Data and licensing

The source tree contains only small seed examples and schemas. Full datasets, downloaded models, generated corpora, and checkpoints are excluded by design. Read [DATA_CARD.md](DATA_CARD.md) before downloading or redistributing data. Third-party datasets and upstream model weights retain their own licenses; this repository's MIT license applies to project code, not automatically to those assets.

Every training corpus should carry source provenance, licenses, generator/model revision, family IDs, split assignment, tokenizer token counts, filters, deduplication statistics, and checksums. The pipeline fails closed on benchmark leakage.

## Evaluation

Two instruments, deliberately measuring different things.

**Core-Bench** — reliability on conversational primitives. Deterministic, frozen
by checksum, tier targets 99/95/90%.

```bash
python scripts/eval_core.py --checkpoint /path/to/checkpoint --tag my-run
```

Frozen manifests live in `benchmarks/`, not `artifacts/`: a score is meaningless
without the checksum it was measured under, so they are part of the durable
record rather than regenerable output.

**SmallTalkBench-v2** — trajectory quality. SmallTalkBench-v2 contains **396 scenarios across 18 skills** and has frozen checksum `a2ce68928e780ce5`. It is not training data. It evaluates complete trajectories with separate metrics for semantic coherence, long-range memory, state updates, absent-memory discrimination, epistemic correctness, emotional response, repair, ambiguity, response length, repetition, lexical diversity, question overuse, and stopping.

Run the frozen evaluator:

```bash
python scripts/eval_bench_v2.py --checkpoint /path/to/checkpoint \
  --tag my-run --device auto
```

Raw transcripts, aggregates, per-skill scores, confidence intervals, length distributions, diversity metrics, memory metrics, and epistemic metrics are saved under `reports/evals/<tag>/`. Never select a model from one stochastic sample or raw validation loss alone.

## Training

```bash
python scripts/prepare_data.py --help
python scripts/train_tokenizer.py --help
python scripts/train.py --config configs/train/stage1_7m.yaml
python scripts/sft.py --config configs/train/sft_7m.yaml
python scripts/evaluate.py --help
```

For the AMD ROCm environment used during development:

```bash
bash scripts/rocm.sh scripts/check_device.py
bash scripts/rocm.sh scripts/train.py --config configs/train/stage1_7m.yaml
```

The launcher is hardware-specific; run the device check on your own system.

## Repository layout

```text
smalltalk-ai/
├── smalltalk/             model, tokenizer, data, training, inference, eval
├── configs/               versioned model and training YAML
├── scripts/               reproducible CLI entry points
├── tests/                 unit, smoke, and contract tests
├── data/seed/             tiny redistributable examples only
├── docs/                  protocol, decisions, and research specifications
├── reports/               checked-in methodology reports
├── .github/               CI, issue forms, PR template, Dependabot
├── MODEL_CARD.md          scope, limitations, and intended use
├── DATA_CARD.md           source-data and provenance policy
└── CONTRIBUTING.md        branch, release, and integrity workflow
```

## Versioning and branches

- `main`: stable public releases only;
- `develop`: integration branch for the next release;
- `archive/<line>-<version>`: superseded lines, preserved not deleted;
- `exp/<name>`: one controlled experiment or ablation;
- `release/<version>`: short-lived stabilization branch;
- `vMAJOR.MINOR.PATCH`: reproducible source releases.

Commits follow the scheme in [docs/NAMING.md](docs/NAMING.md), including an
`exp(...)` type that records a measurement in the subject line so `git log
--oneline` reads as an experiment history.

Model/data experiments are versioned independently of source releases. Each result must record the git commit, config, exact parameter count, tokenizer identifier, dataset manifest/checksum, seed, optimizer, token budget, checkpoint path, and evaluation tag. See [EXPERIMENTS.md](EXPERIMENTS.md).

## Limitations

This is not a general-purpose assistant and must not be used for medical, legal, financial, safety-critical, or emotionally dependent applications. Tiny models can produce fluent but wrong, repetitive, insensitive, or contextually broken replies. Do not upload private conversations to public issues or untrusted training runs.

The research question is: **what data, training, and inference recipe gives the most socially fluent behavior per byte while the student remains exactly 6,689,024 parameters?**

## Citation

See [CITATION.cff](CITATION.cff). Cite the exact release tag and include the dataset manifest and evaluation report with published results.
