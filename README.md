# smalltalk-ai

**A research repository for finding the smallest language model that can sustain natural, casual one-on-one conversation.**

`smalltalk-ai` is an open, reproducible attempt to study social fluency at very small parameter counts. The project deliberately does not optimize for factual knowledge, mathematics, coding, tool use, long-form writing, or general assistant behavior. The target is ordinary conversation: acknowledgement, emotional reaction, nearby memory, follow-up questions, topic drift, repair, light humor, and natural goodbyes.

> The released student model is frozen at exactly **6,689,024 trainable parameters**.

> **Status: `main` is unstable and moving.** This is an active research line, not
> a released product. The checkpoint `main` points at is the best *demonstrated*
> one, not the newest, and several benchmarks on this branch have been
> deliberately retired and replaced as we learned they measured the wrong thing.
> Scores from different benchmark checksums are not comparable. What `main`
> represents is the direction the work is going, with the evidence for why.

## Where the project is: v0.4, elaboration

The objective has moved three times, each time because measurement contradicted
a design assumption. That history is the useful part of this repository, so it is
recorded rather than tidied away.

| line | objective | why it ended |
| --- | --- | --- |
| v0.1 | conversational diversity | at 6.7M, spread probability mass and made greedy decoding a coin flip |
| v0.2-Core | reliability: one canonical reply per intent | worked, then became the cause of the next problem |
| v0.3-Core-State | short-horizon state persistence | achieved; exposed that reliability had made the model deterministic |
| **v0.4** | **elaboration: reuse what the user actually said** | current |

The v0.2 design rule was *broad input paraphrases, narrow output behaviour*. It
bought reliability and it is exactly why the model later felt mechanical: with
one canonical target per intent, the model says the same phrase every turn and
every benchmark scores that as success.

The current objective is narrower and harder to fake:

> When someone tells you something, say something back **about the thing they
> said** -- and when you do not recognise it, react to the feeling and invent
> nothing.

### Measured state

All checkpoints answer "i got a new dog", "i'm a nurse" and "we went to italy"
with the same stock phrase. ElaborationBench scores every one of them at **0.0%
elaboration**: no reply in any corpus this project has built has ever reused the
user's own noun. False elaboration is also **0.0%** -- the restraint half of the
target behaviour already works and is not at risk.

A 1,500-step pilot on the frame curriculum produced the most informative result
so far. It learned the syntax immediately and filled the slot wrong:

```
"i got a new dog"  ->  "what's the puppy called?"
"i'm a nurse"      ->  "do you like being a teacher?"
"we went to italy" ->  "how long were you in brighton?"
```

The frame is correct every time -- right shape, right type, a slot expecting a
noun -- and the noun is a *different member of the same bank*. That is slot
memorisation, the v0.1 failure reappearing one level up: it learned "pet frames
end in a pet word" rather than "copy the word they said". Elaboration on unseen
subjects moved 0% -> 8.3%, so genuine copying has started.

## The harness

The second half of the project asks a separate question:

> How much conversational quality can be extracted from the **unchanged** 6.7M
> model by moving deterministic work outside it?

The base model, tokenizer and weights are never modified. `A_RAW` is asserted by
test to be byte-identical to unmodified inference, so any measured gain is
attributable to a mechanism rather than to incidental changes in prompting.

```
user message
  -> deterministic features -> conversation state -> bounded memory
  -> context selection -> constrained policy -> confidence gate
  -> generation (+ optional steering) -> validation -> reply
```

The dividing line: the neural model does fuzzy language and local judgement; the
harness does exact bookkeeping, state, context selection, repetition control and
validation. A 6.7M model should not spend capacity on operations ordinary
software performs perfectly.

```bash
python scripts/chat_harness.py --checkpoint <ckpt> --mode F_FULL_HARNESS --trace
python scripts/eval_harness.py --checkpoint <ckpt>     # full ablation ladder
python scripts/report_harness_size.py                  # size against the budget
```

**Size budget.** The harness may never exceed the base model's own FP32 weight
size (26,756,096 bytes), or the comparison stops being interesting. It currently
uses **123,153 bytes -- 0.46%** of that ceiling, including all learned assets.
The ceiling is not a target.

### What the harness experiments established

| finding | evidence |
| --- | --- |
| Policy is **linearly encoded** in the hidden state | a 4,883-parameter linear probe reads it at 96.6% held-out, vs 19.2% for the model's own exemplar scoring |
| The model **cannot select** among its own candidates | reranking its top-k by its own likelihood scores 0.09 against greedy's 0.675 |
| Correct policy is worth a lot | hidden-state steering reaches **0.775** against a 0.675 baseline |
| Token-level control **does not work** | forcing the opening token scores 0.208 *even with a correct policy* |
| Repetition is the harness's job | within-conversation repeats 40% -> 0%, zero parameters |
| Elaboration is **not** the harness's job | biasing toward the referent goes from ignoring it to "dog dog dog dog dog?" with no useful setting between |

The last two together are the useful pair: the harness can decide *which* stock
phrase and refuse to repeat one, and it cannot invent syntax the model never
learned. That is what sent the elaboration work back to training.

## Benchmarks, and why there are four

Each was built because the previous one was measured to reward the wrong thing.

| benchmark | asks | why it exists |
| --- | --- | --- |
| Core-Bench | is a single reply appropriate? | reflex reliability |
| StateBench | does a state survive several turns? | Core-Bench was 100% single-turn and blind to continuity |
| VarietyBench | does it repeat itself? | a model emitting one high-scoring phrase every turn wins Core-Bench and is unbearable |
| ElaborationBench | does it reuse what the user said? | all of the above are satisfied by stock phrases |

They are read together on purpose: a model can win VarietyBench by babbling and
Core-Bench by repeating. Only one doing well on all four is holding a
conversation. Every benchmark is frozen by checksum, and a changed checksum
invalidates every earlier score rather than silently shifting the target.

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
