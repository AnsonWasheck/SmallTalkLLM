# Design choices and their justification

Where several implementations were reasonable, we took the simplest experimentally
defensible one. Each choice below states the alternative and why we rejected it, so
any single assumption can be flipped and re-measured.

## Architecture

| Choice | Alternative | Why |
|---|---|---|
| Tied input/output embeddings | Untied | At `hidden=256, vocab=4096` an untied head costs 1.05M params — **27% of the 4M model** — for a decoder whose output distribution is over the same conversational tokens. Untying would confound the scaling curve with a vocabulary-capacity change. |
| GQA with 1–2 KV heads | Full MHA | Halves/quarters KV cache and attention params at no measurable small-talk cost. Attention is a small share of these models anyway; the MLP dominates. |
| No dropout | Dropout 0.1 | At ≤30M params on a narrow domain we are **capacity-limited, not overfitting-limited**. Dropout would slow the fit and add a hyperparameter that interacts with model size, muddying the cliff. Revisit if val loss diverges from train loss. |
| No biases anywhere | Biases on linears | Standard Llama practice; negligible capacity, one fewer thing to differ across sizes. |
| Pre-norm RMSNorm | LayerNorm / post-norm | RMSNorm is cheaper and Llama-compatible so checkpoints export cleanly. |
| Scaled init on `o_proj`/`down_proj` | Uniform init | Keeps residual variance stable as depth grows from 4 → 10 layers. Depth is our main scaling axis, so depth-dependent init instability would masquerade as a capability cliff. |
| Attention **not** reset at packed document boundaries | Per-document block masks | Documented simplification. `<|eos|>` between dialogues is a strong enough signal at 1024 tokens, and it keeps attention a single fast SDPA call. Flagged as a known confound; flip it if stage-1 loss plateaus. |
| Per-layer RoPE offset from that layer's own cache | Global cache length | **This was a real bug we found and fixed.** Layer 0 writes its cache before deeper layers run, so a global `cache.length` gave layers 1..N a position offset inflated by the current chunk. Symptom: KV-cached logits diverged from the full forward by ~2e-3 (correct to 4e-7 after the fix). Covered by `test_rope_offset_is_per_layer`. |

## Tokenizer

- **Byte-level BPE, 4096/6144 only.** A 32k vocab at `hidden=256` costs 8.4M
  embedding params — larger than the entire 8M model. Vocabulary is the single most
  expensive way to spend parameters at this scale, and our domain is intentionally
  narrow English small talk. RQ6 measures the 4096-vs-6144 effect directly.
- **No UNK.** Byte-level fallback means emoji, typos and slang always round-trip.
- **Curated conversational atoms** (`ALWAYS_KEEP`) are repeated in the training
  corpus so BPE reliably merges `don't`, `i'm`, ` lol`, ` yeah`, ` tired`. A 4-layer
  model should not spend depth re-assembling `don't` from three pieces. Ablate with
  `train_tokenizer.py --no-curated-atoms`.
- **Vocab is padded to the requested size** with unused `<|reserved_N|>` rows.
  Vocab size is an *architectural budget*: if a small corpus under-filled it, the
  parameter count would silently drop and the scaling comparison would stop being
  matched. Reserved rows cost params but are never emitted — exactly the cost RQ6
  is asking about.
- **Turn-structural special tokens only** (`<|system|> <|user|> <|assistant|>
  <|endofturn|>`), no instruction-style tokens. We are not training an assistant.

## Data

- **The filters are the scientific instrument.** `FilterConfig` defines what
  "conversationally focused" means (RQ5's independent variable);
  `FilterConfig.permissive()` is the matched generic-LM control. Every drop is
  counted in `corpus_report.json` so the corpus is auditable.
- **Code/math detection is regex-based, not substring-based.** A real bug found in
  testing: the substring `class ` deleted *"class was kinda fun today"* — school
  small talk is an explicit target topic. Naive keyword filters silently destroy
  on-domain data; see `test_filters_drop_off_domain_content`.
- **Left truncation.** Keeping the most recent turns preserves the generation
  target and its local context, which is what we actually evaluate.
- **Same-role runs are merged, not split.** Preserves genuine dialogue boundaries
  and prevents the model learning to emit two assistant turns in a row.
- **Dedup is exact + cheap near-dup** (8-gram shingle containment). Full MinHash
  was rejected as unnecessary complexity at this corpus scale.
- **Weighting is integer-only** (repeat a document ≤2×) rather than per-token loss
  scaling, so the effective token budget stays legible across sizes.

## Training

- **Three stages, one optimiser path.** Stage differences live entirely in the
  dataset and the loss mask. Identical optimisation across the scaling study is
  what makes the comparison "matched".
- **Stage 1 trains on both speakers.** The model must learn what a *user* sounds
  like to predict what comes next; assistant-only loss from scratch wastes most of
  the corpus. Stage 2 then specialises via masking.
- **The assistant's `<|endofturn|>` is supervised** but its role header is not.
  Learning to stop is part of the target behaviour; predicting its own cue is not.
- **No weight decay on 1-D params** (norms), standard practice.
- **bf16 autocast on CUDA/ROCm, fp32 elsewhere.** fp16 was rejected: these models
  are small enough that fp32 on CPU/MPS is fine, and fp16 adds a loss-scaler
  failure mode for no throughput gain at this size.

## Evaluation

- **Whole conversations, not isolated responses.** Conversational collapse is a
  trajectory property; per-response scoring structurally cannot see loops.
- **"Obviously broken" is an explicit rule set** (`BrokenThresholds` +
  `broken_reasons`), not a vibe. Every size is judged by the identical rule, which
  is the only way the cliff location means anything.
- **Automatic metrics are a screening proxy.** The headline claim (RQ8) requires
  human pairwise votes or an LLM judge. `clean_10turn_rate` is what we optimise
  against cheaply; it is not the claim.
- **The judge never runs at inference.** `judge.py` only emits and ingests JSON, so
  a human, hosted LLM or local model are interchangeable and the deployed artifact
  stays the micro model alone.
- **The heuristic candidate scorer is labelled a proxy.** It encodes our stylistic
  priors so rejection sampling can run without an API; swap in `PrecomputedScorer`
  or `CallbackScorer` for numbers that go in the report.
- **Zero-parameter conversation state is off by default.** Otherwise we would be
  measuring the application layer, not the model. RQ-relevant runs must report both.
