# smalltalk-ai — research report

Status: **protocol fixed, infrastructure verified, full-scale training not yet run.**

This document is the pre-registered protocol. Numbers are filled in by
`scripts/scaling_report.py`, which writes `docs/RESULTS.md` from measured evidence
and refuses to guess where evidence is missing. Thresholds below are declared
*before* training so the cliff location cannot be chosen after the fact.

## 1. Question

How small can a generative language model become while still sustaining natural,
human-like one-on-one casual conversation, when broad factual knowledge, reasoning
and instruction following have been deliberately removed from the training
objective?

## 2. Independent variable

Parameter count, via depth and width:
3.87M, 5.28M, 6.69M, 8.10M, 14.95M, and 25.70M as an upper baseline. Data,
tokenizer, optimiser, schedule, seed and evaluation are held fixed across sizes
(the 15M/26M configs additionally change vocab to 6144, which is why RQ6 is
measured separately as a controlled ablation rather than read off the main curve).

## 3. Primary metric

`clean_10turn_rate`: the fraction of held-out SmallTalkBench scenarios completed
for 10 turns with **zero** obviously broken, repetitive or contextually nonsensical
assistant replies.

"Obviously broken" is defined operationally, identically at every size, as any of:

| reason | rule |
|---|---|
| `empty` | no content, or only `...` |
| `too_long` | > 60 words |
| `intra_repetition` | > 34% duplicate 3-grams within one reply |
| `context_copy` | ≥ 85% of reply words lifted from the user's turn (≥ 4 words) |
| `ungrammatical` | adjacent word repetition, consonant soup, ≥ 5-char runs, or type/token < 0.45 |
| `ai_assistant_style` | matches the assistant-verbosity marker list |
| `repeats_previous_turn` | verbatim reuse of an earlier reply |
| `probe_*` | a scenario probe failed (e.g. did not recall the dog's name; fabricated a fact) |

Plus a conversation-level `loop` check: two consecutive near-identical replies
(Jaccard ≥ 0.8) or any reply repeated three times.

Automatic metrics are the **screening proxy**. The headline claim (RQ8) requires
human or LLM judgement, which is a deliberate separation.

## 4. Emergence thresholds (pre-declared)

| capability | metric | threshold |
|---|---|---|
| grammatical English | `grammatical_rate` | ≥ 0.90 |
| context-sensitive generation | `probe_pass_rate` | ≥ 0.60 |
| reliable 5-turn conversation | `clean_5turn_rate` | ≥ 0.80 |
| reliable 10-turn conversation | `clean_10turn_rate` | ≥ 0.80 |
| judged conversational | `human_likeness` (1–5) | ≥ 3.5 |

A capability is credited to the **smallest** model that meets the threshold *and*
is not undercut by any larger model, so a single lucky sample cannot claim a cliff.

## 5. Secondary metrics

Validation loss/perplexity on assistant tokens only (the quantity we actually
care about), all-token validation loss, loop rate, broken-turn rate, mean reply
length, fraction of replies in the 3–25 word band, distinct-1/2, type-token ratio,
self-overlap between replies, question ratio, context-copy ratio, and the eight
1–5 judge criteria.

## 6. Protocol

1. `scripts/smoke_test.py` must pass (10/10) before any expensive run.
2. Build one corpus. Freeze it. Record `corpus_report.json`.
3. Train tokenizers at 4096 and 6144, padded to exactly that size so vocabulary
   remains a fixed architectural budget.
4. For each config: stage 1 (CLM) → stage 2 (assistant-only SFT) → optional
   stage 3 (distillation on rejection-sampled teacher winners).
5. Sweep decoding per checkpoint (`evaluate.py --sweep`) and report each model at
   *its own* best decoding settings — otherwise the curve measures our choice of
   temperature, not capability.
6. Evaluate all checkpoints on identical scenarios and seeds.
7. Emit blind pairwise items and judge packets; collect human/LLM scores.
8. `scripts/scaling_report.py` generates the curves and answers RQ1–8.

## 7. Ablations (matched at equal parameter count)

| RQ | comparison |
|---|---|
| 5 | conversational filtering vs `FilterConfig.permissive()` generic filtering |
| 6 | vocab 4096 vs 6144 at fixed hidden/depth; plus curated-atoms on/off |
| 7 | stage-2 SFT only vs stage-2 + stage-3 distillation |
| — | assistant-only masking vs all-token loss in stage 2 |
| — | zero-parameter conversation state on/off (must be reported separately) |

## 8. Findings so far

Infrastructure verified; the scaling study itself has not been run. What is
established:

- **Parameter counts are exact.** All six configs match their targets to the
  parameter (3,868,928 / 5,278,976 / 6,689,024 / 8,099,072 / 14,948,736 /
  25,698,816), confirmed both by formula and by module instantiation.
- **Two real bugs were found and fixed by the pre-flight gate**, either of which
  would have silently corrupted results:
  1. *Per-layer RoPE offset.* A global `cache.length` gave layers 1..N a position
     offset inflated by layer 0's just-written cache chunk. KV-cached logits
     diverged from the full forward by ~2e-3; after the fix, ~4e-7. This would
     have made every generated conversation subtly different from the trained
     distribution — invisible in training loss, fatal at evaluation.
  2. *Substring code filter.* The marker `class ` deleted *"class was kinda fun
     today"*. School small talk is an explicit target topic, so the filter was
     silently removing on-domain data. Now regex/context-based.
- **A plumbing-scale run completed** for `smalltalk-4m` (1.5k template
  conversations, 400 CLM + 300 SFT steps, CPU): assistant-token val ppl 2.90,
  `grammatical_rate` 0.986, `clean_5turn_rate` 0.571, `clean_10turn_rate` 0.143,
  `loop_rate` 0.643. This is **not** a result about 4M capability — the template
  corpus has only ~40 distinct reply phrasings, so the model can be locally
  fluent while looping globally. It is evidence the harness measures what it
  claims: it credits grammaticality, and it catches the loops.

## 9. Known confounds and limitations

- The offline template generator exists to make the repo runnable and hermetic. It
  is low-diversity and must not be used for headline numbers; real conclusions
  need DailyDialog + EmpatheticDialogues + teacher data.
- The 15M/26M configs change vocab as well as width, so RQ6 must come from the
  controlled ablation, not the main curve.
- Attention is not reset at packed document boundaries in stage 1 (documented
  simplification).
- The heuristic candidate scorer encodes our stylistic priors. It is a proxy for
  scaling rejection sampling cheaply; reported distillation gains should use
  `PrecomputedScorer` with human or LLM scores.
- `grammatical_rate` uses heuristics, not a grammar model. It reliably catches the
  failure modes tiny models exhibit (word salad, immediate repetition) and will
  miss subtler agreement errors.
- Automatic metrics cannot distinguish "safe and bland" from "genuinely engaging".
  This is precisely why RQ8 is defined on human judgement.

## 10. Answers

To be generated into `docs/RESULTS.md`:

```bash
python scripts/evaluate.py --checkpoint artifacts/runs/sft-*/best --out artifacts/eval
python scripts/scaling_report.py --eval-dir artifacts/eval --out docs/RESULTS.md
```

1. At what parameter count does grammatical English emerge? — *pending*
2. At what parameter count does context-sensitive response generation emerge? — *pending*
3. At what parameter count does five-turn conversation become reliable? — *pending*
4. At what parameter count does ten-turn conversation become reliable? — *pending*
5. How much does conversationally focused training outperform generic LM training at equal parameter count? — *pending ablation*
6. How much does reducing vocabulary size help? — *pending ablation*
7. How much does teacher-generated conversational distillation help? — *pending ablation*
8. What is the smallest model that humans consistently judge as conversational rather than obviously broken? — *pending human evaluation*
