# v0.2 experimental contract

`smalltalk-7m` remains the deployable control: 6,689,024 parameters and the
existing 4096-token BPE. Nothing under `configs/model/smalltalk-7m.yaml` may be
changed for a v0.2 result.

## Correctness gates

- Cached decoding keeps absolute RoPE positions after sliding-cache trimming.
- Packed CLM blocks use a document-block causal mask; neighbouring conversations
  cannot condition one another.
- Generated corpus families use append-stable hash partitions, with train/val/test
  disjoint before surface realisation.
- Loss and perplexity are token-weighted, never an average of padded batch means.
- A frozen benchmark is run only at declared milestones. Iterative selection uses
  a separately versioned development set; it must never be fed to the generator.

## Native-tokenizer teacher bridge

Qwen (or another upstream writer) supplies text only. A native `teacher-50m-native`
checkpoint learns that corpus under the exact student tokenizer, then transfers
its online token distribution through CE + KL. Rejection-selected candidates are
called **rejection-selected SFT**, not logit distillation.

## Architecture tournament boundary

Potential alternate allocations live on an experiment branch and require their
own tokenizer/data manifests. They are not comparable to the frozen 7M result
until matched-corpus runs complete, and none may replace `smalltalk-7m` silently.
The initially requested v0.2 work is data/training improvement on the frozen
control, not an architecture claim.

## Promotion rule

Promote a candidate only when it improves held-out development behaviour without
regressing semantic coherence, epistemic discrimination, or diversity. Validation
loss is diagnostic, not sufficient evidence. Every promotion records commit,
manifest checksum, initialization, token count, loss masking, seed, and raw
generations in the experiment ledger.
