# Model card: smalltalk-7m

## Summary

`smalltalk-7m` is an experimental 6,689,024-parameter decoder-only language model for short, casual, one-on-one English conversation. It is intentionally narrow and is not a general-purpose assistant.

## Intended use

Research on conversational fluency at very small parameter counts; lightweight,
low-stakes local chat experiments; and controlled evaluation of context,
emotion, repair, and nearby memory.

## Out of scope

Factual QA, coding, mathematics, tool use, autonomous actions, long-form
writing, professional advice, medical/legal/financial guidance, crisis support,
or decisions affecting people.

## Architecture

8 layers, hidden size 256, 4 query heads, 1 KV head, dimension 64, SwiGLU size
704, RMSNorm, RoPE, tied embeddings, bias-free linears, 4096-token tokenizer,
and 1024-token maximum context. Exact trainable parameter count:
**6,689,024**.

## Training

The research pipeline uses conversational datasets, assistant-response SFT,
optional teacher distillation, and targeted failure mining. Dataset versions,
licenses, filters, and evaluation results must be read from the corresponding
manifest and report. Do not infer that a checkpoint contains only public or
permissively licensed data without checking its provenance.

## Limitations and risks

The model can hallucinate, lose track of entities, repeat itself, overfit
templates, respond with the wrong emotional valence, and sound confident when
it should be uncertain. Conversation history may contain sensitive information;
deploy with appropriate retention and privacy controls. Its small size does not
make it safe by default.

## Evaluation

Use the frozen SmallTalkBench-v2 evaluator and report naked-model and optional
application-state tracks separately. A good grammar score is not evidence of
memory, naturalness, or epistemic reliability.
