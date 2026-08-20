# Naming and release scheme

The repository records an *experiment log*, not just a codebase. The scheme below
exists so that any number quoted in a report can be traced back to the exact
commit, corpus and benchmark checksum that produced it.

## Branches

| branch | meaning |
| --- | --- |
| `main` | published, reproducible states only |
| `develop` | working branch; autonomous-loop progress lands here |

## Commits

Conventional Commits with the study version as the scope:

```
<type>(<scope>): <subject>
```

`type` is one of:

| type | use |
| --- | --- |
| `feat` | new capability, curriculum, or instrument |
| `fix` | corrected defect (state what was measured wrong) |
| `exp` | an experimental result: a loop round worth recording |
| `bench` | benchmark definition or freeze change (always call out invalidation) |
| `docs` | reports, design notes |
| `chore` | infrastructure, tooling, housekeeping |

`scope` is the study phase: `core-v0.2`, `core-v0.3`, `teacher`, `infra`.

### Result commits

A round that improves the champion is recorded with its measurement in the
subject line, so `git log --oneline` reads as an experiment history:

```
exp(core-v0.2): r014 core-bench 0.831 (+0.042) tier1 0.98 -- up-weight tired/goodbye
```

The body must state: benchmark checksum, corpus counts, what changed, and what
did *not* improve. A commit that reports only the wins is not a result.

## Tags

Milestones are annotated tags carrying the frozen benchmark checksum:

```
core-v0.2.1-p1      # phase 1 gates passed
core-v0.2.1-p2      # phase 2 gates passed
```

Format: `<bench-version>-p<phase>`. The tag message records per-tier scores
against their targets and the checksum the scores were measured under.

## Benchmark versions

`CORE_VERSION` in `smalltalk/core/intents.py` moves whenever the intent set
changes, and the checksum changes with it. **A checksum change invalidates
comparison with every earlier score.** Any commit that moves it must say so in
the subject line, and the ledger note must record it.

`SmallTalkBench-v2` (checksum `a2ce68928e780ce5`) is frozen permanently and is
never edited.

## What is not committed

Generated corpora (`data/core/`), checkpoints and run artifacts (`artifacts/`)
stay local -- they are reproducible from the committed code plus the recorded
seed and config. The ledger and reports are the durable record.
