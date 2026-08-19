# Contributing to smalltalk-ai

smalltalk-ai is a research repository first. Contributions should make an
assumption easier to inspect, reproduce, or falsify.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
make smoke
```

## Branches and releases

- `main` is the stable, reproducible public line.
- `develop` is the integration line for the next minor release.
- `exp/<short-name>` is for one controlled research experiment.
- `release/<version>` is a short-lived stabilization branch.
- Tags `vMAJOR.MINOR.PATCH` identify reproducible code releases.

Do not rewrite `main` history. Use pull requests, keep commits focused, and
never commit credentials, raw/private data, local environments, checkpoints,
or generated artifacts. Large model files belong in a release asset or a
future Hugging Face repository with their own card and license.

## Research changes

Every model result should record the config, exact parameter count, tokenizer,
dataset manifest/checksum, seed, optimizer, token budget, checkpoint, and raw
evaluation output. Do not edit frozen benchmark scenarios to improve a score.
If a validation or leakage issue is discovered, report it explicitly and
invalidate affected comparisons.
