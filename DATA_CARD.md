# Data card and provenance policy

## Scope

smalltalk-ai supports DailyDialog, EmpatheticDialogues, user-supplied JSONL
conversations, curated seed examples, and upstream-generated dialogue. Full
data is intentionally not committed to this GitHub repository.

## Requirements for a dataset release

Every corpus used for a published checkpoint must include a manifest with source
URLs or identifiers, licenses, download date, generator/model revision, planner
revision, critic revision, family IDs, split assignment, token counts measured
with the immutable 4096-token student tokenizer, filters, deduplication
statistics, and checksums.

## Privacy

Do not submit private conversations, credentials, personally identifying
information, or unconsented user data. Generated conversations must not contain
real personal data. Run leakage checks against the frozen benchmark before
training.

## Third-party data

Third-party datasets and upstream model outputs may impose terms beyond this
repository's MIT code license. Users are responsible for reviewing and complying
with those terms before downloading, training, or redistributing artifacts.
