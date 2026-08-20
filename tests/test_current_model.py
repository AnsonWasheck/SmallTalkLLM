"""The repository must state unambiguously which checkpoint is primary.

A pointer that drifts from the archive refs or the frozen benchmark is worse than
no pointer: it makes a stale number look authoritative.
"""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "CURRENT_MODEL.json"


@pytest.fixture(scope="module")
def pointer():
    return json.loads(POINTER.read_text())


def test_pointer_exists_and_names_a_current_model(pointer):
    assert pointer["current"]["label"]
    assert pointer["current"]["checkpoint"]


def test_current_model_is_the_frozen_parameter_count(pointer):
    assert pointer["current"]["params"] == 6_689_024
    assert pointer["current"]["tokenizer_vocab"] == 4096


def test_benchmark_checksum_matches_the_frozen_manifest(pointer):
    frozen = json.loads((ROOT / "benchmarks" / "core_bench_frozen.json").read_text())
    assert pointer["current"]["benchmark"]["checksum"] == frozen["checksum"], (
        "CURRENT_MODEL.json quotes a score measured under a different benchmark "
        "than the one now frozen; the number is not comparable"
    )


def test_superseded_models_are_archived_not_deleted(pointer):
    refs = subprocess.run(["git", "branch", "-a"], cwd=ROOT,
                          capture_output=True, text=True).stdout
    for entry in pointer["history"]:
        assert entry["archive_ref"] in refs, (
            f"{entry['label']} is listed as history but {entry['archive_ref']} "
            "does not exist; superseded models must stay recoverable"
        )
