#!/usr/bin/env python3
"""Check a corpus file for overlap with every frozen SmallTalkBench benchmark.

python scripts/check_leakage.py data/processed/train.jsonl
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from smalltalk.data.schema import load_conversations
from smalltalk.eval.leakage import check_conversations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    args = ap.parse_args()
    convs = load_conversations(args.path)
    report = check_conversations(convs)
    print(report.summary())
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
