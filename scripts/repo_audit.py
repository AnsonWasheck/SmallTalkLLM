"""Fail-fast checks for a clean public source repository."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {".env", "id_rsa", "credentials.json"}
FORBIDDEN_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
FORBIDDEN_DIRS = {".venv", ".venv-rocm", "models", "artifacts", ".git"}


def visible_files() -> list[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
    return [x for x in out.splitlines() if x]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-mb", type=float, default=10.0)
    args = parser.parse_args()
    failures: list[str] = []
    files = visible_files()
    for name in files:
        p = ROOT / name
        parts = set(p.parts)
        if parts & FORBIDDEN_DIRS:
            failures.append(f"tracked runtime directory: {name}")
        if p.name in FORBIDDEN_NAMES or p.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"tracked private/binary artifact: {name}")
        if p.is_file() and p.stat().st_size > args.max_file_mb * 1024 * 1024:
            failures.append(f"tracked file exceeds {args.max_file_mb:g} MB: {name}")
    required = ["README.md", "LICENSE", "pyproject.toml", "CONTRIBUTING.md", ".github/workflows/ci.yml"]
    failures.extend(f"missing required public file: {x}" for x in required if not (ROOT / x).exists())
    if failures:
        print("REPOSITORY AUDIT FAILED")
        print("\n".join(f"- {x}" for x in failures))
        return 1
    print(f"repository audit passed: {len(files)} visible files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
