#!/usr/bin/env bash
# Unified validation checks: linting, formatting, type checking, & unit tests.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

printf '\n==> Ruff lint\n'
uv run --locked ruff check faceledger tests scripts

printf '\n==> Ruff format check\n'
uv run --locked ruff format --check faceledger tests scripts

printf '\n==> Mypy type check\n'
uv run --locked mypy faceledger tests scripts

printf '\n==> Unit tests\n'
uv run --locked python -m unittest discover -s tests

printf '\n==> Distribution qualification\n'
uv run --locked python scripts/check_distribution.py

printf '\nAll validation checks passed.\n'
