#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  elif command -v python >/dev/null 2>&1; then
    python -m venv .venv
  else
    echo "Error: python3 (or python) is required to create .venv" >&2
    exit 1
  fi

  ".venv/bin/python" -m pip install --upgrade pip
  ".venv/bin/pip" install -r requirements.txt
fi

exec ".venv/bin/python" -m pytest "$@"

