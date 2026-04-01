#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable at ${PYTHON_BIN}" >&2
  echo "Create the local virtualenv first." >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_dapo_data.py" "$@"
