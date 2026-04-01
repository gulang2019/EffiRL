#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable at ${PYTHON_BIN}" >&2
  echo "Create the local virtualenv first." >&2
  exit 1
fi

GSM8K_DIR="${GSM8K_DIR:-${ROOT_DIR}/data/verl/gsm8k}"
MATH_DIR="${MATH_DIR:-${ROOT_DIR}/data/verl/math}"

mkdir -p "${GSM8K_DIR}" "${MATH_DIR}"

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

"${PYTHON_BIN}" "${ROOT_DIR}/thirdparty/verl/examples/data_preprocess/gsm8k.py" \
  --local_save_dir "${GSM8K_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/thirdparty/verl/examples/data_preprocess/math_dataset.py" \
  --local_save_dir "${MATH_DIR}"

echo "Prepared:"
echo "  ${GSM8K_DIR}/train.parquet"
echo "  ${GSM8K_DIR}/test.parquet"
echo "  ${MATH_DIR}/train.parquet"
echo "  ${MATH_DIR}/test.parquet"
