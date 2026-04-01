#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable at ${PYTHON_BIN}" >&2
  echo "Create the local virtualenv first." >&2
  exit 1
fi

PROFILE_CSV="${PROFILE_CSV:-}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-200}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
POOL_DIR="${POOL_DIR:-${ROOT_DIR}/outputs/selector_pools}"
SEED="${SEED:-7}"

if [[ -z "${PROFILE_CSV}" ]]; then
  echo "Set PROFILE_CSV to a ground_truth_profile.csv path." >&2
  exit 1
fi
if [[ ! -f "${PROFILE_CSV}" ]]; then
  echo "Missing profile CSV: ${PROFILE_CSV}" >&2
  exit 1
fi

SOURCE_PARQUETS=("$@")
if [[ "${#SOURCE_PARQUETS[@]}" -eq 0 ]]; then
  echo "Pass one or more source parquet files after the env vars." >&2
  exit 1
fi

mkdir -p "${POOL_DIR}"

build_one() {
  local name="$1"
  local metric="$2"
  local selector="$3"

  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_profile_selector_dataset.py" \
    --profile-csv "${PROFILE_CSV}" \
    --source-parquets "${SOURCE_PARQUETS[@]}" \
    --checkpoint-step "${CHECKPOINT_STEP}" \
    --metric "${metric}" \
    --selector "${selector}" \
    --keep-ratio "${KEEP_RATIO}" \
    --seed "${SEED}" \
    --output-parquet "${POOL_DIR}/${name}.parquet" \
    --output-selection-csv "${POOL_DIR}/${name}_selection.csv"
}

build_one gradient_top "gradient_statistical_efficiency" "top"
build_one gradient_bottom "gradient_statistical_efficiency" "bottom"
build_one gradient_random "gradient_statistical_efficiency" "random"
build_one dapo_keep_top "dapo_keep_efficiency" "top"

echo "Built selector pools in ${POOL_DIR}"
