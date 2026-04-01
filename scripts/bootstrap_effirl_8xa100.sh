#!/usr/bin/env bash
set -euo pipefail

# End-to-end bootstrap for a fresh machine:
# 1) clone EffiRL (+ submodules)
# 2) create .venv
# 3) install runtime deps used by the GRPO + periodic-selector pipelines
# 4) run smoke checks
#
# Usage:
#   bash scripts/bootstrap_effirl_8xa100.sh /path/to/workspace
#
# Optional env overrides:
#   REPO_URL=git@github.com:gulang2019/EffiRL.git
#   REPO_BRANCH=main
#   PYTHON_BIN=python3.10
#   VENV_DIR=.venv
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#   INSTALL_FLASH_ATTN=1

TARGET_PARENT="${1:-$PWD}"
REPO_URL="${REPO_URL:-git@github.com:gulang2019/EffiRL.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

mkdir -p "${TARGET_PARENT}"
cd "${TARGET_PARENT}"

if [[ ! -d EffiRL/.git ]]; then
  git clone "${REPO_URL}" EffiRL
fi

cd EffiRL
git fetch origin
git checkout "${REPO_BRANCH}"
git pull --ff-only origin "${REPO_BRANCH}"
git submodule update --init --recursive

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

PY="${PWD}/${VENV_DIR}/bin/python"
PIP="${PWD}/${VENV_DIR}/bin/pip"

"${PY}" -m pip install --upgrade pip wheel "setuptools<81"

# Core training / profiling stack used by this repo.
"${PIP}" install \
  "torch==2.9.0" --index-url "${TORCH_INDEX_URL}"
"${PIP}" install \
  "ray==2.54.1" \
  "tensordict==0.11.0" \
  "transformers==4.57.6" \
  "vllm==0.12.0" \
  "math-verify==0.9.0" \
  "peft>=0.12.0" \
  "omegaconf>=2.3.0" \
  "accelerate>=0.34.0" \
  "datasets>=2.20.0" \
  "pyarrow>=17.0.0" \
  "pandas>=2.2.0" \
  "numpy>=1.26.0" \
  "scipy>=1.11.0" \
  "matplotlib>=3.8.0" \
  "pyyaml" \
  "tensorboard==2.20.0"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  # Optional: needed when verl paths require flash_attn. On some hosts this may need CUDA toolchain setup.
  "${PIP}" install flash-attn --no-build-isolation || {
    echo "flash-attn installation failed. You can still run with USE_REMOVE_PADDING=false." >&2
    exit 1
  }
fi

# Use the vendored verl code in this repo.
"${PIP}" install --no-deps -e thirdparty/verl

echo
echo "Running smoke checks..."
"${PY}" - <<'PY'
import torch
import ray
import verl
import vllm
import transformers
import pyarrow
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("ray:", ray.__version__)
print("verl import: ok")
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("pyarrow:", pyarrow.__version__)
PY

"${PY}" -m verl.trainer.main_ppo --help >/dev/null

echo
echo "Bootstrap complete."
echo "Repo: ${PWD}"
echo "Python: ${PY}"
echo
echo "Next:"
echo "  1) bash scripts/prepare_verl_math_data.sh     # if datasets not prepared"
echo "  2) bash scripts/run_periodic_gradient_selector_8xa100.sh"
