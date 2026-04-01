#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "The Hugging Face rollout backend is not runnable in this pinned verl revision; using the supported vLLM launcher instead." >&2
exec "${ROOT_DIR}/scripts/run_verl_math_1p5b_single_gpu_vllm.sh" "$@"
