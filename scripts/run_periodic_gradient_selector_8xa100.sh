#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable: ${PYTHON_BIN}" >&2
  exit 1
fi

# One periodic run per GPU (8 total). This uses the existing single-GPU launcher.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"

# =============================
# Core experiment schedule
# =============================
# Refresh cadence for selector updates. Lower = more adaptive, higher profiling overhead.
WINDOW_STEPS="${WINDOW_STEPS:-5}"
# End-to-end GRPO horizon.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
# Selector policy: top / bottom / random by chosen metric.
SELECTOR="${SELECTOR:-top}"
# Fraction kept after each profiling refresh (ignored if KEEP_COUNT is set).
KEEP_RATIO="${KEEP_RATIO:-0.5}"
# Absolute keep size. Leave empty to use KEEP_RATIO.
KEEP_COUNT="${KEEP_COUNT:-}"

# =============================
# Profiling (V1/V2 + rollout)
# =============================
# V1 and V2 are both: TRAIN_BATCH_SIZE * WINDOW_STEPS * PROFILE_WINDOW_MULTIPLIER.
# Higher = lower variance selector estimate, but more profiling cost.
PROFILE_WINDOW_MULTIPLIER="${PROFILE_WINDOW_MULTIPLIER:-4}"
# Number of rollout samples per prompt for gradient estimator.
PROFILE_GROUP_SIZE="${PROFILE_GROUP_SIZE:-5}"
# Prompts profiled concurrently. Main OOM lever for profiling.
PROFILE_ROLLOUT_BATCH_SIZE="${PROFILE_ROLLOUT_BATCH_SIZE:-4}"
PROFILE_MAX_PROMPT_TOKENS="${PROFILE_MAX_PROMPT_TOKENS:-512}"
# Max generated tokens during profiling rollouts. Main OOM/latency lever.
PROFILE_MAX_NEW_TOKENS="${PROFILE_MAX_NEW_TOKENS:-4096}"

# =============================
# Training launcher overrides
# =============================
# Prompts per RL step in launcher.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
# Training generation cap inside verl launcher.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
# GRPO rollout count per prompt during training.
ROLLOUT_N="${ROLLOUT_N:-5}"
# Validation/save cadence for each window run.
TEST_FREQ="${TEST_FREQ:-25}"
SAVE_FREQ="${SAVE_FREQ:-${WINDOW_STEPS}}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-${ROOT_DIR}/runs/periodic_gradient_selector_8xa100}"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-qwen2.5-1.5b-grpo-periodic-gradient-selector-8xa100}"

KEEP_ARGS=(--keep-ratio "${KEEP_RATIO}")
if [[ -n "${KEEP_COUNT}" ]]; then
  KEEP_ARGS=(--keep-count "${KEEP_COUNT}")
fi

mkdir -p "${BASE_RUN_ROOT}"

for idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$idx]}"
  seed="$((7 + idx))"
  run_root="${BASE_RUN_ROOT}/seed_${seed}"
  experiment_name="${BASE_EXPERIMENT_NAME}-seed-${seed}"
  log_path="${run_root}/launcher.log"
  tb_dir="${run_root}/tensorboard_log"

  mkdir -p "${run_root}"

  echo "[launch] gpu=${gpu} seed=${seed} run_root=${run_root}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    # Helps reduce fragmentation-related OOMs on long profiling runs.
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_periodic_gradient_selector.py" \
      --run-root "${run_root}" \
      --experiment-name "${experiment_name}" \
      --window-steps "${WINDOW_STEPS}" \
      --total-training-steps "${TOTAL_TRAINING_STEPS}" \
      --selector "${SELECTOR}" \
      --profile-window-multiplier "${PROFILE_WINDOW_MULTIPLIER}" \
      --profile-seed "${seed}" \
      --profile-group-size "${PROFILE_GROUP_SIZE}" \
      --profile-rollout-batch-size "${PROFILE_ROLLOUT_BATCH_SIZE}" \
      --profile-max-prompt-tokens "${PROFILE_MAX_PROMPT_TOKENS}" \
      --profile-max-new-tokens "${PROFILE_MAX_NEW_TOKENS}" \
      "${KEEP_ARGS[@]}" \
      --launcher-env "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" \
      --launcher-env "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}" \
      --launcher-env "ROLLOUT_N=${ROLLOUT_N}" \
      --launcher-env "TEST_FREQ=${TEST_FREQ}" \
      --launcher-env "SAVE_FREQ=${SAVE_FREQ}" \
      --launcher-env "TRAINER_LOGGER=[\"console\",\"tensorboard\"]" \
      --launcher-env "TENSORBOARD_DIR=${tb_dir}" \
      --launcher-env "PYTORCH_ALLOC_CONF=expandable_segments:True" \
      > "${log_path}" 2>&1 &
done

echo
echo "Launched ${#GPU_ARRAY[@]} runs."
echo "Status check:"
echo "  ps -eo pid,etime,cmd | rg 'run_periodic_gradient_selector.py' | rg -v rg"
echo "Tail one run:"
echo "  tail -f ${BASE_RUN_ROOT}/seed_7/launcher.log"
echo "TensorBoard (example for seed_7):"
echo "  ./.venv/bin/python -m tensorboard.main --logdir ${BASE_RUN_ROOT}/seed_7/tensorboard_log --host 0.0.0.0 --port 6006"
