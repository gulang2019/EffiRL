#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable at ${PYTHON_BIN}" >&2
  echo "Create the local virtualenv first." >&2
  exit 1
fi

TRAIN_PARQUET="${TRAIN_PARQUET:-}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

if [[ -z "${TRAIN_PARQUET}" ]]; then
  echo "Set TRAIN_PARQUET to a selected parquet file." >&2
  exit 1
fi
if [[ -z "${RESUME_FROM_PATH}" ]]; then
  echo "Set RESUME_FROM_PATH to a global_step_* checkpoint directory." >&2
  exit 1
fi
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
  echo "Missing selected train parquet: ${TRAIN_PARQUET}" >&2
  exit 1
fi
if [[ ! -d "${RESUME_FROM_PATH}" ]]; then
  echo "Missing resume checkpoint directory: ${RESUME_FROM_PATH}" >&2
  exit 1
fi

GSM8K_TEST="${GSM8K_TEST:-${ROOT_DIR}/data/verl/gsm8k/test.parquet}"
MATH_TEST="${MATH_TEST:-${ROOT_DIR}/data/verl/math/test.parquet}"
VAL_SUBSET_SIZE="${VAL_SUBSET_SIZE:-500}"
VAL_SUBSET_FILE="${VAL_SUBSET_FILE:-${ROOT_DIR}/data/verl/validation_${VAL_SUBSET_SIZE}.parquet}"

missing=0
for path in "${GSM8K_TEST}" "${MATH_TEST}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing dataset file: ${path}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Prepare the math datasets first with: bash scripts/prepare_verl_math_data.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export EFFIRL_PATCH_RESOURCE_TRACKER="${EFFIRL_PATCH_RESOURCE_TRACKER:-1}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-${PPO_MINI_BATCH_SIZE}}"
ROLLOUT_LOGPROB_MB_PER_GPU="${ROLLOUT_LOGPROB_MB_PER_GPU:-${PPO_MINI_BATCH_SIZE}}"
REF_LOGPROB_MB_PER_GPU="${REF_LOGPROB_MB_PER_GPU:-${PPO_MINI_BATCH_SIZE}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
ROLLOUT_N="${ROLLOUT_N:-5}"
ACTOR_LR="${ACTOR_LR:-3e-5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
KL_CTRL_COEF="${KL_CTRL_COEF:-0.001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-1}"
MODEL_USE_SHM="${MODEL_USE_SHM:-true}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
ROLLOUT_ENABLE_SLEEP_MODE="${ROLLOUT_ENABLE_SLEEP_MODE:-true}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-true}"
ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-true}"
EXTRA_TRAINING_STEPS="${EXTRA_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-25}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-1.5b-grpo-gradient-selector-continuation}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/${EXPERIMENT_NAME}}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard_log}"

resume_basename="$(basename "${RESUME_FROM_PATH}")"
if [[ "${resume_basename}" != global_step_* ]]; then
  echo "RESUME_FROM_PATH must end with global_step_*: ${RESUME_FROM_PATH}" >&2
  exit 1
fi
START_STEP="${resume_basename#global_step_}"
if ! [[ "${START_STEP}" =~ ^[0-9]+$ ]]; then
  echo "Failed to parse resume step from ${RESUME_FROM_PATH}" >&2
  exit 1
fi
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-$((START_STEP + EXTRA_TRAINING_STEPS))}"

TRAIN_FILES="['${TRAIN_PARQUET}']"
if [[ "${VAL_SUBSET_SIZE}" -gt 0 ]]; then
  if [[ ! -f "${VAL_SUBSET_FILE}" ]]; then
    echo "Missing validation subset file: ${VAL_SUBSET_FILE}" >&2
    echo "Create it with: ${PYTHON_BIN} scripts/make_val_subset.py --output ${VAL_SUBSET_FILE} --size ${VAL_SUBSET_SIZE} ${GSM8K_TEST} ${MATH_TEST}" >&2
    exit 1
  fi
  VAL_FILES="['${VAL_SUBSET_FILE}']"
else
  VAL_FILES="['${GSM8K_TEST}', '${MATH_TEST}']"
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.use_legacy_worker_impl=disable \
  trainer.val_before_train=False \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_shm="${MODEL_USE_SHM}" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  +actor_rollout_ref.rollout.enable_sleep_mode="${ROLLOUT_ENABLE_SLEEP_MODE}" \
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon="${ROLLOUT_LAYERED_SUMMON}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  algorithm.kl_ctrl.kl_coef="${KL_CTRL_COEF}" \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=EffiRL \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${RUN_DIR}" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  "$@"
