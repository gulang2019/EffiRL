#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable at ${PYTHON_BIN}" >&2
  echo "Create the local virtualenv first." >&2
  exit 1
fi

DAPO_TRAIN="${DAPO_TRAIN:-${ROOT_DIR}/data/dapo/train.parquet}"
AIME24_TEST="${AIME24_TEST:-${ROOT_DIR}/data/dapo/aime_2024.parquet}"

missing=0
for path in "${DAPO_TRAIN}" "${AIME24_TEST}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing dataset file: ${path}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Prepare the DAPO datasets first with: bash scripts/prepare_dapo_data.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-30}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-${PPO_MINI_BATCH_SIZE}}"
ROLLOUT_LOGPROB_MB_PER_GPU="${ROLLOUT_LOGPROB_MB_PER_GPU:-${PPO_MINI_BATCH_SIZE}}"
REF_LOGPROB_MB_PER_GPU="${REF_LOGPROB_MB_PER_GPU:-${PPO_MINI_BATCH_SIZE}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
ROLLOUT_N="${ROLLOUT_N:-5}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-10}"
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-1.5b-dapo-single-gpu-vllm}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/${EXPERIMENT_NAME}}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  +algorithm.filter_groups.enable=True \
  +algorithm.filter_groups.metric=acc \
  +algorithm.filter_groups.max_num_gen_batches="${MAX_NUM_GEN_BATCHES}" \
  trainer.use_legacy_worker_impl=disable \
  trainer.val_before_train=False \
  data.train_files="${DAPO_TRAIN}" \
  data.val_files="${AIME24_TEST}" \
  data.prompt_key=prompt \
  data.dataloader_num_workers=0 \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  +data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_shm=True \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
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
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  reward.reward_manager.name=dapo \
  +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
  +reward.reward_kwargs.overlong_buffer_cfg.len=128 \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  +reward.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=EffiRL \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${RUN_DIR}" \
  trainer.resume_mode=auto \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  "$@"
