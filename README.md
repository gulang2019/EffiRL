# EffiRL

## Quickstart (Clean Repro Setup)

### 1) Clone + build a dedicated environment

Use a separate env for this project (recommended).

```bash
git clone git@github.com:gulang2019/EffiRL.git
cd EffiRL
git submodule update --init --recursive

python3 -m venv .venv_effirl
source .venv_effirl/bin/activate
python -m pip install --upgrade pip wheel "setuptools<81"
```

Install core dependencies:

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install \
  ray==2.54.1 \
  tensordict==0.11.0 \
  transformers==4.57.6 \
  vllm==0.12.0 \
  peft>=0.12.0 \
  omegaconf>=2.3.0 \
  accelerate>=0.34.0 \
  datasets>=2.20.0 \
  pyarrow>=17.0.0 \
  pandas>=2.2.0 \
  numpy>=1.26.0 \
  scipy>=1.11.0 \
  matplotlib>=3.8.0 \
  pyyaml \
  tensorboard==2.20.0
pip install --no-deps -e thirdparty/verl
```

Optional (if your host supports it): install `flash_attn`

```bash
pip install flash-attn --no-build-isolation
```

If `flash_attn` is unavailable on your host, run launchers with:

```bash
--launcher-env USE_REMOVE_PADDING=false
```

Alternative: one-command bootstrap:

```bash
bash scripts/bootstrap_effirl_8xa100.sh /path/to/workspace
# optional: INSTALL_FLASH_ATTN=1 bash scripts/bootstrap_effirl_8xa100.sh /path/to/workspace
```

### 2) Prepare data

```bash
bash scripts/prepare_verl_math_data.sh
./.venv_effirl/bin/python scripts/make_val_subset.py \
  --output data/verl/validation_500.parquet \
  --size 500 \
  data/verl/gsm8k/test.parquet \
  data/verl/math/test.parquet
```

### 3) Run policy-grid training (apple-to-apple)

Continuation from an existing checkpoint:

```bash
./.venv_effirl/bin/python scripts/run_policy_grid_8gpu.py \
  --run-root runs/policy_grid_8gpu \
  --mode continuation \
  --start-from-path <path_to_global_step_x> \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --arm-training-steps 100 \
  --window-steps 5 \
  --gradient-update-mode periodic \
  --profile-window-multiplier 4 \
  --gradient-keep-ratios 0.2,0.4,0.6,0.8,1.0 \
  --launcher-env 'TRAINER_LOGGER=["console","tensorboard"]' \
  --launcher-env USE_REMOVE_PADDING=false
```

Run only specific policies / baselines (useful for retrying failed jobs):

```bash
./.venv_effirl/bin/python scripts/run_policy_grid_8gpu.py \
  --run-root runs/policy_grid_retry_random \
  --mode continuation \
  --start-from-path <path_to_global_step_x> \
  --only-policies random \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --policy-gpu random:6 \
  --policy-launch-mode continuation \
  --gradient-update-mode static \
  --launcher-env USE_REMOVE_PADDING=false
```

Flags:

- `--only-policies`: comma-separated policy names to launch.
- `--policy-gpu POLICY:GPU`: pin selected policies to specific GPU IDs.

Scratch start (warmup first):

```bash
./.venv_effirl/bin/python scripts/run_policy_grid_8gpu.py \
  --run-root runs/policy_grid_8gpu \
  --mode scratch \
  --checkpoint-step 15 \
  --gpu-ids 0,1,2,3,4,5,6,7
```

Notes for detailed per-step analysis:

- Policy launcher now saves checkpoints at `window_steps` cadence, so step-wise rollouts can be reconstructed.
- Gradient policies with `--gradient-update-mode periodic` refresh gradient direction each window.

### 4) Track progress + generate report bundle

Live tracking:

```bash
./.venv_effirl/bin/python scripts/track_policy_grid_progress.py \
  --run-root runs/policy_grid_8gpu \
  --watch-seconds 30
```

Generate curves:

```bash
./.venv_effirl/bin/python scripts/plot_policy_grid_curves.py \
  --run-root runs/policy_grid_8gpu \
  --metrics actor_loss,gsm8k_acc,math_acc,response_length_mean \
  --smooth 0.2
```

Generate aggregate deltas:

```bash
./.venv_effirl/bin/python scripts/export_policy_learning_delta.py \
  --run-root runs/policy_grid_8gpu
```

Detailed per-checkpoint rollout snapshots (for qualitative comparisons):

```bash
./.venv_effirl/bin/python scripts/collect_policy_rollout_snapshots.py \
  --run-root runs/policy_grid_8gpu \
  --eval-files data/verl/validation_500.parquet \
  --eval-limit 200 \
  --max-new-tokens 512 \
  --max-checkpoints-per-policy 5
```

One-command report bundle (curves + deltas + selected-data overlap + optional rollouts):

```bash
./.venv_effirl/bin/python scripts/build_policy_grid_report.py \
  --run-root runs/policy_grid_8gpu \
  --collect-rollouts \
  --rollout-eval-files data/verl/validation_500.parquet \
  --rollout-eval-limit 200
```

### 5) Compare newly solved examples between two checkpoints

```bash
./.venv_effirl/bin/python scripts/report_newly_solved_examples.py \
  --start-step-dir <run>/global_step_15 \
  --end-step-dir <run>/global_step_100 \
  --train-files data/verl/gsm8k/train.parquet data/verl/math/train.parquet \
  --val-files data/verl/validation_500.parquet \
  --max-new-tokens 1024 \
  --output-dir runs/analysis/newly_solved_step15_to_100
```

This exports:

- per-example solved transitions (`newly_solved`, `forgotten`)
- start/end rollout text
- split/source summaries

**Background** 

* It is well studied that online data selection matters for training performance, but less discusses the compute efficiency of data selection. 

* In reinforcement learning with verifiable reward (RLVR), due to the straggler problem in the rollout stage, data selection is more important for compute efficiency compared to pre-training or SFT.

## Motivation

*  We observe that compute efficiency (training time / sample) and statistical efficiency (loss drop/sample) is 1. Misaligned and 2. Changes over training. 
[Figure 1] a) Early training, b) Middle training, c) Late training
	X: compute efficiency, Y: statistical efficiency; dots per data sample;

## Idea
Based on that, we propose efficient-aware online data selection to minimize the training time towards a performance target. 

## Challenge & Solution

* Challenge #1: quantify training Goodput (loss drop/s): it is a complex mixture of compute efficiency (data sample / s) and statistical efficiency (loss drop /data sample). The statistical efficiency is well studied with heuristics, but compute efficiency for RL is not addressed. 
* Solution #1: Modeling the data selection problem as a constrained optimization; And model the compute efficiency by performance modeling of each RL stage; 

* Challenge #2: compute the Goodput efficiently; 

* Solution #2: Bucketing the data to reduce the profiling quantity;  Use idle compute (idle rollout/training server when training/rollout is on) to profile these metrics non-stop, and asynchronously; 
## Evaluation:

* Our approach takes less time to train to a good (better) quality model.
E2E: 
Baseline: DAPO (statistical efficiency-only) [1], SkyRL,  Curriculum learning.
	Task: RL on math; RL on code; on-policy distillation; 

* Sensitivity Analysis: Does our data selection create bias in data selection?
Case study: is our data selection algorithm selecting the qualitative right data at the right time? 

## Todos 

* The Motivation experiment 
    - Define statistical efficiency metric;
        - Do a survey on different approaches for selecting data: 
            - how this is done in non-RL training? 
            - how this is done in RL training? 
            - what is the computational cost of each? 
            - what is the performance gain for doing each?
            - curate a paper list for deeper read later; 
    - Define compute efficiency metric;
        - For a data sample, how to quantify its runtime contribution under batched rollout? But for the prifiling, we can assume we know this after-run.
    - Run canonical GRPO training and profile the statistical efficiency and data efficiency.
        - Training framework: verl
        - Training dataset: 
            - math-training: agentica-org/DeepScalerR-Preview-Dataset (40K)
            - coding: PROMERL;
        - Evaluation dataset: 
            - Livecodebench for coding eval; 
            - AIME24 for training evaluation; 
        

Related Work
- Truncated Proximal Policy Optimization


## Statistical Efficiencies

- DAPO: $stt_effi(d) = 1_{acc\not\in\{0,1\}}$
- PODS: $stt_effi(d) = p_{acc} * (1 - p_{acc})$
- Graident: $stt_effi(d) = \Delta_{\theta_t} J(\theta) \cdot \Delta_{\theta_t} j_{data}(\theta)$

Comparison:
- Removing data with poor stt_effi (DAPO, GREATS)
- Removing data with poor comp_effi (Rollpacker, Truncated PPO) 
- Removing data with poor goodput (Ours)


![](outputs/qwen2.5_1.5b_training_dashboard.svg)
Figure. Training Loss, Validation accuracy validation.

![](outputs/qwen2.5_1.5b_generation_length_over_training.svg)
Figure. Generation length and P99 Generation length. The tail rollout is way longer than normal rollout. 

![](outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/analysis/time_breakdown_by_checkpoint.svg)
Figure. Runtime breakdown between rollout and gradient update. The training time is dominated by the rollout stage. 

![](outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/analysis/stat_vs_compute_by_checkpoint.svg)
Figure. Statistical efficiency vs. Compute Efficiency. There is not an unified relationship between `stt_effi` and `compute_effi`. Also the `stt_effi` is not aligned between the `DAPO` and `Gradient` approach.


![](outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/analysis/trajectory_direction_counts.svg)
Figure. The changing of statistical efficiency and compute efficiency across training steps. The statistical efficiency and compute efficiency for each data point is changing.

![](outputs/profiling_gsm8k_efficiency_2026-03-27_v32_len1024/analysis/mini_batch_goodput_vs_removed_data_batch16.svg)
Figure. The comparison between different policies. Removing data with low statistical efficiency or goodput will always benefit the training. However, removing the data with poor statistical efficiency can actually hurt the training. Removing data based on goodput helps more for the `ckpt-50` compared to the `stt_effi` only approach.

Implement three training algorithms: 

 * `DAPO`: upsampling and downsampling until the batch contains enough data with `stt_effi=1` to train; 
 * `PODS`: upsampling and downsampling until the batch contains enough data with `stt_effi=1` to train; 
 * `Partial Rollout`(https://verl.readthedocs.io/en/latest/advance/fully_async.html): Continuous (asynchronous) rollout across training steps. 
 * `EffiAware Downsampling`: Upsampling, then downsampling by constructing a batch that maximimizes the goodput.

Do normal GRPO training, every K steps, compute the `ground_truth` gradient on `V1`.

## 8-GPU Policy Grid Workflow

Use this when you want one policy per GPU (instead of only different seeds), plus gradient-policy grid search.

### 1) Launch policies on 8 GPUs

```bash
./.venv/bin/python scripts/run_policy_grid_8gpu.py \
  --run-root runs/policy_grid_8gpu \
  --start-from-path runs/periodic_gradient_selector_100steps_w5_mult4_b2/window_002_resume_10_to_15/global_step_15 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --arm-training-steps 100 \
  --window-steps 5 \
  --train-batch-size 2 \
  --profile-window-multiplier 4 \
  --profile-rollout-batch-size 2 \
  --profile-max-new-tokens 4096 \
  --gradient-keep-ratios 0.2,0.4,0.6,0.8,1.0 \
  --launcher-env 'TRAINER_LOGGER=["console","tensorboard"]'
```

This writes:

- `runs/policy_grid_8gpu/policy_grid_manifest.json`
- per-policy run logs under `runs/policy_grid_8gpu/policies/*/launcher.log`

### 2) Track progress across all policies

```bash
./.venv/bin/python scripts/track_policy_grid_progress.py \
  --run-root runs/policy_grid_8gpu \
  --watch-seconds 30
```

This reports and writes:

- `runs/policy_grid_8gpu/progress/latest_status.csv`
- `runs/policy_grid_8gpu/progress/step_metrics_long.csv`
- `runs/policy_grid_8gpu/progress/family_step_summary.csv`

Tracked metrics include train loss, validation accuracy (GSM8K/MATH), and response length (mean/std) vs step/time.

### 3) Shutdown all policy-grid jobs

```bash
bash scripts/shutdown_policy_grid_8gpu.sh runs/policy_grid_8gpu
```

The shutdown script reads policy PIDs from `policy_grid_manifest.json` and sends `SIGTERM` to active jobs.

### Current Pseudocode (Policy Grid + Periodic Gradient Refresh)

```text
Inputs:
  mode ∈ {continuation, scratch}
  policy_launch_mode ∈ {continuation, fresh}
  gradient_update_mode ∈ {static, periodic}
  gpu_ids = [g0, g1, ..., gN]
  policies = [dapo_top, pods_top, random, gradient_top_r020, ...]

1) Resolve start checkpoint:
  if mode == continuation:
    start_ckpt = --start-from-path (global_step_K)
  else if mode == scratch:
    run warmup training to --checkpoint-step = K
    start_ckpt = warmup/global_step_K

2) Split policies:
  if gradient_update_mode == periodic:
    periodic_policies = {p | p.family startswith "gradient"}
    static_policies   = policies - periodic_policies
  else:
    periodic_policies = {}
    static_policies   = policies

3) Build static selector pool once (only if static_policies non-empty):
  profile start_ckpt on V1/V2
  for each p in static_policies:
    selected_train[p] = build_profile_selector_dataset(
      metric=p.metric, selector=p.selector, keep_ratio/keep_count
    )

4) Launch one policy per GPU (index aligned):
  for i, p in enumerate(policies):
    gpu = gpu_ids[i]
    if p in periodic_policies:
      launch run_periodic_gradient_selector.py on gpu with:
        --start-from-path start_ckpt
        --window-steps W
        --total-training-steps K + arm_training_steps
        --selector-metric p.metric
        --selector p.selector
        --keep-ratio/--keep-count from p
      # gradient direction refreshes every window
    else:
      if policy_launch_mode == continuation:
        branch start_ckpt -> policy run dir
        launch continuation launcher on selected_train[p]
      else:
        launch base launcher (fresh) on selected_train[p]

5) Write policy_grid_manifest.json:
  store per-policy:
    policy, gpu_id, pid, run_dir, log_path, tensorboard_dir
    job_type ∈ {static_selector, periodic_gradient}
```

## Download-Only Tracking Data (rsync)

### Minimal progress-only sync (no env/model/checkpoints)

```bash
rsync -avz --progress \
  --exclude='.venv/' \
  --exclude='thirdparty/' \
  --exclude='checkpoints/' \
  --exclude='**/global_step_*/' \
  --exclude='**/actor/' \
  --exclude='**/optimizer/' \
  --exclude='**/hf_model/' \
  --exclude='**/tensorboard_log/' \
  --exclude='outputs/' \
  --exclude='analysis/' \
  --include='runs/**/policy_grid_manifest.json' \
  --include='runs/**/periodic_gradient_selector_manifest.json' \
  --include='runs/**/launcher.log' \
  --include='runs/**/profile.log' \
  --include='runs/**/progress/*.csv' \
  --include='runs/**/selector_pools/**/*.csv' \
  --include='runs/**/selector_pools/**/*.json' \
  --include='runs/**/profile/**/*.csv' \
  --include='runs/**/profile/**/*.json' \
  --include='scripts/*.py' \
  --include='README.md' \
  --exclude='*' \
  siyuanch@CATALYST-FLEET.PC.CS.CMU.EDU:/path/to/EffiRL/ \
  ./EffiRL_progress_only/
```

### Rollout-aware sync (for richer analysis)

Use this if you also want rollout artifacts and detailed delta reports generated on the server.

```bash
rsync -avz --progress \
  --exclude='.venv/' \
  --exclude='thirdparty/' \
  --exclude='checkpoints/' \
  --exclude='**/global_step_*/' \
  --exclude='**/actor/' \
  --exclude='**/optimizer/' \
  --exclude='**/hf_model/' \
  --exclude='**/tensorboard_log/' \
  --exclude='outputs/' \
  --include='runs/**/policy_grid_manifest.json' \
  --include='runs/**/periodic_gradient_selector_manifest.json' \
  --include='runs/**/launcher.log' \
  --include='runs/**/profile.log' \
  --include='runs/**/progress/*.csv' \
  --include='runs/**/selector_pools/**/*.csv' \
  --include='runs/**/selector_pools/**/*.json' \
  --include='runs/**/profile/rollout_details.csv' \
  --include='runs/**/profile/ground_truth_profile.csv' \
  --include='runs/**/profile/**/*.json' \
  --include='runs/**/analysis/*.csv' \
  --include='runs/**/analysis/*.json' \
  --include='scripts/*.py' \
  --include='README.md' \
  --exclude='*' \
  siyuanch@CATALYST-FLEET.PC.CS.CMU.EDU:/path/to/EffiRL/ \
  ./EffiRL_rollout_tracking/
```

## Progress + Learned-Delta Reports

### A) Keep track of training progress

```bash
./.venv/bin/python scripts/track_policy_grid_progress.py \
  --run-root runs/policy_grid_8gpu \
  --watch-seconds 30
```

Useful outputs:

- `runs/policy_grid_8gpu/progress/latest_status.csv`
- `runs/policy_grid_8gpu/progress/step_metrics_long.csv`
- `runs/policy_grid_8gpu/progress/family_step_summary.csv`

### B) Aggregate learned delta (per policy/family)

```bash
./.venv/bin/python scripts/export_policy_learning_delta.py \
  --run-root runs/policy_grid_8gpu
```

Outputs:

- `runs/policy_grid_8gpu/analysis/per_run_specs.json`
- `runs/policy_grid_8gpu/analysis/per_run_delta.csv`
- `runs/policy_grid_8gpu/analysis/family_delta_summary.csv`

### C) Detailed examples: newly solved / forgotten (train + validation)

This compares two checkpoints and reports per-example changes.

```bash
./.venv/bin/python scripts/report_newly_solved_examples.py \
  --start-step-dir <run>/global_step_15 \
  --end-step-dir <run>/global_step_100 \
  --train-files data/verl/gsm8k/train.parquet data/verl/math/train.parquet \
  --val-files data/verl/validation_500.parquet \
  --max-new-tokens 1024 \
  --output-dir runs/analysis/newly_solved_step15_to_100
```

Detailed outputs:

- `delta_per_example.csv` (start/end solved + delta per problem)
- `newly_solved_examples.csv`
- `summary_by_split_source.csv`
- `start_predictions.csv`
- `end_predictions.csv`
