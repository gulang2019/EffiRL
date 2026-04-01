# EffiRL

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
