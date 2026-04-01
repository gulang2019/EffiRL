#!/usr/bin/env python3
"""Offline GRPO ground-truth profiling for saved Qwen LoRA checkpoints.

This script runs a small offline pilot around finished verl checkpoints:

1. Build a deterministic held-out pool from parquet files.
2. Optionally exclude examples that were used in validation during training.
3. Split the held-out pool into V1 / V2.
4. For each checkpoint:
   - estimate a reference gradient on V1
   - run batched rollouts on V2
   - compute per-group gradients on saved rollouts
   - score each group by gradient projection onto the reference gradient

The objective matches the dominant GRPO actor terms used in the launcher:
clipped PPO policy loss plus low-variance KL-to-reference. The entropy term is
disabled by default because its coefficient is tiny in this recipe and it adds
substantial profiling cost.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.reward_score import default_compute_score


EPS = 1e-6


@dataclass
class Example:
    data_source: str
    prompt_messages: list[dict[str, str]]
    ground_truth: str
    extra_info: dict[str, Any]
    key: str


@dataclass
class RolloutSample:
    response_ids: list[int]
    response_text: str
    response_length: int
    reward: float


@dataclass
class RolloutGroup:
    example: Example
    prompt_ids: list[int]
    samples: list[RolloutSample]
    rollout_cost_s: float = 0.0
    batch_wall_time_s: float = 0.0
    sample_rollout_costs_s: list[float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/EffiRL/qwen2.5-1.5b-grpo-math-single-gpu-vllm"),
        help="Directory containing global_step_* actor checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200],
        help="Checkpoint global steps to profile.",
    )
    parser.add_argument(
        "--data-files",
        type=Path,
        nargs="*",
        default=[Path("data/verl/gsm8k/test.parquet"), Path("data/verl/math/test.parquet")],
        help="Held-out parquet files used to build the profiling pool.",
    )
    parser.add_argument(
        "--exclude-files",
        type=Path,
        nargs="*",
        default=[Path("data/verl/validation_500.parquet")],
        help="Optional parquet files to exclude from the profiling pool.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--v1-size", type=int, default=8, help="Number of prompts used for the reference gradient.")
    parser.add_argument(
        "--v2-size",
        type=int,
        default=12,
        help="Number of prompts scored per checkpoint. Non-positive means score all remaining prompts after V1.",
    )
    parser.add_argument("--group-size", type=int, default=5, help="GRPO rollout count per prompt.")
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=4,
        help="Number of prompt-groups rolled out together on V1/V2.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Generation cap. Non-positive means use the model context limit minus padded prompt length.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--clip-ratio-c", type=float, default=3.0)
    parser.add_argument("--kl-loss-coef", type=float, default=0.001)
    parser.add_argument(
        "--use-kl-loss",
        action="store_true",
        default=True,
        help="Include KL-to-reference in the offline objective.",
    )
    parser.add_argument(
        "--disable-kl-loss",
        action="store_false",
        dest="use_kl_loss",
        help="Disable KL-to-reference in the offline objective.",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.0,
        help="Optional entropy coefficient. Default is 0 for a faster pilot.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model path used to reconstruct the LoRA policy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/profiling"),
        help="Directory for profiling outputs.",
    )
    return parser.parse_args()


def read_examples(parquet_paths: list[Path]) -> list[Example]:
    records: list[Example] = []
    for path in parquet_paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            extra_info = row.get("extra_info") or {}
            data_source = row["data_source"]
            index = extra_info.get("index", -1)
            key = f"{data_source}::{index}"
            records.append(
                Example(
                    data_source=data_source,
                    prompt_messages=row["prompt"],
                    ground_truth=row["reward_model"]["ground_truth"],
                    extra_info=extra_info,
                    key=key,
                )
            )
    return records


def build_heldout_split(
    data_files: list[Path],
    exclude_files: list[Path],
    seed: int,
    v1_size: int,
    v2_size: int,
) -> tuple[list[Example], list[Example], dict[str, Any]]:
    all_examples = read_examples(data_files)
    excluded_keys: set[str] = set()
    for path in exclude_files:
        if path.exists():
            excluded_keys.update(example.key for example in read_examples([path]))

    pool = [example for example in all_examples if example.key not in excluded_keys]
    effective_v2_size = len(pool) - v1_size if v2_size <= 0 else v2_size
    if len(pool) < v1_size + effective_v2_size:
        raise ValueError(
            f"Need at least {v1_size + effective_v2_size} profiling examples after exclusion, got {len(pool)}."
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    v1 = pool[:v1_size]
    v2 = pool[v1_size:] if v2_size <= 0 else pool[v1_size : v1_size + v2_size]

    def count_sources(examples: list[Example]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for example in examples:
            counts[example.data_source] = counts.get(example.data_source, 0) + 1
        return counts

    manifest = {
        "seed": seed,
        "pool_size": len(pool),
        "excluded_size": len(excluded_keys),
        "v1_size": len(v1),
        "v2_size": len(v2),
        "requested_v2_size": v2_size,
        "v1_sources": count_sources(v1),
        "v2_sources": count_sources(v2),
        "v1_keys": [example.key for example in v1],
        "v2_keys": [example.key for example in v2],
    }
    return v1, v2, manifest


def make_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_model(base_model: str, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.eval()
    return model


def render_prompt(tokenizer, prompt_messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_prompt(tokenizer, prompt_messages: list[dict[str, str]], max_prompt_tokens: int) -> list[int]:
    text = render_prompt(tokenizer, prompt_messages)
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_prompt_tokens).input_ids
    return list(ids)


def prepare_generation_batch(
    tokenizer,
    prompt_ids_list: list[list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer.padding_side = "left"
    padded = tokenizer.pad(
        {"input_ids": prompt_ids_list},
        padding=True,
        return_tensors="pt",
    )
    return padded["input_ids"].to(device), padded["attention_mask"].to(device)


def extract_response_ids(generated_suffix: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return generated_suffix
    result: list[int] = []
    for token_id in generated_suffix:
        result.append(token_id)
        if token_id == eos_token_id:
            break
    return result


def attribute_batch_rollout_cost(total_batch_time_s: float, response_lengths: list[int], group_size: int) -> list[float]:
    if not response_lengths:
        return []
    max_len = max(response_lengths)
    if max_len <= 0 or total_batch_time_s <= 0:
        return [0.0 for _ in range(len(response_lengths) // group_size)]

    active_counts = []
    for step in range(1, max_len + 1):
        active_counts.append(sum(length >= step for length in response_lengths))

    group_count = len(response_lengths) // group_size
    group_costs = [0.0 for _ in range(group_count)]
    per_step_batch_time = total_batch_time_s / max_len

    for sequence_idx, length in enumerate(response_lengths):
        group_idx = sequence_idx // group_size
        cost = 0.0
        for step in range(length):
            cost += per_step_batch_time / active_counts[step]
        group_costs[group_idx] += cost

    return group_costs


def attribute_batch_rollout_costs(
    total_batch_time_s: float,
    response_lengths: list[int],
    group_size: int,
) -> tuple[list[float], list[float]]:
    if not response_lengths:
        return [], []
    max_len = max(response_lengths)
    if max_len <= 0 or total_batch_time_s <= 0:
        sequence_costs = [0.0 for _ in response_lengths]
        group_costs = [0.0 for _ in range(len(response_lengths) // group_size)]
        return group_costs, sequence_costs

    active_counts = [sum(length >= step for length in response_lengths) for step in range(1, max_len + 1)]
    per_step_batch_time = total_batch_time_s / max_len
    sequence_costs: list[float] = []
    for length in response_lengths:
        cost = 0.0
        for step in range(length):
            cost += per_step_batch_time / active_counts[step]
        sequence_costs.append(cost)

    group_count = len(response_lengths) // group_size
    group_costs = [0.0 for _ in range(group_count)]
    for sequence_idx, sequence_cost in enumerate(sequence_costs):
        group_costs[sequence_idx // group_size] += sequence_cost
    return group_costs, sequence_costs


def model_context_limit(model, tokenizer) -> int:
    candidates: list[int] = []
    config_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(config_limit, int) and config_limit > 0:
        candidates.append(config_limit)
    token_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(token_limit, int) and 0 < token_limit < 1_000_000:
        candidates.append(token_limit)
    if not candidates:
        raise ValueError("Could not infer a valid context limit from model/tokenizer configuration.")
    return min(candidates)


def rollout_groups_batched(
    model,
    tokenizer,
    examples: list[Example],
    group_size: int,
    rollout_batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    measure_time: bool,
) -> list[RolloutGroup]:
    groups: list[RolloutGroup] = []
    eos_token_id = tokenizer.eos_token_id

    for batch_start in range(0, len(examples), rollout_batch_size):
        batch_examples = examples[batch_start : batch_start + rollout_batch_size]
        prompt_ids_by_group = [encode_prompt(tokenizer, ex.prompt_messages, max_prompt_tokens) for ex in batch_examples]

        flat_prompt_ids: list[list[int]] = []
        for prompt_ids in prompt_ids_by_group:
            for _ in range(group_size):
                flat_prompt_ids.append(prompt_ids)

        input_ids, attention_mask = prepare_generation_batch(tokenizer, flat_prompt_ids, device)
        prompt_padded_len = input_ids.shape[1]

        start = time.perf_counter()
        effective_max_new_tokens = max_new_tokens
        if effective_max_new_tokens <= 0:
            effective_max_new_tokens = model_context_limit(model, tokenizer) - prompt_padded_len
        if effective_max_new_tokens <= 0:
            raise ValueError(
                f"Prompt length {prompt_padded_len} exceeds or matches the model context limit; cannot generate."
            )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=effective_max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
            )
        batch_wall_time_s = time.perf_counter() - start

        sequences = outputs.sequences.detach().cpu().tolist()
        response_lengths: list[int] = []
        flat_samples: list[RolloutSample] = []
        for flat_idx, sequence in enumerate(sequences):
            response_ids = extract_response_ids(sequence[prompt_padded_len:], eos_token_id)
            if response_ids and eos_token_id is not None and response_ids[-1] == eos_token_id:
                decode_ids = response_ids[:-1]
            else:
                decode_ids = response_ids
            response_text = tokenizer.decode(decode_ids, skip_special_tokens=True)
            example = batch_examples[flat_idx // group_size]
            reward = float(
                default_compute_score(
                    example.data_source,
                    response_text,
                    example.ground_truth,
                    extra_info=example.extra_info,
                )
            )
            flat_samples.append(
                RolloutSample(
                    response_ids=response_ids,
                    response_text=response_text,
                    response_length=len(response_ids),
                    reward=reward,
                )
            )
            response_lengths.append(len(response_ids))

        group_costs, sequence_costs = attribute_batch_rollout_costs(
            total_batch_time_s=batch_wall_time_s if measure_time else 0.0,
            response_lengths=response_lengths,
            group_size=group_size,
        )

        for group_idx, example in enumerate(batch_examples):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            groups.append(
                RolloutGroup(
                    example=example,
                    prompt_ids=prompt_ids_by_group[group_idx],
                    samples=flat_samples[start_idx:end_idx],
                    rollout_cost_s=group_costs[group_idx] if measure_time else 0.0,
                    batch_wall_time_s=batch_wall_time_s if measure_time else 0.0,
                    sample_rollout_costs_s=sequence_costs[start_idx:end_idx] if measure_time else None,
                )
            )

    return groups


def scalar_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) == 1:
        return [0.0]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / max(len(rewards) - 1, 1)
    std = math.sqrt(variance)
    return [(reward - mean) / (std + EPS) for reward in rewards]


def response_loss_terms(
    model,
    prompt_ids: list[int],
    response_ids: list[int],
    advantage_scalar: float,
    clip_ratio: float,
    clip_ratio_c: float,
    use_kl_loss: bool,
    kl_loss_coef: float,
    entropy_coeff: float,
) -> tuple[torch.Tensor, int]:
    full_ids = torch.tensor([prompt_ids + response_ids], device=model.device, dtype=torch.long)
    prompt_len = len(prompt_ids)
    response_len = len(response_ids)
    attention_mask = torch.ones_like(full_ids)

    outputs = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    response_logits = logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]

    target_ids = full_ids[:, prompt_len : prompt_len + response_len]
    flat_response_logits = response_logits.reshape(-1, response_logits.size(-1))
    flat_target_ids = target_ids.reshape(-1)
    # Avoid materializing full [T, vocab] log-softmax tensors to reduce peak memory.
    token_log_probs = (-F.cross_entropy(flat_response_logits, flat_target_ids, reduction="none")).view(-1)
    old_log_probs = token_log_probs.detach()

    advantages = torch.full_like(token_log_probs, float(advantage_scalar))
    ratio = torch.exp(token_log_probs - old_log_probs)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    loss_sum = pg_losses.sum()

    if entropy_coeff:
        log_probs = F.log_softmax(response_logits.float(), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).squeeze(0)
        loss_sum = loss_sum - entropy_coeff * entropy.sum()

    if use_kl_loss:
        with torch.no_grad():
            with model.disable_adapter():
                ref_outputs = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_response_logits = ref_logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
                flat_ref_logits = ref_response_logits.reshape(-1, ref_response_logits.size(-1))
                ref_token_log_probs = (-F.cross_entropy(flat_ref_logits, flat_target_ids, reduction="none")).view(-1)
        kld = kl_penalty(token_log_probs, ref_token_log_probs, "low_var_kl")
        loss_sum = loss_sum + kl_loss_coef * kld.sum()

    return loss_sum, response_len


def trainable_grad_vector(model) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float32))
        else:
            chunks.append(param.grad.detach().float().view(-1).cpu())
    return torch.cat(chunks)


def group_gradient_vector(
    model,
    group: RolloutGroup,
    clip_ratio: float,
    clip_ratio_c: float,
    use_kl_loss: bool,
    kl_loss_coef: float,
    entropy_coeff: float,
) -> tuple[torch.Tensor, float]:
    rewards = [sample.reward for sample in group.samples]
    advantages = scalar_advantages(rewards)
    total_tokens = sum(max(sample.response_length, 1) for sample in group.samples)

    model.zero_grad(set_to_none=True)
    grad_start = time.perf_counter()
    for sample, advantage_scalar in zip(group.samples, advantages):
        if sample.response_length == 0:
            continue
        loss_sum, response_len = response_loss_terms(
            model=model,
            prompt_ids=group.prompt_ids,
            response_ids=sample.response_ids,
            advantage_scalar=advantage_scalar,
            clip_ratio=clip_ratio,
            clip_ratio_c=clip_ratio_c,
            use_kl_loss=use_kl_loss,
            kl_loss_coef=kl_loss_coef,
            entropy_coeff=entropy_coeff,
        )
        scaled_loss = loss_sum / total_tokens
        scaled_loss.backward()
    grad_time_s = time.perf_counter() - grad_start
    grad_vector = trainable_grad_vector(model)
    model.zero_grad(set_to_none=True)
    return grad_vector, grad_time_s


def load_checkpoint_into_model(model, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load checkpoint cleanly: {checkpoint_path}\nmissing={missing[:10]}\nunexpected={unexpected[:10]}"
        )


def estimate_reference_gradient(
    model,
    v1_groups: list[RolloutGroup],
    clip_ratio: float,
    clip_ratio_c: float,
    use_kl_loss: bool,
    kl_loss_coef: float,
    entropy_coeff: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    grad_sum: torch.Tensor | None = None
    grad_times: list[float] = []

    for group in v1_groups:
        grad_vector, grad_time_s = group_gradient_vector(
            model=model,
            group=group,
            clip_ratio=clip_ratio,
            clip_ratio_c=clip_ratio_c,
            use_kl_loss=use_kl_loss,
            kl_loss_coef=kl_loss_coef,
            entropy_coeff=entropy_coeff,
        )
        if grad_sum is None:
            grad_sum = grad_vector
        else:
            grad_sum += grad_vector
        grad_times.append(grad_time_s)

    assert grad_sum is not None
    g_ref = grad_sum / len(v1_groups)
    summary = {
        "v1_group_count": len(v1_groups),
        "g_ref_norm": float(torch.linalg.vector_norm(g_ref).item()),
        "mean_v1_grad_time_s": float(sum(grad_times) / len(grad_times)),
    }
    return g_ref, summary


def profile_checkpoint(
    model,
    tokenizer,
    checkpoint_step: int,
    checkpoint_dir: Path,
    v1_examples: list[Example],
    v2_examples: list[Example],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = checkpoint_dir / f"global_step_{checkpoint_step}" / "actor" / "model_world_size_1_rank_0.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    load_checkpoint_into_model(model, checkpoint_path)

    v1_groups = rollout_groups_batched(
        model=model,
        tokenizer=tokenizer,
        examples=v1_examples,
        group_size=args.group_size,
        rollout_batch_size=args.rollout_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        measure_time=False,
    )
    g_ref, ref_summary = estimate_reference_gradient(
        model=model,
        v1_groups=v1_groups,
        clip_ratio=args.clip_ratio,
        clip_ratio_c=args.clip_ratio_c,
        use_kl_loss=args.use_kl_loss,
        kl_loss_coef=args.kl_loss_coef,
        entropy_coeff=args.entropy_coeff,
    )
    g_ref_norm = float(torch.linalg.vector_norm(g_ref).item())

    v2_groups = rollout_groups_batched(
        model=model,
        tokenizer=tokenizer,
        examples=v2_examples,
        group_size=args.group_size,
        rollout_batch_size=args.rollout_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        measure_time=True,
    )

    rows: list[dict[str, Any]] = []
    rollout_detail_rows: list[dict[str, Any]] = []
    for group in v2_groups:
        grad_vector, grad_time_s = group_gradient_vector(
            model=model,
            group=group,
            clip_ratio=args.clip_ratio,
            clip_ratio_c=args.clip_ratio_c,
            use_kl_loss=args.use_kl_loss,
            kl_loss_coef=args.kl_loss_coef,
            entropy_coeff=args.entropy_coeff,
        )
        dot = float(torch.dot(grad_vector, g_ref).item())
        proj = float(dot / (g_ref_norm + EPS))
        total_cost_s = float(group.rollout_cost_s + grad_time_s)
        computational_efficiency = float(1.0 / (total_cost_s + EPS))
        rewards = [sample.reward for sample in group.samples]
        response_lengths = [sample.response_length for sample in group.samples]
        reward_mean = float(sum(rewards) / len(rewards))
        pods_statistical_efficiency = float(reward_mean * (1.0 - reward_mean))
        dapo_keep_indicator = int(0.0 < reward_mean < 1.0)
        unsolved_indicator = int(reward_mean < 1.0)
        rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "checkpoint_id": checkpoint_step,
                "example_key": group.example.key,
                "data_source": group.example.data_source,
                "prompt_index": group.example.extra_info.get("index"),
                "group_size": args.group_size,
                "prompt_tokens": len(group.prompt_ids),
                "completion_tokens_total": sum(response_lengths),
                "completion_tokens_max": max(response_lengths) if response_lengths else 0,
                "completion_lengths": json.dumps(response_lengths),
                "rewards": json.dumps(rewards),
                "reward_mean": reward_mean,
                "reward_std": float(torch.tensor(rewards, dtype=torch.float32).std().item()) if len(rewards) > 1 else 0.0,
                "group_accuracy": reward_mean,
                "pods_statistical_efficiency": pods_statistical_efficiency,
                "dapo_keep_indicator": dapo_keep_indicator,
                "dapo_unsolved_indicator": unsolved_indicator,
                "dapo_statistical_efficiency": float(dapo_keep_indicator),
                "dapo_keep_efficiency": float(dapo_keep_indicator),
                "C_prefill": 0.0,
                "C_decode": float(group.rollout_cost_s),
                "C_roll": float(group.rollout_cost_s),
                "rollout_cost_s": float(group.rollout_cost_s),
                "T_grad": float(grad_time_s),
                "grad_cost_s": float(grad_time_s),
                "total_cost_s": total_cost_s,
                "stat_eff_dot": dot,
                "stat_eff_proj": proj,
                "statistical_efficiency": proj,
                "gradient_statistical_efficiency": proj,
                "computational_efficiency": computational_efficiency,
                "gradient_dot": dot,
                "gradient_projection": proj,
                "goodput": float(proj / (total_cost_s + EPS)),
                "gradient_goodput": float(proj / (total_cost_s + EPS)),
                "pods_goodput": float(pods_statistical_efficiency / (total_cost_s + EPS)),
                "dapo_goodput": float(dapo_keep_indicator / (total_cost_s + EPS)),
                "dapo_keep_goodput": float(dapo_keep_indicator / (total_cost_s + EPS)),
                "g_ref_norm": g_ref_norm,
            }
        )
        sample_costs = group.sample_rollout_costs_s or [0.0 for _ in group.samples]
        for sample_idx, (sample, sample_cost_s) in enumerate(zip(group.samples, sample_costs, strict=True)):
            rollout_detail_rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_id": checkpoint_step,
                    "example_key": group.example.key,
                    "data_source": group.example.data_source,
                    "prompt_index": group.example.extra_info.get("index"),
                    "sample_index": sample_idx,
                    "group_size": args.group_size,
                    "prompt_tokens": len(group.prompt_ids),
                    "response_length": sample.response_length,
                    "reward": sample.reward,
                    "group_accuracy": reward_mean,
                    "dapo_keep_indicator": dapo_keep_indicator,
                    "dapo_unsolved_indicator": unsolved_indicator,
                    "sample_rollout_cost_s": float(sample_cost_s),
                    "group_rollout_cost_s": float(group.rollout_cost_s),
                    "group_total_cost_s": total_cost_s,
                    "batch_wall_time_s": float(group.batch_wall_time_s),
                    "response_text": sample.response_text,
                }
            )

    checkpoint_summary = {
        "checkpoint_step": checkpoint_step,
        **ref_summary,
        "v2_group_count": len(v2_groups),
        "mean_v2_rollout_cost_s": float(sum(group.rollout_cost_s for group in v2_groups) / len(v2_groups)),
        "mean_v2_grad_cost_s": float(sum(row["T_grad"] for row in rows) / len(rows)),
        "mean_v2_total_cost_s": float(sum(row["total_cost_s"] for row in rows) / len(rows)),
        "mean_v2_completion_tokens": float(
            sum(sum(sample.response_length for sample in group.samples) for group in v2_groups) / len(v2_groups)
        ),
        "mean_v2_gradient_statistical_efficiency": float(sum(row["gradient_statistical_efficiency"] for row in rows) / len(rows)),
        "mean_v2_pods_statistical_efficiency": float(sum(row["pods_statistical_efficiency"] for row in rows) / len(rows)),
        "mean_v2_dapo_statistical_efficiency": float(sum(row["dapo_statistical_efficiency"] for row in rows) / len(rows)),
        "mean_v2_dapo_keep_efficiency": float(sum(row["dapo_keep_efficiency"] for row in rows) / len(rows)),
    }
    return rows, rollout_detail_rows, checkpoint_summary


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v1_examples, v2_examples, manifest = build_heldout_split(
        data_files=args.data_files,
        exclude_files=args.exclude_files,
        seed=args.seed,
        v1_size=args.v1_size,
        v2_size=args.v2_size,
    )
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    tokenizer = make_tokenizer(args.base_model)
    model = build_model(args.base_model, device)

    all_rows: list[dict[str, Any]] = []
    all_rollout_detail_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for checkpoint_step in args.checkpoint_steps:
        print(f"[profiling] checkpoint {checkpoint_step}", flush=True)
        rows, rollout_detail_rows, summary = profile_checkpoint(
            model=model,
            tokenizer=tokenizer,
            checkpoint_step=checkpoint_step,
            checkpoint_dir=args.checkpoint_dir,
            v1_examples=v1_examples,
            v2_examples=v2_examples,
            args=args,
            device=device,
        )
        all_rows.extend(rows)
        all_rollout_detail_rows.extend(rollout_detail_rows)
        summaries.append(summary)
        write_csv(all_rows, args.output_dir / "ground_truth_profile.csv")
        write_csv(all_rollout_detail_rows, args.output_dir / "rollout_details.csv")
        (args.output_dir / "checkpoint_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(
            f"[profiling] checkpoint {checkpoint_step} complete: {len(rows)} rows, g_ref_norm={summary['g_ref_norm']:.6g}",
            flush=True,
        )
        torch.cuda.empty_cache()

    csv_path = args.output_dir / "ground_truth_profile.csv"
    write_csv(all_rows, csv_path)
    write_csv(all_rollout_detail_rows, args.output_dir / "rollout_details.csv")
    (args.output_dir / "checkpoint_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_rows)} profiling rows to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
