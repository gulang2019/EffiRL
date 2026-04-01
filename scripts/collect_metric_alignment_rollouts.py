#!/usr/bin/env python3
"""Collect raw rollouts and prompt-level metric targets for self-alignment SFT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROFILE_HELPER_PATH = Path(__file__).resolve().with_name("profile_grpo_ground_truth.py")
PROFILE_SPEC = importlib.util.spec_from_file_location("profile_grpo_ground_truth", PROFILE_HELPER_PATH)
if PROFILE_SPEC is None or PROFILE_SPEC.loader is None:
    raise ImportError(f"Could not load helper module from {PROFILE_HELPER_PATH}")
PROFILE_MODULE = importlib.util.module_from_spec(PROFILE_SPEC)
sys.modules[PROFILE_SPEC.name] = PROFILE_MODULE
PROFILE_SPEC.loader.exec_module(PROFILE_MODULE)

Example = PROFILE_MODULE.Example
attribute_batch_rollout_costs = PROFILE_MODULE.attribute_batch_rollout_costs
build_model = PROFILE_MODULE.build_model
encode_prompt = PROFILE_MODULE.encode_prompt
extract_response_ids = PROFILE_MODULE.extract_response_ids
load_checkpoint_into_model = PROFILE_MODULE.load_checkpoint_into_model
make_tokenizer = PROFILE_MODULE.make_tokenizer
model_context_limit = PROFILE_MODULE.model_context_limit
prepare_generation_batch = PROFILE_MODULE.prepare_generation_batch
read_examples = PROFILE_MODULE.read_examples
render_prompt = PROFILE_MODULE.render_prompt

from verl.utils.reward_score import default_compute_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/EffiRL/qwen2.5-1.5b-grpo-math-single-gpu-vllm"),
    )
    parser.add_argument("--checkpoint-step", type=int, default=200)
    parser.add_argument(
        "--data-files",
        type=Path,
        nargs="+",
        default=[Path("data/verl/math/train.parquet")],
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=0,
        help="Number of prompts to collect. Non-positive means all prompts in the selected shard.",
    )
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--repeats-per-prompt", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metric_alignment_rollouts_math_train_ckpt200"),
    )
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row))
            handle.write("\n")


def append_jsonl(rows: list[dict[str, Any]], handle) -> None:
    for row in rows:
        handle.write(json_dumps(row))
        handle.write("\n")
    handle.flush()


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def accuracy_bin(pass_rate: float) -> str:
    if pass_rate < 0.25:
        return "0_25"
    if pass_rate < 0.50:
        return "25_50"
    if pass_rate < 0.75:
        return "50_75"
    return "75_100"


def stt_label(pass_rate: float) -> str:
    if pass_rate <= 0.0:
        return "fail"
    if pass_rate >= 1.0:
        return "solved"
    return "partial"


def length_budget_bin(length_value: float) -> str:
    thresholds = [32, 64, 128, 256, 512, 1024, 2048]
    for threshold in thresholds:
        if length_value <= threshold:
            return f"le_{threshold}"
    return f"gt_{thresholds[-1]}"


def build_prompt_pool(
    data_files: list[Path],
    prompt_count: int,
    seed: int,
    shard_index: int,
    num_shards: int,
) -> tuple[list[Example], dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}.")

    pool = read_examples(data_files)
    if not pool:
        raise ValueError("No examples found in the provided data files.")

    rng = random.Random(seed)
    rng.shuffle(pool)

    shard_start = math.floor(len(pool) * shard_index / num_shards)
    shard_end = math.floor(len(pool) * (shard_index + 1) / num_shards)
    shard_examples = pool[shard_start:shard_end]
    if prompt_count > 0:
        if len(shard_examples) < prompt_count:
            raise ValueError(
                f"Need at least {prompt_count} prompts in shard {shard_index}/{num_shards}, got {len(shard_examples)}."
            )
        shard_examples = shard_examples[:prompt_count]

    manifest = {
        "seed": seed,
        "pool_size": len(pool),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "shard_size": len(shard_examples),
        "prompt_count": len(shard_examples),
        "selected_keys": [example.key for example in shard_examples],
        "sources": {
            data_source: sum(example.data_source == data_source for example in shard_examples)
            for data_source in sorted({example.data_source for example in shard_examples})
        },
    }
    return shard_examples, manifest


def collect_rollouts(
    model,
    tokenizer,
    examples: list[Example],
    checkpoint_step: int,
    group_size: int,
    repeats_per_prompt: int,
    rollout_batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    sample_output_path: Path,
) -> tuple[int, list[dict[str, Any]]]:
    sample_row_count = 0
    prompt_stats: dict[str, dict[str, Any]] = {}
    eos_token_id = tokenizer.eos_token_id
    sample_output_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_output_path.exists():
        sample_output_path.unlink()

    with sample_output_path.open("w", encoding="utf-8") as sample_handle:
        for repeat_id in range(repeats_per_prompt):
            print(f"[collect] repeat {repeat_id + 1}/{repeats_per_prompt}", flush=True)
            for batch_start in range(0, len(examples), rollout_batch_size):
                batch_examples = examples[batch_start : batch_start + rollout_batch_size]
                prompt_ids_by_group = [
                    encode_prompt(tokenizer, example.prompt_messages, max_prompt_tokens) for example in batch_examples
                ]
                flat_prompt_ids: list[list[int]] = []
                for prompt_ids in prompt_ids_by_group:
                    for _ in range(group_size):
                        flat_prompt_ids.append(prompt_ids)

                input_ids, attention_mask = prepare_generation_batch(tokenizer, flat_prompt_ids, device)
                prompt_padded_len = input_ids.shape[1]
                effective_max_new_tokens = max_new_tokens
                if effective_max_new_tokens <= 0:
                    effective_max_new_tokens = model_context_limit(model, tokenizer) - prompt_padded_len
                if effective_max_new_tokens <= 0:
                    raise ValueError(
                        f"Prompt length {prompt_padded_len} exceeds or matches the model context limit; cannot generate."
                    )

                start = time.perf_counter()
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
                grouped_samples: list[list[dict[str, Any]]] = [[] for _ in batch_examples]
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
                    sample_record = {
                        "response_text": response_text,
                        "response_length": len(response_ids),
                        "reward": reward,
                        "passed": int(reward > 0.5),
                    }
                    grouped_samples[flat_idx // group_size].append(sample_record)
                    response_lengths.append(len(response_ids))

                _, sequence_costs = attribute_batch_rollout_costs(
                    total_batch_time_s=batch_wall_time_s,
                    response_lengths=response_lengths,
                    group_size=group_size,
                )

                batch_sample_rows: list[dict[str, Any]] = []
                sequence_cursor = 0
                for group_idx, example in enumerate(batch_examples):
                    group_prompt_text = render_prompt(tokenizer, example.prompt_messages)
                    group_rewards = [sample["reward"] for sample in grouped_samples[group_idx]]
                    group_accuracy = float(sum(group_rewards) / len(group_rewards))
                    group_stt_label = stt_label(group_accuracy)
                    stats = prompt_stats.setdefault(
                        example.key,
                        {
                            "checkpoint_step": checkpoint_step,
                            "example_key": example.key,
                            "prompt_index": int(example.extra_info.get("index", -1)),
                            "data_source": example.data_source,
                            "ground_truth": example.ground_truth,
                            "prompt_messages": example.prompt_messages,
                            "prompt_text": group_prompt_text,
                            "prompt_tokens": len(prompt_ids_by_group[group_idx]),
                            "total_samples": 0,
                            "pass_count": 0,
                            "reward_sum": 0.0,
                            "length_sum": 0.0,
                            "lengths": [],
                            "repeat_group_accuracies": [],
                        },
                    )
                    stats["repeat_group_accuracies"].append(group_accuracy)

                    for sample_idx, sample in enumerate(grouped_samples[group_idx]):
                        sample_cost_s = float(sequence_costs[sequence_cursor])
                        sequence_cursor += 1
                        stats["total_samples"] += 1
                        stats["pass_count"] += int(sample["passed"])
                        stats["reward_sum"] += float(sample["reward"])
                        stats["length_sum"] += float(sample["response_length"])
                        stats["lengths"].append(int(sample["response_length"]))
                        batch_sample_rows.append(
                            {
                                "checkpoint_step": checkpoint_step,
                                "repeat_id": repeat_id,
                                "example_key": example.key,
                                "prompt_index": int(example.extra_info.get("index", -1)),
                                "data_source": example.data_source,
                                "ground_truth": example.ground_truth,
                                "sample_index": sample_idx,
                                "group_size": group_size,
                                "prompt_tokens": len(prompt_ids_by_group[group_idx]),
                                "prompt_messages": example.prompt_messages,
                                "prompt_text": group_prompt_text,
                                "response_text": sample["response_text"],
                                "response_length": int(sample["response_length"]),
                                "reward": float(sample["reward"]),
                                "passed": int(sample["passed"]),
                                "group_accuracy": group_accuracy,
                                "group_stt_label": group_stt_label,
                                "batch_wall_time_s": float(batch_wall_time_s),
                                "sample_rollout_cost_s": sample_cost_s,
                            }
                        )

                append_jsonl(batch_sample_rows, sample_handle)
                sample_row_count += len(batch_sample_rows)
                print(
                    f"[collect] repeat={repeat_id} prompts={min(batch_start + len(batch_examples), len(examples))}/{len(examples)} samples={sample_row_count}",
                    flush=True,
                )

    prompt_rows: list[dict[str, Any]] = []
    for stats in prompt_stats.values():
        lengths = sorted(int(length) for length in stats.pop("lengths"))
        total_samples = int(stats["total_samples"])
        pass_rate = float(stats["pass_count"] / total_samples) if total_samples else 0.0
        mean_length = float(stats["length_sum"] / total_samples) if total_samples else 0.0
        p50_length = percentile(lengths, 0.50)
        p90_length = percentile(lengths, 0.90)
        p95_length = percentile(lengths, 0.95)
        p99_length = percentile(lengths, 0.99)
        prompt_rows.append(
            {
                **stats,
                "pass_rate": pass_rate,
                "accuracy_bin": accuracy_bin(pass_rate),
                "stt_label": stt_label(pass_rate),
                "mean_length": mean_length,
                "min_length": int(lengths[0]) if lengths else 0,
                "max_length": int(lengths[-1]) if lengths else 0,
                "p50_length": p50_length,
                "p90_length": p90_length,
                "p95_length": p95_length,
                "p99_length": p99_length,
                "mean_length_bin": length_budget_bin(mean_length),
                "p90_length_bin": length_budget_bin(p90_length),
                "max_length_bin": length_budget_bin(float(lengths[-1]) if lengths else 0.0),
                "repeat_group_accuracies": stats["repeat_group_accuracies"],
            }
        )

    prompt_rows.sort(key=lambda row: (row["data_source"], row["prompt_index"], row["example_key"]))
    return sample_row_count, prompt_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_examples, manifest = build_prompt_pool(
        data_files=args.data_files,
        prompt_count=args.prompt_count,
        seed=args.seed,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "checkpoint_step": args.checkpoint_step,
                "group_size": args.group_size,
                "repeats_per_prompt": args.repeats_per_prompt,
                "rollout_batch_size": args.rollout_batch_size,
                "max_prompt_tokens": args.max_prompt_tokens,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tokenizer = make_tokenizer(args.base_model)
    model = build_model(args.base_model, device)
    checkpoint_path = args.checkpoint_dir / f"global_step_{args.checkpoint_step}" / "actor" / "model_world_size_1_rank_0.pt"
    load_checkpoint_into_model(model, checkpoint_path)
    model.eval()

    sample_output_path = args.output_dir / "sample_rollouts.jsonl"
    sample_row_count, prompt_rows = collect_rollouts(
        model=model,
        tokenizer=tokenizer,
        examples=selected_examples,
        checkpoint_step=args.checkpoint_step,
        group_size=args.group_size,
        repeats_per_prompt=args.repeats_per_prompt,
        rollout_batch_size=args.rollout_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        sample_output_path=sample_output_path,
    )

    write_jsonl(prompt_rows, args.output_dir / "prompt_targets.jsonl")
    print(f"Wrote {sample_output_path} ({sample_row_count} rows)", flush=True)
    print(f"Wrote {args.output_dir / 'prompt_targets.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
