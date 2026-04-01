#!/usr/bin/env python3
"""Collect a larger rollout-only pool for worst-case generation length targets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
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
build_model = PROFILE_MODULE.build_model
encode_prompt = PROFILE_MODULE.encode_prompt
extract_response_ids = PROFILE_MODULE.extract_response_ids
make_tokenizer = PROFILE_MODULE.make_tokenizer
model_context_limit = PROFILE_MODULE.model_context_limit
prepare_generation_batch = PROFILE_MODULE.prepare_generation_batch
read_examples = PROFILE_MODULE.read_examples
render_prompt = PROFILE_MODULE.render_prompt


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
        default=[Path("data/verl/gsm8k/test.parquet")],
    )
    parser.add_argument(
        "--exclude-files",
        type=Path,
        nargs="*",
        default=[Path("data/verl/validation_500.parquet")],
    )
    parser.add_argument("--prompt-count", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--repeats-per-prompt", type=int, default=3)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/worst_case_length_pool_gsm8k_ckpt200"),
    )
    return parser.parse_args()


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint_into_model(model, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load checkpoint cleanly: {checkpoint_path}\nmissing={missing[:10]}\nunexpected={unexpected[:10]}"
        )


def build_prompt_pool(
    data_files: list[Path],
    exclude_files: list[Path],
    prompt_count: int,
    seed: int,
) -> tuple[list[Example], dict[str, Any]]:
    all_examples = read_examples(data_files)
    excluded_keys: set[str] = set()
    for path in exclude_files:
        if path.exists():
            excluded_keys.update(example.key for example in read_examples([path]))

    pool = [example for example in all_examples if example.key not in excluded_keys]
    if len(pool) < prompt_count:
        raise ValueError(f"Need at least {prompt_count} prompts after exclusion, got {len(pool)}.")

    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = pool[:prompt_count]
    manifest = {
        "seed": seed,
        "pool_size": len(pool),
        "excluded_size": len(excluded_keys),
        "prompt_count": len(selected),
        "selected_keys": [example.key for example in selected],
        "sources": {
            data_source: sum(example.data_source == data_source for example in selected)
            for data_source in sorted({example.data_source for example in selected})
        },
    }
    return selected, manifest


def rollout_lengths_batched(
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
) -> list[dict[str, Any]]:
    group_rows: list[dict[str, Any]] = []
    eos_token_id = tokenizer.eos_token_id

    for batch_start in range(0, len(examples), rollout_batch_size):
        batch_examples = examples[batch_start : batch_start + rollout_batch_size]
        prompt_ids_by_group = [encode_prompt(tokenizer, example.prompt_messages, max_prompt_tokens) for example in batch_examples]
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
        group_lengths: list[list[int]] = [[] for _ in batch_examples]
        for flat_idx, sequence in enumerate(sequences):
            response_ids = extract_response_ids(sequence[prompt_padded_len:], eos_token_id)
            group_idx = flat_idx // group_size
            group_lengths[group_idx].append(len(response_ids))

        for group_idx, example in enumerate(batch_examples):
            lengths = group_lengths[group_idx]
            group_rows.append(
                {
                    "example_key": example.key,
                    "prompt_index": int(example.extra_info.get("index", -1)),
                    "data_source": example.data_source,
                    "prompt_tokens": len(prompt_ids_by_group[group_idx]),
                    "prompt_text": render_prompt(tokenizer, example.prompt_messages),
                    "sample_lengths": json.dumps(lengths),
                    "mean_length": float(sum(lengths) / len(lengths)),
                    "max_length": int(max(lengths)),
                    "min_length": int(min(lengths)),
                    "batch_wall_time_s": float(batch_wall_time_s),
                }
            )

    return group_rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_examples, manifest = build_prompt_pool(
        data_files=args.data_files,
        exclude_files=args.exclude_files,
        prompt_count=args.prompt_count,
        seed=args.seed,
    )

    tokenizer = make_tokenizer(args.base_model)
    model = build_model(args.base_model, device)
    checkpoint_path = args.checkpoint_dir / f"global_step_{args.checkpoint_step}" / "actor" / "model_world_size_1_rank_0.pt"
    load_checkpoint_into_model(model, checkpoint_path)
    model.eval()

    repeat_rows: list[dict[str, Any]] = []
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for repeat_id in range(args.repeats_per_prompt):
        print(f"[collect] repeat {repeat_id + 1}/{args.repeats_per_prompt}", flush=True)
        rows = rollout_lengths_batched(
            model=model,
            tokenizer=tokenizer,
            examples=selected_examples,
            group_size=args.group_size,
            rollout_batch_size=args.rollout_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        for row in rows:
            full_row = {
                "repeat_id": repeat_id,
                "checkpoint_step": args.checkpoint_step,
                "group_size": args.group_size,
                **row,
            }
            repeat_rows.append(full_row)
            by_key[row["example_key"]].append(full_row)

    aggregated_rows: list[dict[str, Any]] = []
    for example in selected_examples:
        rows = by_key[example.key]
        max_lengths = [int(row["max_length"]) for row in rows]
        mean_lengths = [float(row["mean_length"]) for row in rows]
        aggregated_rows.append(
            {
                "checkpoint_step": args.checkpoint_step,
                "group_size": args.group_size,
                "repeats_per_prompt": args.repeats_per_prompt,
                "example_key": example.key,
                "prompt_index": int(example.extra_info.get("index", -1)),
                "data_source": example.data_source,
                "prompt_tokens": int(rows[0]["prompt_tokens"]),
                "prompt_text": rows[0]["prompt_text"],
                "expected_worst_case_length": float(sum(max_lengths) / len(max_lengths)),
                "worst_case_length_std": float(torch.tensor(max_lengths, dtype=torch.float32).std(unbiased=False).item()),
                "mean_sample_length": float(sum(mean_lengths) / len(mean_lengths)),
                "max_lengths": json.dumps(max_lengths),
                "mean_lengths": json.dumps(mean_lengths),
            }
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "checkpoint_step": args.checkpoint_step,
                "group_size": args.group_size,
                "repeats_per_prompt": args.repeats_per_prompt,
                "rollout_batch_size": args.rollout_batch_size,
                "max_new_tokens": args.max_new_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(repeat_rows, args.output_dir / "worst_case_length_repeats.csv")
    write_csv(aggregated_rows, args.output_dir / "worst_case_length_profile.csv")
    print(f"Wrote {args.output_dir / 'worst_case_length_profile.csv'}")


if __name__ == "__main__":
    main()
