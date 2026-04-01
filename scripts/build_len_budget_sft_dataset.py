#!/usr/bin/env python3
"""Build a len-budget SFT dataset from sampled rollout JSONL."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-rollouts",
        type=Path,
        default=Path("outputs/metric_alignment_rollouts_math_train_ckpt200_full/sample_rollouts.jsonl"),
        help="Input rollout JSONL from collect_metric_alignment_rollouts.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/len_budget_sft_math_ckpt200"),
        help="Directory for all/train/val/test JSONL plus manifest.",
    )
    parser.add_argument(
        "--budget-bins",
        type=int,
        nargs="+",
        default=[256, 512, 768, 1024],
        help="Ascending upper-bound bins for len_budget labels.",
    )
    parser.add_argument(
        "--slack-tokens",
        type=int,
        default=32,
        help="Additive slack on top of the observed correct-length maximum.",
    )
    parser.add_argument(
        "--min-correct-per-prompt",
        type=int,
        default=1,
        help="Skip prompts with fewer than this many correct samples.",
    )
    parser.add_argument("--train-prompt-fraction", type=float, default=0.8)
    parser.add_argument("--val-prompt-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def ensure_valid_bins(budget_bins: list[int]) -> list[int]:
    if not budget_bins:
        raise ValueError("budget_bins must be non-empty.")
    sorted_bins = sorted(int(value) for value in budget_bins)
    if sorted_bins != budget_bins:
        raise ValueError(f"budget_bins must be in ascending order, got {budget_bins}.")
    if len(set(sorted_bins)) != len(sorted_bins):
        raise ValueError(f"budget_bins must be unique, got {budget_bins}.")
    if any(value <= 0 for value in sorted_bins):
        raise ValueError(f"budget_bins must be positive, got {budget_bins}.")
    return sorted_bins


def read_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                skipped_lines += 1
                continue
            if not isinstance(row, dict):
                skipped_lines += 1
                continue
            rows.append(row)
    return rows, skipped_lines


def select_budget_bin(raw_budget: int, budget_bins: list[int]) -> tuple[int, str]:
    for upper in budget_bins:
        if raw_budget <= upper:
            return upper, f"lb_{upper}"
    return budget_bins[-1], f"lb_{budget_bins[-1]}"


def make_assistant_content(len_budget_bin: str, response_text: str) -> str:
    stripped = response_text.lstrip()
    if stripped:
        return f"<forecast len_budget={len_budget_bin}>\n{stripped}"
    return f"<forecast len_budget={len_budget_bin}>"


def split_prompt_keys(prompt_keys: list[str], train_fraction: float, val_fraction: float, seed: int) -> dict[str, set[str]]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_prompt_fraction must be in (0, 1), got {train_fraction}.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_prompt_fraction must be in (0, 1), got {val_fraction}.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train_prompt_fraction + val_prompt_fraction must be < 1, got {train_fraction + val_fraction}."
        )

    ordered_keys = sorted(prompt_keys)
    rng = random.Random(seed)
    rng.shuffle(ordered_keys)

    total = len(ordered_keys)
    if total < 3:
        raise ValueError(f"Need at least 3 prompts to build train/val/test splits, got {total}.")

    train_count = max(1, int(round(total * train_fraction)))
    val_count = max(1, int(round(total * val_fraction)))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    train_keys = set(ordered_keys[:train_count])
    val_keys = set(ordered_keys[train_count : train_count + val_count])
    test_keys = set(ordered_keys[train_count + val_count :])
    return {"train": train_keys, "val": val_keys, "test": test_keys}


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def summarize_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(row["meta"]["len_budget_bin"] for row in rows)
    return {label: int(counter[label]) for label in sorted(counter)}


def main() -> None:
    args = parse_args()
    budget_bins = ensure_valid_bins(args.budget_bins)
    max_budget = budget_bins[-1]

    rollout_rows, skipped_lines = read_jsonl_rows(args.sample_rollouts)
    if not rollout_rows:
        raise ValueError(f"No usable rollout rows found in {args.sample_rollouts}.")

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        example_key = row.get("example_key")
        if not isinstance(example_key, str) or not example_key:
            continue
        grouped_rows[example_key].append(row)

    sft_rows: list[dict[str, Any]] = []
    skipped_no_correct = 0
    skipped_too_few_correct = 0
    prompt_label_counts: Counter[str] = Counter()
    prompt_correct_counts: list[int] = []
    prompt_sample_counts: list[int] = []

    for example_key, rows in grouped_rows.items():
        correct_rows = [row for row in rows if int(row.get("passed", 0)) == 1]
        if not correct_rows:
            skipped_no_correct += 1
            continue
        if len(correct_rows) < args.min_correct_per_prompt:
            skipped_too_few_correct += 1
            continue

        correct_lengths = [int(row["response_length"]) for row in correct_rows]
        raw_budget = min(max(correct_lengths) + args.slack_tokens, max_budget)
        budget_upper, budget_label = select_budget_bin(raw_budget, budget_bins)

        prompt_label_counts[budget_label] += 1
        prompt_correct_counts.append(len(correct_rows))
        prompt_sample_counts.append(len(rows))

        representative = rows[0]
        prompt_index = int(representative.get("prompt_index", -1))
        data_source = str(representative.get("data_source", ""))

        for row in correct_rows:
            prompt_messages = row.get("prompt_messages")
            if not isinstance(prompt_messages, list):
                continue
            response_text = str(row.get("response_text", ""))
            assistant_content = make_assistant_content(budget_label, response_text)
            sft_rows.append(
                {
                    "messages": [
                        *prompt_messages,
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "meta": {
                        "task": "len_budget_sft",
                        "example_key": example_key,
                        "prompt_index": prompt_index,
                        "data_source": data_source,
                        "len_budget_bin": budget_label,
                        "declared_budget_tokens": budget_upper,
                        "raw_budget_tokens": raw_budget,
                        "response_length": int(row["response_length"]),
                        "prompt_correct_count": len(correct_rows),
                        "prompt_sample_count": len(rows),
                        "sample_index": int(row.get("sample_index", -1)),
                        "group_size": int(row.get("group_size", 0)),
                        "reward": float(row.get("reward", 0.0)),
                    },
                }
            )

    if not sft_rows:
        raise ValueError("No SFT rows were produced after filtering for correct samples.")

    split_keys = split_prompt_keys(
        prompt_keys=list({row["meta"]["example_key"] for row in sft_rows}),
        train_fraction=args.train_prompt_fraction,
        val_fraction=args.val_prompt_fraction,
        seed=args.seed,
    )

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name, keys in split_keys.items():
        split_rows[split_name] = [row for row in sft_rows if row["meta"]["example_key"] in keys]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(sft_rows, args.output_dir / "all.jsonl")
    for split_name, rows in split_rows.items():
        write_jsonl(rows, args.output_dir / f"{split_name}.jsonl")

    manifest = {
        "sample_rollouts": str(args.sample_rollouts),
        "input_rollout_rows": len(rollout_rows),
        "skipped_json_lines": skipped_lines,
        "grouped_prompts": len(grouped_rows),
        "usable_prompts": len({row["meta"]["example_key"] for row in sft_rows}),
        "skipped_no_correct_prompts": skipped_no_correct,
        "skipped_too_few_correct_prompts": skipped_too_few_correct,
        "sft_examples": len(sft_rows),
        "budget_bins": budget_bins,
        "slack_tokens": args.slack_tokens,
        "min_correct_per_prompt": args.min_correct_per_prompt,
        "prompt_label_counts": {label: int(prompt_label_counts[label]) for label in sorted(prompt_label_counts)},
        "split_prompt_counts": {split_name: len(keys) for split_name, keys in split_keys.items()},
        "split_example_counts": {split_name: len(rows) for split_name, rows in split_rows.items()},
        "split_label_counts": {split_name: summarize_labels(rows) for split_name, rows in split_rows.items()},
        "avg_correct_samples_per_prompt": (
            float(sum(prompt_correct_counts) / len(prompt_correct_counts)) if prompt_correct_counts else 0.0
        ),
        "avg_total_samples_per_prompt": (
            float(sum(prompt_sample_counts) / len(prompt_sample_counts)) if prompt_sample_counts else 0.0
        ),
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_dir / 'all.jsonl'} ({len(sft_rows)} rows)")
    print(f"Wrote {args.output_dir / 'train.jsonl'} ({len(split_rows['train'])} rows)")
    print(f"Wrote {args.output_dir / 'val.jsonl'} ({len(split_rows['val'])} rows)")
    print(f"Wrote {args.output_dir / 'test.jsonl'} ({len(split_rows['test'])} rows)")
    print(f"Wrote {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
