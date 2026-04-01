#!/usr/bin/env python3
"""Prepare local parquet files for the single-GPU DAPO baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/dapo/train.parquet"),
        help="Where to write the processed DAPO-Math-17k train parquet.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("data/dapo/aime_2024.parquet"),
        help="Where to write the processed AIME 2024 eval parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_train(output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"Skipping existing train parquet: {output_path}")
        return

    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "all", split="train")
    ensure_parent(output_path)
    dataset.to_parquet(str(output_path))
    print(f"Wrote {len(dataset)} train rows to {output_path}")


def write_eval(output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"Skipping existing eval parquet: {output_path}")
        return

    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    def map_example(example: dict, idx: int) -> dict:
        prompt = (
            example["Problem"].strip()
            + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
        )
        return {
            "data_source": "aime24",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "Math",
            "reward_model": {"style": "rule", "ground_truth": str(example["Answer"])},
            "extra_info": {
                "index": idx,
                "split": "test",
                "id": example["ID"],
                "solution": example["Solution"],
                "question": example["Problem"],
            },
        }

    processed = dataset.map(map_example, with_indices=True, remove_columns=dataset.column_names)
    ensure_parent(output_path)
    processed.to_parquet(str(output_path))
    print(f"Wrote {len(processed)} eval rows to {output_path}")


def main() -> None:
    args = parse_args()
    write_train(args.train_output, args.force)
    write_eval(args.eval_output, args.force)


if __name__ == "__main__":
    main()
