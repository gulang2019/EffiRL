#!/usr/bin/env python3
"""Join rollout details with prompt and ground-truth context."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-details-csv", type=Path, required=True)
    parser.add_argument("--data-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def render_prompt(prompt_messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in prompt_messages)


def load_examples(data_files: list[Path]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for path in data_files:
        table = pq.read_table(path)
        for row in table.to_pylist():
            extra = row.get("extra_info") or {}
            key = f"{row['data_source']}::{extra.get('index', -1)}"
            lookup[key] = {
                "prompt_text": render_prompt(row["prompt"]),
                "ground_truth": row["reward_model"]["ground_truth"],
            }
    return lookup


def main() -> None:
    args = parse_args()
    example_lookup = load_examples(args.data_files)

    with args.rollout_details_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) + ["prompt_text", "ground_truth"] if rows else []

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            extra = example_lookup.get(row["example_key"], {})
            row["prompt_text"] = extra.get("prompt_text", "")
            row["ground_truth"] = extra.get("ground_truth", "")
            writer.writerow(row)

    print(args.output_csv)


if __name__ == "__main__":
    main()
