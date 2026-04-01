#!/usr/bin/env python3
"""Normalize rollout logs into a flat CSV for analysis.

The exact `verl` log schema is not fixed yet, so this script accepts generic
JSONL records and extracts a stable subset of fields used by the metric spec.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "sample_id",
    "group_id",
    "task_id",
    "phase",
    "domain",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reward",
    "group_normalized_reward",
    "passed",
    "rollout_wall_time_s",
    "rollout_gpu_seconds",
    "bucket_id",
]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    total_tokens = record.get("total_tokens")
    if total_tokens is None:
        prompt = record.get("prompt_tokens") or 0
        completion = record.get("completion_tokens") or 0
        total_tokens = prompt + completion

    return {
        "sample_id": record.get("sample_id"),
        "group_id": record.get("group_id"),
        "task_id": record.get("task_id"),
        "phase": record.get("phase"),
        "domain": record.get("domain"),
        "prompt_tokens": record.get("prompt_tokens"),
        "completion_tokens": record.get("completion_tokens"),
        "total_tokens": total_tokens,
        "reward": record.get("reward"),
        "group_normalized_reward": record.get("group_normalized_reward"),
        "passed": record.get("passed"),
        "rollout_wall_time_s": record.get("rollout_wall_time_s"),
        "rollout_gpu_seconds": record.get("rollout_gpu_seconds"),
        "bucket_id": record.get("bucket_id"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Raw rollout JSONL.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Normalized CSV output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(normalize_record(json.loads(line)))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rollout rows to {args.output_csv}")


if __name__ == "__main__":
    main()
