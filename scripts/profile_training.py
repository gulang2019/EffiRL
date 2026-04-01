#!/usr/bin/env python3
"""Normalize training-step logs into a flat CSV for later cost attribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "optimizer_step",
    "phase",
    "train_wall_time_s",
    "train_gpu_seconds",
    "samples_in_step",
    "tokens_in_step",
    "bucket_id",
]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimizer_step": record.get("optimizer_step"),
        "phase": record.get("phase"),
        "train_wall_time_s": record.get("train_wall_time_s"),
        "train_gpu_seconds": record.get("train_gpu_seconds"),
        "samples_in_step": record.get("samples_in_step"),
        "tokens_in_step": record.get("tokens_in_step"),
        "bucket_id": record.get("bucket_id"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Raw training JSONL.")
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

    print(f"Wrote {len(rows)} training rows to {args.output_csv}")


if __name__ == "__main__":
    main()
