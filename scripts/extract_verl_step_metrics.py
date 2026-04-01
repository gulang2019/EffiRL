#!/usr/bin/env python3
"""Extract step-level metrics from a verl console log into CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PREFERRED_FIELDS = [
    "step",
    "training/global_step",
    "training/epoch",
    "elapsed_train_s",
    "elapsed_total_s",
    "actor/loss",
    "critic/rewards/mean",
    "critic/score/mean",
    "timing_s/step",
    "timing_s/testing",
    "timing_s/save_checkpoint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-log", type=Path, required=True, help="verl worker .out log path")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV output path")
    return parser.parse_args()


def parse_scalar(raw: str):
    raw = raw.strip()
    if raw == "":
        return raw
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def parse_step_line(line: str) -> dict[str, object] | None:
    if "step:" not in line:
        return None

    payload = line[line.index("step:") :].strip()
    metrics: dict[str, object] = {}
    for chunk in payload.split(" - "):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        metrics[key.strip()] = parse_scalar(value)
    return metrics if metrics else None


def ordered_fields(rows: list[dict[str, object]]) -> list[str]:
    seen = {key for row in rows for key in row.keys()}
    ordered = [field for field in PREFERRED_FIELDS if field in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    elapsed_train = 0.0
    elapsed_total = 0.0

    with args.input_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = parse_step_line(line)
            if row is None:
                continue

            step_time = float(row.get("timing_s/step", 0.0) or 0.0)
            test_time = float(row.get("timing_s/testing", 0.0) or 0.0)
            save_time = float(row.get("timing_s/save_checkpoint", 0.0) or 0.0)

            elapsed_train += step_time
            elapsed_total += step_time + test_time + save_time

            row["elapsed_train_s"] = round(elapsed_train, 6)
            row["elapsed_total_s"] = round(elapsed_total, 6)
            rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fields(rows)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
