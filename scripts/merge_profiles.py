#!/usr/bin/env python3
"""Merge rollout and training profiles at the bucket level.

The first implementation intentionally aggregates by `bucket_id` and `phase`
instead of pretending we already have exact per-sample training attribution.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-csv", type=Path, required=True)
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return 1.0
        if lowered in {"false", "no"}:
            return 0.0
    return float(value)


def main() -> None:
    args = parse_args()
    rollout_rows = read_csv(args.rollout_csv)
    training_rows = read_csv(args.training_csv)

    bucketed: dict[tuple[str, str], dict[str, float | str]] = {}

    for row in rollout_rows:
        key = (row.get("bucket_id") or "unbucketed", row.get("phase") or "unknown")
        entry = bucketed.setdefault(
            key,
            {
                "bucket_id": key[0],
                "phase": key[1],
                "num_samples": 0.0,
                "mean_reward_sum": 0.0,
                "mean_group_normalized_reward_sum": 0.0,
                "pass_count": 0.0,
                "rollout_cost_s": 0.0,
                "total_tokens": 0.0,
            },
        )
        entry["num_samples"] += 1.0
        entry["mean_reward_sum"] += as_float(row.get("reward"))
        entry["mean_group_normalized_reward_sum"] += as_float(
            row.get("group_normalized_reward")
        )
        entry["pass_count"] += as_float(row.get("passed"))
        entry["rollout_cost_s"] += as_float(row.get("rollout_wall_time_s"))
        entry["total_tokens"] += as_float(row.get("total_tokens"))

    train_cost_by_key: dict[tuple[str, str], float] = defaultdict(float)
    phase_train_cost: dict[str, float] = defaultdict(float)
    for row in training_rows:
        key = (row.get("bucket_id") or "unbucketed", row.get("phase") or "unknown")
        train_cost = as_float(row.get("train_wall_time_s"))
        train_cost_by_key[key] += train_cost
        phase_train_cost[key[1]] += train_cost

    samples_by_phase: dict[str, float] = defaultdict(float)
    for entry in bucketed.values():
        samples_by_phase[str(entry["phase"])] += float(entry["num_samples"])

    fieldnames = [
        "bucket_id",
        "phase",
        "num_samples",
        "pass_rate",
        "mean_reward",
        "mean_group_normalized_reward",
        "rollout_cost_s",
        "train_cost_s",
        "total_cost_s",
        "compute_efficiency",
        "statistical_efficiency",
        "goodput_proxy",
    ]

    output_rows = []
    for key, entry in sorted(bucketed.items()):
        num_samples = float(entry["num_samples"]) or 1.0
        train_cost_s = train_cost_by_key[key]
        if train_cost_s == 0.0 and phase_train_cost[key[1]] > 0.0:
            phase_total = samples_by_phase[key[1]] or num_samples
            train_cost_s = phase_train_cost[key[1]] * (num_samples / phase_total)
        total_cost_s = float(entry["rollout_cost_s"]) + train_cost_s
        mean_group_normalized_reward = float(
            entry["mean_group_normalized_reward_sum"]
        ) / num_samples
        compute_efficiency = 0.0 if total_cost_s == 0.0 else num_samples / total_cost_s
        statistical_efficiency = mean_group_normalized_reward
        goodput_proxy = (
            0.0 if total_cost_s == 0.0 else statistical_efficiency / total_cost_s
        )

        output_rows.append(
            {
                "bucket_id": entry["bucket_id"],
                "phase": entry["phase"],
                "num_samples": int(num_samples),
                "pass_rate": float(entry["pass_count"]) / num_samples,
                "mean_reward": float(entry["mean_reward_sum"]) / num_samples,
                "mean_group_normalized_reward": mean_group_normalized_reward,
                "rollout_cost_s": float(entry["rollout_cost_s"]),
                "train_cost_s": train_cost_s,
                "total_cost_s": total_cost_s,
                "compute_efficiency": compute_efficiency,
                "statistical_efficiency": statistical_efficiency,
                "goodput_proxy": goodput_proxy,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} merged rows to {args.output_csv}")


if __name__ == "__main__":
    main()
