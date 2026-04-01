#!/usr/bin/env python3
"""Build a retained parquet dataset from profile scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-csv", type=Path, required=True, help="CSV with per-example profile scores.")
    parser.add_argument(
        "--source-parquets",
        type=Path,
        nargs="+",
        required=True,
        help="Source parquet files used to recover the original rows.",
    )
    parser.add_argument("--checkpoint-step", type=int, required=True, help="Checkpoint step to select from.")
    parser.add_argument(
        "--metric",
        type=str,
        default="gradient_statistical_efficiency",
        help="Score column used for ranking.",
    )
    parser.add_argument(
        "--selector",
        choices=["top", "bottom", "random"],
        default="top",
        help="How to select rows after ranking the scored pool.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=None,
        help="Fraction of scored examples to keep. Specify exactly one of --keep-ratio or --keep-count.",
    )
    parser.add_argument(
        "--keep-count",
        type=int,
        default=None,
        help="Absolute number of scored examples to keep. Specify exactly one of --keep-ratio or --keep-count.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed used for random selection and tie-breaking.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Output parquet path.")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON path. Defaults to <output-parquet stem>.manifest.json",
    )
    parser.add_argument(
        "--output-selection-csv",
        type=Path,
        default=None,
        help="Optional CSV containing selected keys and scores.",
    )
    return parser.parse_args()


def load_profile_rows(path: Path, checkpoint_step: int, metric: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if int(row["checkpoint_step"]) == checkpoint_step]

    if not rows:
        raise ValueError(f"No profile rows found for checkpoint_step={checkpoint_step} in {path}.")
    if metric not in rows[0]:
        raise ValueError(f"Metric column '{metric}' not found in {path}.")

    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_key = row["example_key"]
        if example_key in dedup:
            raise ValueError(f"Duplicate example_key '{example_key}' for checkpoint_step={checkpoint_step}.")
        dedup[example_key] = {
            "example_key": example_key,
            "score": float(row[metric]),
            "data_source": row.get("data_source"),
            "prompt_index": row.get("prompt_index"),
        }
    return list(dedup.values())


def load_source_table(paths: list[Path]) -> pa.Table:
    tables = [pq.read_table(path) for path in paths]
    if not tables:
        raise ValueError("No source parquet files provided.")
    return pa.concat_tables(tables, promote_options="default")


def example_key_from_row(row: dict[str, Any]) -> str:
    data_source = row.get("data_source")
    extra_info = row.get("extra_info") or {}
    index = extra_info.get("index")
    if data_source is None or index is None:
        raise ValueError(f"Cannot derive example key from row: data_source={data_source!r}, extra_info={extra_info!r}")
    return f"{data_source}::{index}"


def resolve_keep_count(total: int, keep_ratio: float | None, keep_count: int | None) -> int:
    if (keep_ratio is None) == (keep_count is None):
        raise ValueError("Specify exactly one of --keep-ratio or --keep-count.")
    if total <= 0:
        raise ValueError("Cannot select from an empty scored pool.")
    if keep_ratio is not None:
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError("--keep-ratio must be in (0, 1].")
        return max(1, min(total, math.floor(total * keep_ratio)))
    assert keep_count is not None
    if keep_count <= 0:
        raise ValueError("--keep-count must be positive.")
    return min(total, keep_count)


def select_rows(
    scored_rows: list[dict[str, Any]],
    selector: str,
    keep_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if selector == "random":
        chosen = rng.sample(scored_rows, keep_count)
        return sorted(chosen, key=lambda row: row["example_key"])

    randomized = list(scored_rows)
    rng.shuffle(randomized)
    reverse = selector == "top"
    ranked = sorted(randomized, key=lambda row: row["score"], reverse=reverse)
    return ranked[:keep_count]


def write_selection_csv(path: Path, rows: list[dict[str, Any]], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_key", metric, "data_source", "prompt_index"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "example_key": row["example_key"],
                    metric: row["score"],
                    "data_source": row.get("data_source"),
                    "prompt_index": row.get("prompt_index"),
                }
            )


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = [row["score"] for row in rows]
    return {
        "min": float(min(scores)),
        "max": float(max(scores)),
        "mean": float(sum(scores) / len(scores)),
    }


def main() -> None:
    args = parse_args()
    scored_rows = load_profile_rows(args.profile_csv, args.checkpoint_step, args.metric)
    keep_count = resolve_keep_count(len(scored_rows), args.keep_ratio, args.keep_count)
    selected_score_rows = select_rows(scored_rows, args.selector, keep_count, args.seed)

    source_table = load_source_table(args.source_parquets)
    source_rows = source_table.to_pylist()
    selected_keys = {row["example_key"] for row in selected_score_rows}
    matched_indices: list[int] = []
    matched_keys: set[str] = set()
    for index, row in enumerate(source_rows):
        example_key = example_key_from_row(row)
        if example_key in selected_keys:
            matched_indices.append(index)
            matched_keys.add(example_key)

    missing_keys = sorted(selected_keys - matched_keys)
    if missing_keys:
        raise ValueError(
            f"Selected {len(selected_keys)} keys but only matched {len(matched_keys)} source rows. "
            f"Missing examples include: {missing_keys[:10]}"
        )

    subset = source_table.take(pa.array(matched_indices, type=pa.int64()))
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, args.output_parquet)

    manifest_path = args.output_manifest
    if manifest_path is None:
        manifest_path = args.output_parquet.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_selection_csv is not None:
        write_selection_csv(args.output_selection_csv, selected_score_rows, args.metric)

    manifest = {
        "profile_csv": str(args.profile_csv),
        "source_parquets": [str(path) for path in args.source_parquets],
        "checkpoint_step": args.checkpoint_step,
        "metric": args.metric,
        "selector": args.selector,
        "seed": args.seed,
        "scored_pool_size": len(scored_rows),
        "keep_count": keep_count,
        "selected_row_count": subset.num_rows,
        "output_parquet": str(args.output_parquet),
        "output_selection_csv": str(args.output_selection_csv) if args.output_selection_csv is not None else None,
        "selected_score_summary": summarize_scores(selected_score_rows),
        "pool_score_summary": summarize_scores(scored_rows),
        "selected_keys_preview": [row["example_key"] for row in selected_score_rows[:20]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {subset.num_rows} rows to {args.output_parquet}")
    print(f"Wrote {manifest_path}")
    if args.output_selection_csv is not None:
        print(f"Wrote {args.output_selection_csv}")


if __name__ == "__main__":
    main()
