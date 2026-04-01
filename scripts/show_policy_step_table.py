#!/usr/bin/env python3
"""Show step-aligned policy metrics in a terminal-friendly table."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_COLUMNS = [
    "actor_loss",
    "gsm8k_acc",
    "math_acc",
    "response_length_mean",
    "response_length_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Defaults to <run-root>/progress/step_metrics_long.csv",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Target global step. If omitted, use latest common step across all policies.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated metric columns to display.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="policy",
        help="Column used to sort output rows.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Optional output CSV for the displayed table.",
    )
    return parser.parse_args()


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return None
    try:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return float(value)
    except ValueError:
        return value


def ensure_progress_csv(run_root: Path, input_csv: Path) -> None:
    if input_csv.exists():
        return
    tracker = ROOT_DIR / "scripts" / "track_policy_grid_progress.py"
    py = ROOT_DIR / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    cmd = [str(py), str(tracker), "--run-root", str(run_root)]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing progress CSV after tracker run: {input_csv}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {key: parse_scalar(value) for key, value in row.items()}
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def latest_common_step(rows: list[dict[str, Any]]) -> int:
    steps_by_policy: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        policy = row.get("policy")
        step = row.get("global_step")
        if isinstance(policy, str) and isinstance(step, int):
            steps_by_policy[policy].add(step)
    if not steps_by_policy:
        raise ValueError("No valid policy/global_step rows found.")
    common = None
    for _, steps in steps_by_policy.items():
        common = set(steps) if common is None else (common & steps)
    if not common:
        raise ValueError("No common global_step exists across all policies yet.")
    return max(common)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    headers = ["policy", "family", "global_step"] + columns
    widths: dict[str, int] = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(fmt(row.get(h))))
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(fmt(row.get(h)).ljust(widths[h]) for h in headers))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["policy", "family", "global_step"] + columns
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h) for h in headers})


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or (args.run_root / "progress" / "step_metrics_long.csv")
    columns = [x.strip() for x in args.columns.split(",") if x.strip()]
    ensure_progress_csv(args.run_root, input_csv)
    rows = load_rows(input_csv)

    target_step = args.step if args.step is not None else latest_common_step(rows)
    selected = [row for row in rows if row.get("global_step") == target_step]
    if not selected:
        raise ValueError(f"No rows found for global_step={target_step} in {input_csv}")

    sort_key = args.sort_by
    selected.sort(key=lambda r: (str(r.get(sort_key, "")), str(r.get("policy", ""))))

    print(f"run_root: {args.run_root}")
    print(f"step: {target_step}")
    print_table(selected, columns)

    if args.export_csv is not None:
        write_csv(args.export_csv, selected, columns)
        print(f"\nWrote: {args.export_csv}")


if __name__ == "__main__":
    main()
