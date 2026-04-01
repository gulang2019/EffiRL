#!/usr/bin/env python3
"""Render policy metric curves in terminal using ASCII."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ASCII_LEVELS = " .:-=+*#%@"
DEFAULT_ALL_METRICS = [
    "actor_loss",
    "gsm8k_acc",
    "math_acc",
    "response_length_mean",
    "response_length_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument(
        "--metric",
        type=str,
        default="math_acc",
        help="Metric column name, or 'all' to render multiple default metrics.",
    )
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--policies", type=str, default="", help="Comma-separated policy names filter.")
    parser.add_argument("--max-policies", type=int, default=20)
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
    subprocess.run([str(py), str(tracker), "--run-root", str(run_root)], cwd=ROOT_DIR, check=True)
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing CSV after tracker run: {input_csv}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: parse_scalar(v) for k, v in row.items()})
    return rows


def compress_series(values: list[float], width: int) -> list[float]:
    if not values:
        return []
    if len(values) <= width:
        return values
    out: list[float] = []
    n = len(values)
    for i in range(width):
        start = int(i * n / width)
        end = int((i + 1) * n / width)
        if end <= start:
            end = start + 1
        bucket = values[start:end]
        out.append(sum(bucket) / len(bucket))
    return out


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        mid = len(ASCII_LEVELS) // 2
        return ASCII_LEVELS[mid] * len(values)
    chars = []
    for v in values:
        t = (v - vmin) / (vmax - vmin)
        idx = int(t * (len(ASCII_LEVELS) - 1))
        chars.append(ASCII_LEVELS[idx])
    return "".join(chars)


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or (args.run_root / "progress" / "step_metrics_long.csv")
    ensure_progress_csv(args.run_root, input_csv)
    rows = load_rows(input_csv)
    if not rows:
        raise ValueError(f"No rows in {input_csv}")

    print(f"run_root: {args.run_root}")
    print(f"metric: {args.metric}")
    print(f"input: {input_csv}")
    print()

    policy_filter = {x.strip() for x in args.policies.split(",") if x.strip()}
    metrics = DEFAULT_ALL_METRICS if args.metric == "all" else [args.metric]
    rendered_any = False
    for metric_name in metrics:
        series: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in rows:
            policy = row.get("policy")
            step = row.get("global_step")
            value = row.get(metric_name)
            if not isinstance(policy, str) or not isinstance(step, int) or not isinstance(value, (int, float)):
                continue
            if policy_filter and policy not in policy_filter:
                continue
            series[policy].append((step, float(value)))

        if not series:
            continue
        rendered_any = True
        print(f"=== {metric_name} ===")
        shown = 0
        for policy in sorted(series.keys()):
            if shown >= args.max_policies:
                break
            points = sorted(series[policy], key=lambda x: x[0])
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            ys_compact = compress_series(ys, args.width)
            line = sparkline(ys_compact)
            print(f"{policy}")
            print(
                f"  steps: {xs[0]} -> {xs[-1]} ({len(xs)} points)  min={min(ys):.4f} max={max(ys):.4f} last={ys[-1]:.4f}"
            )
            print(f"  {line}")
            shown += 1
        print()

    if not rendered_any:
        raise ValueError(f"No valid series for requested metric setting: '{args.metric}'")


if __name__ == "__main__":
    main()
