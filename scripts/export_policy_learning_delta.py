#!/usr/bin/env python3
"""Export per-policy run specs and learned deltas from policy-grid runs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROGRESS_CSV = "progress/step_metrics_long.csv"
DEFAULT_METRICS = [
    "actor_loss",
    "gsm8k_acc",
    "math_acc",
    "response_length_mean",
    "response_length_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--progress-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric names to compute deltas for.",
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


def ensure_progress_csv(run_root: Path, progress_csv: Path) -> None:
    if progress_csv.exists():
        return
    tracker = ROOT_DIR / "scripts" / "track_policy_grid_progress.py"
    py = ROOT_DIR / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    subprocess.run([str(py), str(tracker), "--run-root", str(run_root)], cwd=ROOT_DIR, check=True)
    if not progress_csv.exists():
        raise FileNotFoundError(f"Missing progress CSV after tracker run: {progress_csv}")


def load_progress_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: parse_scalar(v) for k, v in row.items()})
    return rows


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or (args.run_root / "policy_grid_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    progress_csv = args.progress_csv or (args.run_root / DEFAULT_PROGRESS_CSV)
    ensure_progress_csv(args.run_root, progress_csv)
    progress_rows = load_progress_rows(progress_csv)
    if not progress_rows:
        raise ValueError(f"No rows in progress CSV: {progress_csv}")

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    if not metrics:
        raise ValueError("--metrics resolved to empty set.")

    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in progress_rows:
        policy = row.get("policy")
        step = row.get("global_step")
        if not isinstance(policy, str) or not isinstance(step, int):
            continue
        by_policy[policy].append(row)

    jobs = manifest.get("jobs", [])
    per_run_specs: list[dict[str, Any]] = []
    per_run_delta_rows: list[dict[str, Any]] = []

    for job in jobs:
        policy = job.get("policy", {})
        policy_name = policy.get("name")
        if not isinstance(policy_name, str):
            continue

        rows = sorted(by_policy.get(policy_name, []), key=lambda r: r["global_step"])
        if not rows:
            per_run_specs.append(
                {
                    "policy": policy_name,
                    "family": policy.get("family"),
                    "gpu_id": job.get("gpu_id"),
                    "pid": job.get("pid"),
                    "job_type": job.get("job_type"),
                    "has_progress": False,
                    "run_dir": job.get("run_dir"),
                    "log_path": job.get("log_path"),
                    "selected_train_parquet": job.get("selected_train_parquet"),
                    "selection_csv": job.get("selection_csv"),
                    "periodic_run_root": job.get("periodic_run_root"),
                    "metric": policy.get("metric"),
                    "selector": policy.get("selector"),
                    "keep_ratio": policy.get("keep_ratio"),
                    "keep_count": policy.get("keep_count"),
                }
            )
            continue

        first = rows[0]
        last = rows[-1]
        spec = {
            "policy": policy_name,
            "family": policy.get("family"),
            "gpu_id": job.get("gpu_id"),
            "pid": job.get("pid"),
            "job_type": job.get("job_type"),
            "has_progress": True,
            "num_points": len(rows),
            "start_step": first.get("global_step"),
            "end_step": last.get("global_step"),
            "start_elapsed_train_s": first.get("elapsed_train_s"),
            "end_elapsed_train_s": last.get("elapsed_train_s"),
            "start_elapsed_total_s": first.get("elapsed_total_s"),
            "end_elapsed_total_s": last.get("elapsed_total_s"),
            "run_dir": job.get("run_dir"),
            "log_path": job.get("log_path"),
            "selected_train_parquet": job.get("selected_train_parquet"),
            "selection_csv": job.get("selection_csv"),
            "periodic_run_root": job.get("periodic_run_root"),
            "metric": policy.get("metric"),
            "selector": policy.get("selector"),
            "keep_ratio": policy.get("keep_ratio"),
            "keep_count": policy.get("keep_count"),
        }
        per_run_specs.append(spec)

        delta_row: dict[str, Any] = {
            "policy": policy_name,
            "family": policy.get("family"),
            "start_step": first.get("global_step"),
            "end_step": last.get("global_step"),
            "num_points": len(rows),
        }
        for metric in metrics:
            start_v = as_float(first.get(metric))
            end_v = as_float(last.get(metric))
            delta_row[f"{metric}_start"] = start_v
            delta_row[f"{metric}_end"] = end_v
            delta_row[f"{metric}_delta"] = (end_v - start_v) if (start_v is not None and end_v is not None) else None
        per_run_delta_rows.append(delta_row)

    family_rows: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run_delta_rows:
        family = row.get("family")
        if isinstance(family, str):
            by_family[family].append(row)
    for family, rows in sorted(by_family.items()):
        out: dict[str, Any] = {"family": family, "num_runs": len(rows)}
        for metric in metrics:
            key = f"{metric}_delta"
            vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                out[f"{metric}_delta_mean"] = mean(vals)
                out[f"{metric}_delta_std"] = pstdev(vals) if len(vals) > 1 else 0.0
            else:
                out[f"{metric}_delta_mean"] = None
                out[f"{metric}_delta_std"] = None
        family_rows.append(out)

    output_dir = args.output_dir or (args.run_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_json = output_dir / "per_run_specs.json"
    delta_csv = output_dir / "per_run_delta.csv"
    family_csv = output_dir / "family_delta_summary.csv"

    specs_json.write_text(json.dumps(per_run_specs, indent=2), encoding="utf-8")
    write_csv(delta_csv, per_run_delta_rows)
    write_csv(family_csv, family_rows)

    print(f"Wrote: {specs_json}")
    print(f"Wrote: {delta_csv}")
    print(f"Wrote: {family_csv}")
    print("\nTop-level summary:")
    for row in sorted(per_run_delta_rows, key=lambda r: str(r.get("policy"))):
        loss_delta = row.get("actor_loss_delta")
        math_delta = row.get("math_acc_delta")
        gsm_delta = row.get("gsm8k_acc_delta")
        print(
            f"policy={row.get('policy')} step={row.get('start_step')}->{row.get('end_step')} "
            f"loss_delta={loss_delta} math_delta={math_delta} gsm8k_delta={gsm_delta}"
        )


if __name__ == "__main__":
    main()
