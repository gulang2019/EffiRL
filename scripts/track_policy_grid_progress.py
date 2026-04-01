#!/usr/bin/env python3
"""Track multi-policy training progress from launcher logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DEFAULT_GSM8K_ACC_KEY = "val-core/openai/gsm8k/acc/mean@1"
DEFAULT_MATH_ACC_KEY = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--math-acc-key", type=str, default=DEFAULT_MATH_ACC_KEY)
    parser.add_argument("--gsm8k-acc-key", type=str, default=DEFAULT_GSM8K_ACC_KEY)
    parser.add_argument("--watch-seconds", type=int, default=0, help="If >0, refresh every N seconds.")
    return parser.parse_args()


def parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return raw
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def parse_step_rows(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_path.exists():
        return rows
    elapsed_train = 0.0
    elapsed_total = 0.0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            if "step:" not in line:
                continue
            payload = line[line.index("step:") :].strip()
            row: dict[str, Any] = {}
            for chunk in payload.split(" - "):
                if ":" not in chunk:
                    continue
                key, value = chunk.split(":", 1)
                row[key.strip()] = parse_scalar(value)
            if not row:
                continue
            step_time = float(row.get("timing_s/step", 0.0) or 0.0)
            test_time = float(row.get("timing_s/testing", 0.0) or 0.0)
            save_time = float(row.get("timing_s/save_checkpoint", 0.0) or 0.0)
            elapsed_train += step_time
            elapsed_total += step_time + test_time + save_time
            row["elapsed_train_s"] = round(elapsed_train, 6)
            row["elapsed_total_s"] = round(elapsed_total, 6)
            rows.append(row)
    return rows


def parse_periodic_rows(periodic_run_root: Path) -> list[dict[str, Any]]:
    manifest_path = periodic_run_root / "periodic_gradient_selector_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for window in manifest.get("windows", []):
        log_path_raw = window.get("log_path")
        if not log_path_raw:
            continue
        log_path = Path(log_path_raw)
        window_rows = parse_step_rows(log_path)
        for row in window_rows:
            gs = row.get("training/global_step")
            if isinstance(gs, int) and gs in seen_steps:
                continue
            if isinstance(gs, int):
                seen_steps.add(gs)
            rows.append(row)
    return rows


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


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


def summarize(args: argparse.Namespace) -> None:
    manifest_path = args.manifest or (args.run_root / "policy_grid_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])

    status_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for job in jobs:
        policy = job["policy"]["name"]
        family = job["policy"]["family"]
        pid = int(job.get("pid", -1))
        log_path = Path(job["log_path"])
        job_type = job.get("job_type", "static_selector")
        if job_type == "periodic_gradient":
            periodic_root = Path(job.get("periodic_run_root", ""))
            rows = parse_periodic_rows(periodic_root)
        else:
            rows = parse_step_rows(log_path)
        latest = rows[-1] if rows else {}
        running = pid_is_running(pid)

        length_mean = latest.get("response_length/mean")
        length_std = latest.get("response_length/std")
        if length_std is None:
            length_std = latest.get("response_length_non_aborted/std")

        status_rows.append(
            {
                "policy": policy,
                "family": family,
                "gpu_id": job.get("gpu_id"),
                "pid": pid,
                "running": int(running),
                "steps_logged": len(rows),
                "step": latest.get("step"),
                "global_step": latest.get("training/global_step"),
                "elapsed_train_s": latest.get("elapsed_train_s"),
                "elapsed_total_s": latest.get("elapsed_total_s"),
                "actor_loss": latest.get("actor/loss"),
                "gsm8k_acc": latest.get(args.gsm8k_acc_key),
                "math_acc": latest.get(args.math_acc_key),
                "response_length_mean": length_mean,
                "response_length_std": length_std,
                "log_path": str(log_path),
            }
        )

        for row in rows:
            gs = row.get("training/global_step")
            if gs is None:
                continue
            length_mean_row = row.get("response_length/mean")
            length_std_row = row.get("response_length/std")
            if length_std_row is None:
                length_std_row = row.get("response_length_non_aborted/std")
            long_row = {
                "policy": policy,
                "family": family,
                "global_step": int(gs),
                "elapsed_train_s": row.get("elapsed_train_s"),
                "elapsed_total_s": row.get("elapsed_total_s"),
                "actor_loss": row.get("actor/loss"),
                "gsm8k_acc": row.get(args.gsm8k_acc_key),
                "math_acc": row.get(args.math_acc_key),
                "response_length_mean": length_mean_row,
                "response_length_std": length_std_row,
            }
            long_rows.append(long_row)
            grouped[(family, int(gs))].append(long_row)

    family_step_rows: list[dict[str, Any]] = []
    for (family, global_step), bucket in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        def collect(metric: str) -> list[float]:
            vals = [float(row[metric]) for row in bucket if isinstance(row.get(metric), (int, float))]
            return vals

        def stat(metric: str) -> tuple[float | None, float | None]:
            vals = collect(metric)
            if not vals:
                return None, None
            return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)

        actor_mean, actor_std = stat("actor_loss")
        gsm_mean, gsm_std = stat("gsm8k_acc")
        math_mean, math_std = stat("math_acc")
        len_mean, len_std = stat("response_length_mean")
        family_step_rows.append(
            {
                "family": family,
                "global_step": global_step,
                "num_runs": len(bucket),
                "actor_loss_mean": actor_mean,
                "actor_loss_std": actor_std,
                "gsm8k_acc_mean": gsm_mean,
                "gsm8k_acc_std": gsm_std,
                "math_acc_mean": math_mean,
                "math_acc_std": math_std,
                "response_length_mean": len_mean,
                "response_length_std_across_runs": len_std,
            }
        )

    progress_dir = args.run_root / "progress"
    write_csv(progress_dir / "latest_status.csv", status_rows)
    write_csv(progress_dir / "step_metrics_long.csv", long_rows)
    write_csv(progress_dir / "family_step_summary.csv", family_step_rows)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] policy status")
    header = (
        "policy".ljust(24)
        + "run ".ljust(5)
        + "step".ljust(8)
        + "loss".ljust(12)
        + "gsm8k".ljust(10)
        + "math".ljust(10)
        + "len(mean±std)"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(status_rows, key=lambda x: str(x["policy"])):
        len_text = f"{fmt(row['response_length_mean'])}±{fmt(row['response_length_std'])}"
        print(
            str(row["policy"]).ljust(24)
            + str(row["running"]).ljust(5)
            + fmt(row["global_step"]).ljust(8)
            + fmt(row["actor_loss"]).ljust(12)
            + fmt(row["gsm8k_acc"]).ljust(10)
            + fmt(row["math_acc"]).ljust(10)
            + len_text
        )
    print(f"\nWrote: {progress_dir / 'latest_status.csv'}")
    print(f"Wrote: {progress_dir / 'step_metrics_long.csv'}")
    print(f"Wrote: {progress_dir / 'family_step_summary.csv'}")


def main() -> None:
    args = parse_args()
    if args.watch_seconds > 0:
        while True:
            summarize(args)
            time.sleep(args.watch_seconds)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
