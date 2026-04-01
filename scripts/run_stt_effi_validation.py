#!/usr/bin/env python3
"""Run the end-to-end stt_effi selector validation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DEFAULT_GSM8K_ACC_KEY = "val-core/openai/gsm8k/acc/mean@1"
DEFAULT_MATH_ACC_KEY = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"


@dataclass
class ArmSpec:
    name: str
    metric: str
    selector: str


@dataclass
class ArmRecord:
    name: str
    metric: str
    selector: str
    selected_train_parquet: str
    run_dir: str
    log_path: str
    metrics_csv: str
    dashboard_svg: str
    status: str
    branch_checkpoint_path: Optional[str] = None
    summary: Optional[Dict[str, object]] = None


DEFAULT_ARMS = [
    ArmSpec(name="gradient_top", metric="gradient_statistical_efficiency", selector="top"),
    ArmSpec(name="gradient_bottom", metric="gradient_statistical_efficiency", selector="bottom"),
    ArmSpec(name="random", metric="gradient_statistical_efficiency", selector="random"),
    ArmSpec(name="dapo_top", metric="dapo_statistical_efficiency", selector="top"),
    ArmSpec(name="pods_top", metric="pods_statistical_efficiency", selector="top"),
]


def default_python_bin() -> Path:
    venv_python = ROOT_DIR / ".venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT_DIR / "runs" / "stt_effi_validation",
        help="Root directory for the validation run.",
    )
    parser.add_argument(
        "--reuse-profile-dir",
        type=Path,
        default=None,
        help="Reuse an existing profile directory instead of rebuilding the offline profile.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="qwen2.5-1.5b-grpo-stt-effi-validation",
        help="Base experiment name used for the launched runs.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=default_python_bin(),
        help="Python executable forwarded to helper scripts and launchers.",
    )
    parser.add_argument(
        "--base-launcher",
        type=Path,
        default=ROOT_DIR / "scripts" / "run_verl_math_1p5b_single_gpu_vllm.sh",
        help="Launcher used for warmup and fresh-selector runs.",
    )
    parser.add_argument(
        "--continuation-launcher",
        type=Path,
        default=ROOT_DIR / "scripts" / "run_grpo_gradient_selector_continuation.sh",
        help="Launcher used for matched continuation runs.",
    )
    parser.add_argument(
        "--train-parquets",
        type=Path,
        nargs="+",
        default=[ROOT_DIR / "data" / "verl" / "gsm8k" / "train.parquet", ROOT_DIR / "data" / "verl" / "math" / "train.parquet"],
        help="Train parquet files for warmup or fresh-selector runs.",
    )
    parser.add_argument(
        "--profile-data-files",
        type=Path,
        nargs="+",
        default=None,
        help="Files used to build the V1/V2 scoring pool. Defaults to --train-parquets.",
    )
    parser.add_argument(
        "--exclude-files",
        type=Path,
        nargs="*",
        default=[ROOT_DIR / "data" / "verl" / "validation_500.parquet"],
        help="Optional parquet files excluded from the profiling pool.",
    )
    parser.add_argument(
        "--start-from-path",
        type=Path,
        default=None,
        help="Optional global_step_* checkpoint to validate from.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Warmup checkpoint step to train to if --start-from-path is not provided.",
    )
    parser.add_argument(
        "--launch-mode",
        choices=["continuation", "fresh"],
        default="continuation",
        help="Whether each arm runs as a matched continuation or as a fresh short run on the selected dataset.",
    )
    parser.add_argument(
        "--arm-training-steps",
        type=int,
        required=True,
        help="Training steps per selector arm after the profile is built.",
    )
    parser.add_argument(
        "--test-freq",
        type=int,
        default=25,
        help="Requested validation frequency. The launcher is adjusted to guarantee a final eval.",
    )
    parser.add_argument(
        "--selector-metric",
        type=str,
        default=None,
        help="Optional single metric to use for all custom --arm specs.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME:METRIC:SELECTOR",
        help="Custom arm spec. If omitted, the default gradient/DAPO/PODS/random arms are used.",
    )
    keep_group = parser.add_mutually_exclusive_group()
    keep_group.add_argument(
        "--keep-ratio",
        type=float,
        default=0.5,
        help="Fraction of the scored pool kept per arm.",
    )
    keep_group.add_argument(
        "--keep-count",
        type=int,
        default=None,
        help="Absolute number of prompts kept per arm.",
    )
    parser.add_argument(
        "--profile-window-multiplier",
        type=float,
        default=None,
        help=(
            "If set, derive both V1 and V2 sizes from "
            "TRAIN_BATCH_SIZE * arm_training_steps * this multiplier."
        ),
    )
    parser.add_argument("--profile-seed", type=int, default=7)
    parser.add_argument("--profile-v1-size", type=int, default=8)
    parser.add_argument("--profile-v2-size", type=int, default=16)
    parser.add_argument("--profile-group-size", type=int, default=5)
    parser.add_argument("--profile-rollout-batch-size", type=int, default=4)
    parser.add_argument("--profile-max-prompt-tokens", type=int, default=512)
    parser.add_argument("--profile-max-new-tokens", type=int, default=0)
    parser.add_argument("--profile-temperature", type=float, default=1.0)
    parser.add_argument("--profile-top-k", type=int, default=0)
    parser.add_argument(
        "--profile-base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model used by the offline profiling script.",
    )
    parser.add_argument(
        "--math-acc-key",
        type=str,
        default=DEFAULT_MATH_ACC_KEY,
        help="Validation metric key used for held-out math accuracy.",
    )
    parser.add_argument(
        "--gsm8k-acc-key",
        type=str,
        default=DEFAULT_GSM8K_ACC_KEY,
        help="Validation metric key used for held-out GSM8K accuracy.",
    )
    parser.add_argument(
        "--launcher-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variables forwarded to the GRPO launchers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    return parser.parse_args()


def parse_env_overrides(items: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Expected KEY=VALUE for --launcher-env.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Launcher env key must not be empty.")
        overrides[key] = value
    return overrides


def configured_train_batch_size(args: argparse.Namespace) -> int:
    overrides = parse_env_overrides(args.launcher_env)
    raw = overrides.get("TRAIN_BATCH_SIZE", os.environ.get("TRAIN_BATCH_SIZE", "8"))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid TRAIN_BATCH_SIZE value for sizing: {raw}") from exc
    if value <= 0:
        raise ValueError("TRAIN_BATCH_SIZE must be positive.")
    return value


def effective_profile_sizes(args: argparse.Namespace) -> tuple[int, int]:
    if args.profile_window_multiplier is None:
        return args.profile_v1_size, args.profile_v2_size
    trained_prompts = configured_train_batch_size(args) * args.arm_training_steps
    derived_size = max(1, math.ceil(trained_prompts * args.profile_window_multiplier))
    return derived_size, derived_size


def parse_arm_specs(raw_specs: List[str], selector_metric: Optional[str]) -> List[ArmSpec]:
    if not raw_specs:
        return list(DEFAULT_ARMS)
    arms: List[ArmSpec] = []
    for raw in raw_specs:
        parts = raw.split(":")
        if len(parts) == 3:
            name, metric, selector = parts
        elif len(parts) == 2 and selector_metric is not None:
            name, selector = parts
            metric = selector_metric
        else:
            raise ValueError(f"Invalid --arm spec '{raw}'. Use NAME:METRIC:SELECTOR.")
        selector = selector.strip().lower()
        if selector not in {"top", "bottom", "random"}:
            raise ValueError(f"Unsupported selector '{selector}' in arm '{raw}'.")
        arms.append(ArmSpec(name=name.strip(), metric=metric.strip(), selector=selector))
    return arms


def ensure_required_paths(args: argparse.Namespace) -> None:
    for path in args.train_parquets:
        if not path.exists():
            raise FileNotFoundError(f"Missing train parquet: {path}")
    for path in args.profile_data_files or args.train_parquets:
        if not path.exists():
            raise FileNotFoundError(f"Missing profile data file: {path}")
    for path in [args.base_launcher, args.continuation_launcher, args.python_bin]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required executable: {path}")
    if args.reuse_profile_dir is not None:
        if not args.reuse_profile_dir.exists():
            raise FileNotFoundError(f"Missing reuse profile dir: {args.reuse_profile_dir}")
        if not (args.reuse_profile_dir / "ground_truth_profile.csv").exists():
            raise FileNotFoundError(
                f"Missing ground_truth_profile.csv in reuse profile dir: {args.reuse_profile_dir}"
            )
    if args.start_from_path is not None and not args.start_from_path.exists():
        raise FileNotFoundError(f"Missing start checkpoint: {args.start_from_path}")
    if args.start_from_path is None and args.checkpoint_step is None:
        raise ValueError("Set either --start-from-path or --checkpoint-step.")
    if args.checkpoint_step is not None and args.checkpoint_step <= 0:
        raise ValueError("--checkpoint-step must be positive.")
    if args.arm_training_steps <= 0:
        raise ValueError("--arm-training-steps must be positive.")


def parse_checkpoint_step(step_dir: Path) -> int:
    name = step_dir.name
    if not name.startswith("global_step_"):
        raise ValueError(f"Expected a global_step_* directory, got {step_dir}")
    return int(name.split("global_step_", 1)[1])


def list_literal(paths: List[Path]) -> str:
    return str([str(path) for path in paths])


def resolve_profile_dir(args: argparse.Namespace) -> Path:
    return args.reuse_profile_dir if args.reuse_profile_dir is not None else args.run_root / "profile"


def make_base_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(args.python_bin)
    env.update(parse_env_overrides(args.launcher_env))
    return env


def effective_test_freq(total_steps: int, requested: int) -> int:
    if requested <= 0 or requested > total_steps or total_steps % requested != 0:
        return total_steps
    return requested


def run_command(command: List[str], *, env: Dict[str, str], log_path: Path, dry_run: bool) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] {rendered}")
        print(f"[dry-run] log -> {log_path}")
        return
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {rendered}\n\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def launch_warmup_run(args: argparse.Namespace, checkpoint_step: int) -> Path:
    run_dir = args.run_root / "warmup"
    env = make_base_env(args)
    env["TRAIN_FILES"] = list_literal(args.train_parquets)
    env["RUN_DIR"] = str(run_dir)
    env["EXPERIMENT_NAME"] = f"{args.experiment_name}-warmup"
    env["TOTAL_TRAINING_STEPS"] = str(checkpoint_step)
    env["SAVE_FREQ"] = str(checkpoint_step)
    env["TEST_FREQ"] = str(effective_test_freq(checkpoint_step, args.test_freq))
    run_command(
        ["bash", str(args.base_launcher)],
        env=env,
        log_path=run_dir / "launcher.log",
        dry_run=args.dry_run,
    )
    return run_dir / f"global_step_{checkpoint_step}"


def profile_checkpoint(args: argparse.Namespace, checkpoint_step_dir: Path) -> Path:
    checkpoint_step = parse_checkpoint_step(checkpoint_step_dir)
    profile_dir = resolve_profile_dir(args)
    profile_v1_size, profile_v2_size = effective_profile_sizes(args)
    command = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "profile_grpo_ground_truth.py"),
        "--checkpoint-dir",
        str(checkpoint_step_dir.parent),
        "--checkpoint-steps",
        str(checkpoint_step),
        "--output-dir",
        str(profile_dir),
        "--seed",
        str(args.profile_seed),
        "--v1-size",
        str(profile_v1_size),
        "--v2-size",
        str(profile_v2_size),
        "--group-size",
        str(args.profile_group_size),
        "--rollout-batch-size",
        str(args.profile_rollout_batch_size),
        "--max-prompt-tokens",
        str(args.profile_max_prompt_tokens),
        "--max-new-tokens",
        str(args.profile_max_new_tokens),
        "--temperature",
        str(args.profile_temperature),
        "--top-k",
        str(args.profile_top_k),
        "--base-model",
        args.profile_base_model,
        "--data-files",
        *[str(path) for path in (args.profile_data_files or args.train_parquets)],
    ]
    if args.exclude_files:
        command.extend(["--exclude-files", *[str(path) for path in args.exclude_files]])
    run_command(
        command,
        env=make_base_env(args),
        log_path=args.run_root / "profile.log",
        dry_run=args.dry_run,
    )
    return profile_dir


def build_selected_parquet(
    args: argparse.Namespace,
    checkpoint_step: int,
    arm: ArmSpec,
) -> Path:
    output_dir = args.run_root / "selector_pools" / arm.name
    output_parquet = output_dir / "selected_train.parquet"
    selection_csv = output_dir / "selected_rows.csv"
    command = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "build_profile_selector_dataset.py"),
        "--profile-csv",
        str(resolve_profile_dir(args) / "ground_truth_profile.csv"),
        "--checkpoint-step",
        str(checkpoint_step),
        "--metric",
        arm.metric,
        "--selector",
        arm.selector,
        "--seed",
        str(args.profile_seed),
        "--source-parquets",
        *[str(path) for path in (args.profile_data_files or args.train_parquets)],
        "--output-parquet",
        str(output_parquet),
        "--output-selection-csv",
        str(selection_csv),
    ]
    if args.keep_count is not None:
        command.extend(["--keep-count", str(args.keep_count)])
    else:
        command.extend(["--keep-ratio", str(args.keep_ratio)])
    run_command(
        command,
        env=make_base_env(args),
        log_path=output_dir / "build.log",
        dry_run=args.dry_run,
    )
    return output_parquet


def branch_checkpoint(args: argparse.Namespace, source_step_dir: Path, output_step_dir: Path) -> None:
    command = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "prepare_resume_checkpoint.py"),
        "--source-step-dir",
        str(source_step_dir),
        "--output-step-dir",
        str(output_step_dir),
    ]
    run_command(
        command,
        env=make_base_env(args),
        log_path=output_step_dir.parent / "branch.log",
        dry_run=args.dry_run,
    )


def launch_arm_run(
    args: argparse.Namespace,
    arm: ArmSpec,
    selected_train_parquet: Path,
    start_checkpoint: Path,
    start_step: int,
) -> ArmRecord:
    run_dir = args.run_root / "arms" / arm.name
    env = make_base_env(args)
    env["RUN_DIR"] = str(run_dir)
    env["EXPERIMENT_NAME"] = f"{args.experiment_name}-{arm.name}"
    env["TEST_FREQ"] = str(effective_test_freq(args.arm_training_steps, args.test_freq))

    if args.launch_mode == "continuation":
        branch_path = run_dir / f"global_step_{start_step}"
        branch_checkpoint(args, start_checkpoint, branch_path)
        env["TRAIN_PARQUET"] = str(selected_train_parquet)
        env["RESUME_FROM_PATH"] = str(branch_path)
        env["EXTRA_TRAINING_STEPS"] = str(args.arm_training_steps)
        env["SAVE_FREQ"] = str(args.arm_training_steps)
        log_path = run_dir / "launcher.log"
        run_command(
            ["bash", str(args.continuation_launcher)],
            env=env,
            log_path=log_path,
            dry_run=args.dry_run,
        )
        status = "planned" if args.dry_run else "completed"
        return ArmRecord(
            name=arm.name,
            metric=arm.metric,
            selector=arm.selector,
            selected_train_parquet=str(selected_train_parquet),
            run_dir=str(run_dir),
            log_path=str(log_path),
            metrics_csv=str(run_dir / "training_metrics.csv"),
            dashboard_svg=str(run_dir / "training_dashboard.svg"),
            status=status,
            branch_checkpoint_path=str(branch_path),
        )

    env["TRAIN_FILES"] = list_literal([selected_train_parquet])
    env["TOTAL_TRAINING_STEPS"] = str(args.arm_training_steps)
    env["SAVE_FREQ"] = str(args.arm_training_steps)
    log_path = run_dir / "launcher.log"
    run_command(
        ["bash", str(args.base_launcher)],
        env=env,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    status = "planned" if args.dry_run else "completed"
    return ArmRecord(
        name=arm.name,
        metric=arm.metric,
        selector=arm.selector,
        selected_train_parquet=str(selected_train_parquet),
        run_dir=str(run_dir),
        log_path=str(log_path),
        metrics_csv=str(run_dir / "training_metrics.csv"),
        dashboard_svg=str(run_dir / "training_dashboard.svg"),
        status=status,
    )


def extract_metrics_csv(args: argparse.Namespace, arm_record: ArmRecord) -> None:
    command = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "extract_verl_step_metrics.py"),
        "--input-log",
        arm_record.log_path,
        "--output-csv",
        arm_record.metrics_csv,
    ]
    run_command(
        command,
        env=make_base_env(args),
        log_path=Path(arm_record.run_dir) / "extract_metrics.log",
        dry_run=args.dry_run,
    )


def render_dashboard(args: argparse.Namespace, arm_record: ArmRecord) -> None:
    command = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "plot_training_metrics_svg.py"),
        "--input-csv",
        arm_record.metrics_csv,
        "--output-svg",
        arm_record.dashboard_svg,
    ]
    run_command(
        command,
        env=make_base_env(args),
        log_path=Path(arm_record.run_dir) / "plot_metrics.log",
        dry_run=args.dry_run,
    )


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


def parse_step_rows(log_path: Path) -> List[dict]:
    rows: List[dict] = []
    elapsed_train = 0.0
    elapsed_total = 0.0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            if "step:" not in line:
                continue
            payload = line[line.index("step:") :].strip()
            row: Dict[str, object] = {}
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


def pick_last_non_null(rows: List[dict], key: str):
    for row in reversed(rows):
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def pick_best_non_null(rows: List[dict], key: str):
    best_value = None
    best_step = None
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        value = float(value)
        if best_value is None or value > best_value:
            best_value = value
            best_step = row.get("training/global_step", row.get("step"))
    return best_value, best_step


def summarize_arm(log_path: Path, math_acc_key: str, gsm8k_acc_key: str) -> dict:
    rows = parse_step_rows(log_path)
    if not rows:
        raise ValueError(f"No step rows found in {log_path}")

    final_row = rows[-1]
    best_math, best_math_step = pick_best_non_null(rows, math_acc_key)
    best_gsm8k, best_gsm8k_step = pick_best_non_null(rows, gsm8k_acc_key)
    return {
        "final_global_step": final_row.get("training/global_step", final_row.get("step")),
        "final_elapsed_train_s": final_row.get("elapsed_train_s"),
        "final_elapsed_total_s": final_row.get("elapsed_total_s"),
        "final_step_time_s": final_row.get("timing_s/step"),
        "final_throughput": final_row.get("perf/throughput"),
        "final_math_acc_at1": pick_last_non_null(rows, math_acc_key),
        "final_gsm8k_acc_at1": pick_last_non_null(rows, gsm8k_acc_key),
        "best_math_acc_at1": best_math,
        "best_math_acc_step": best_math_step,
        "best_gsm8k_acc_at1": best_gsm8k,
        "best_gsm8k_acc_step": best_gsm8k_step,
        "final_actor_loss": final_row.get("actor/loss"),
        "final_train_reward_mean": final_row.get("critic/rewards/mean"),
        "final_response_length_mean": final_row.get("response_length/mean"),
    }


def write_summary(run_root: Path, arm_records: List[ArmRecord]) -> None:
    summary_rows: List[dict] = []
    random_math = None
    random_gsm8k = None
    for arm_record in arm_records:
        row = {
            "name": arm_record.name,
            "metric": arm_record.metric,
            "selector": arm_record.selector,
            "status": arm_record.status,
            "run_dir": arm_record.run_dir,
            "selected_train_parquet": arm_record.selected_train_parquet,
        }
        if arm_record.summary is not None:
            row.update(arm_record.summary)
        summary_rows.append(row)
        if arm_record.name == "random" and arm_record.summary is not None:
            random_math = arm_record.summary.get("final_math_acc_at1")
            random_gsm8k = arm_record.summary.get("final_gsm8k_acc_at1")

    for row in summary_rows:
        math_acc = row.get("final_math_acc_at1")
        gsm8k_acc = row.get("final_gsm8k_acc_at1")
        row["delta_vs_random_math_acc_at1"] = (
            None if math_acc is None or random_math is None else float(math_acc) - float(random_math)
        )
        row["delta_vs_random_gsm8k_acc_at1"] = (
            None if gsm8k_acc is None or random_gsm8k is None else float(gsm8k_acc) - float(random_gsm8k)
        )

    fieldnames = [
        "name",
        "metric",
        "selector",
        "status",
        "final_global_step",
        "final_elapsed_train_s",
        "final_elapsed_total_s",
        "final_step_time_s",
        "final_throughput",
        "final_math_acc_at1",
        "final_gsm8k_acc_at1",
        "best_math_acc_at1",
        "best_math_acc_step",
        "best_gsm8k_acc_at1",
        "best_gsm8k_acc_step",
        "final_actor_loss",
        "final_train_reward_mean",
        "final_response_length_mean",
        "delta_vs_random_math_acc_at1",
        "delta_vs_random_gsm8k_acc_at1",
        "run_dir",
        "selected_train_parquet",
    ]
    summary_csv = run_root / "validation_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(run_root / "validation_summary.json", summary_rows)


def main() -> None:
    args = parse_args()
    ensure_required_paths(args)
    arms = parse_arm_specs(args.arm, args.selector_metric)
    args.run_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_name": args.experiment_name,
        "launch_mode": args.launch_mode,
        "arm_training_steps": args.arm_training_steps,
        "keep_ratio": args.keep_ratio,
        "keep_count": args.keep_count,
        "reuse_profile_dir": None if args.reuse_profile_dir is None else str(args.reuse_profile_dir),
        "profile_window_multiplier": args.profile_window_multiplier,
        "profile_data_files": [str(path) for path in (args.profile_data_files or args.train_parquets)],
        "exclude_files": [str(path) for path in args.exclude_files],
        "arms": [asdict(arm) for arm in arms],
    }
    profile_v1_size, profile_v2_size = effective_profile_sizes(args)
    manifest["effective_profile_v1_size"] = profile_v1_size
    manifest["effective_profile_v2_size"] = profile_v2_size

    if args.start_from_path is not None:
        start_checkpoint = args.start_from_path
    else:
        assert args.checkpoint_step is not None
        start_checkpoint = launch_warmup_run(args, args.checkpoint_step)
    start_step = parse_checkpoint_step(start_checkpoint)
    manifest["start_checkpoint"] = str(start_checkpoint)
    manifest["start_step"] = start_step

    if args.reuse_profile_dir is not None:
        profile_dir = args.reuse_profile_dir
    else:
        profile_dir = profile_checkpoint(args, start_checkpoint)
    manifest["profile_dir"] = str(profile_dir)

    arm_records: List[ArmRecord] = []
    for arm in arms:
        selected_train_parquet = args.run_root / "selector_pools" / arm.name / "selected_train.parquet"
        run_dir = args.run_root / "arms" / arm.name
        branch_path = run_dir / f"global_step_{start_step}" if args.launch_mode == "continuation" else None
        try:
            selected_train_parquet = build_selected_parquet(args, start_step, arm)
            arm_record = launch_arm_run(args, arm, selected_train_parquet, start_checkpoint, start_step)
        except subprocess.CalledProcessError as exc:
            arm_record = ArmRecord(
                name=arm.name,
                metric=arm.metric,
                selector=arm.selector,
                selected_train_parquet=str(selected_train_parquet),
                run_dir=str(run_dir),
                log_path=str(run_dir / "launcher.log"),
                metrics_csv=str(run_dir / "training_metrics.csv"),
                dashboard_svg=str(run_dir / "training_dashboard.svg"),
                status=f"failed_exit_{exc.returncode}",
                branch_checkpoint_path=None if branch_path is None else str(branch_path),
            )
        arm_records.append(arm_record)
        write_json(
            args.run_root / "validation_manifest.json",
            {"config": manifest, "arm_records": [asdict(record) for record in arm_records]},
        )

    if args.dry_run:
        return

    for arm_record in arm_records:
        log_path = Path(arm_record.log_path)
        if not log_path.exists():
            continue
        try:
            extract_metrics_csv(args, arm_record)
            render_dashboard(args, arm_record)
            arm_record.summary = summarize_arm(log_path, args.math_acc_key, args.gsm8k_acc_key)
        except Exception:
            if arm_record.status == "completed":
                arm_record.status = "postprocess_failed"

    write_summary(args.run_root, arm_records)
    write_json(args.run_root / "validation_manifest.json", {"config": manifest, "arm_records": [asdict(record) for record in arm_records]})


if __name__ == "__main__":
    main()
