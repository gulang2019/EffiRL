#!/usr/bin/env python3
"""Run GRPO with periodic gradient-selector refresh windows."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class WindowRecord:
    kind: str
    index: int
    start_step: int
    end_step: int
    run_dir: str
    checkpoint_path: str
    log_path: str
    profile_dir: Optional[str] = None
    selected_train_parquet: Optional[str] = None
    branch_checkpoint_path: Optional[str] = None


def default_python_bin() -> Path:
    venv_python = ROOT_DIR / ".venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT_DIR / "runs" / "periodic_gradient_selector",
        help="Root directory for all window runs and artifacts.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="qwen2.5-1.5b-grpo-periodic-gradient-selector",
        help="Base experiment name used for per-window runs.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=default_python_bin(),
        help="Python executable passed into the shell launchers and helper scripts.",
    )
    parser.add_argument(
        "--base-launcher",
        type=Path,
        default=ROOT_DIR / "scripts" / "run_verl_math_1p5b_single_gpu_vllm.sh",
        help="Launcher used for the first scratch window.",
    )
    parser.add_argument(
        "--continuation-launcher",
        type=Path,
        default=ROOT_DIR / "scripts" / "run_grpo_gradient_selector_continuation.sh",
        help="Launcher used for resumed continuation windows.",
    )
    parser.add_argument(
        "--train-parquets",
        type=Path,
        nargs="+",
        default=[ROOT_DIR / "data" / "verl" / "gsm8k" / "train.parquet", ROOT_DIR / "data" / "verl" / "math" / "train.parquet"],
        help="Training parquet files for the scratch window. These are also the default selector source files.",
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
        "--window-steps",
        type=int,
        required=True,
        help="Training steps per refresh window.",
    )
    parser.add_argument(
        "--total-training-steps",
        type=int,
        required=True,
        help="Final global step to reach across all windows.",
    )
    parser.add_argument(
        "--test-freq",
        type=int,
        default=25,
        help="Validation frequency passed to the GRPO launchers.",
    )
    parser.add_argument(
        "--start-from-path",
        type=Path,
        default=None,
        help="Optional global_step_* directory to start resuming from instead of training the first window from scratch.",
    )
    parser.add_argument(
        "--selector-metric",
        type=str,
        default="gradient_statistical_efficiency",
        help="Score column used for selection.",
    )
    parser.add_argument(
        "--selector",
        choices=["top", "bottom", "random"],
        default="top",
        help="Selection rule applied after scoring.",
    )
    keep_group = parser.add_mutually_exclusive_group()
    keep_group.add_argument(
        "--keep-ratio",
        type=float,
        default=0.5,
        help="Fraction of the scored pool kept after each refresh.",
    )
    keep_group.add_argument(
        "--keep-count",
        type=int,
        default=None,
        help="Absolute number of examples kept after each refresh.",
    )
    parser.add_argument(
        "--profile-window-multiplier",
        type=float,
        default=None,
        help=(
            "If set, derive both V1 and V2 sizes from "
            "TRAIN_BATCH_SIZE * window_steps * this multiplier."
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
        help="Base model path used by the offline profiling script.",
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
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def parse_env_overrides(items: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty env var name in: {item}")
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
    trained_prompts = configured_train_batch_size(args) * args.window_steps
    derived_size = max(1, math.ceil(trained_prompts * args.profile_window_multiplier))
    return derived_size, derived_size


def parse_checkpoint_step(step_dir: Path) -> int:
    name = step_dir.name
    if not name.startswith("global_step_"):
        raise ValueError(f"Expected a global_step_* directory, got {step_dir}")
    try:
        return int(name.split("global_step_", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Failed to parse checkpoint step from {step_dir}") from exc


def list_literal(paths: List[Path]) -> str:
    return str([str(path) for path in paths])


def ensure_required_paths(args: argparse.Namespace) -> None:
    for path in args.train_parquets:
        if not path.exists():
            raise FileNotFoundError(f"Missing train parquet: {path}")
    profile_data_files = args.profile_data_files or args.train_parquets
    for path in profile_data_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing profile data file: {path}")
    for path in [args.base_launcher, args.continuation_launcher, args.python_bin]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required executable: {path}")
    if args.start_from_path is not None and not args.start_from_path.exists():
        raise FileNotFoundError(f"Missing start checkpoint: {args.start_from_path}")
    if args.window_steps <= 0:
        raise ValueError("--window-steps must be positive.")
    if args.total_training_steps <= 0:
        raise ValueError("--total-training-steps must be positive.")


def run_command(
    command: List[str],
    *,
    env: Dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
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


def write_manifest(path: Path, args: argparse.Namespace, windows: List[WindowRecord]) -> None:
    profile_v1_size, profile_v2_size = effective_profile_sizes(args)
    payload = {
        "experiment_name": args.experiment_name,
        "run_root": str(args.run_root),
        "window_steps": args.window_steps,
        "total_training_steps": args.total_training_steps,
        "start_from_path": str(args.start_from_path) if args.start_from_path is not None else None,
        "selector_metric": args.selector_metric,
        "selector": args.selector,
        "keep_ratio": args.keep_ratio,
        "keep_count": args.keep_count,
        "profile_window_multiplier": args.profile_window_multiplier,
        "effective_profile_v1_size": profile_v1_size,
        "effective_profile_v2_size": profile_v2_size,
        "train_parquets": [str(path) for path in args.train_parquets],
        "profile_data_files": [str(path) for path in (args.profile_data_files or args.train_parquets)],
        "exclude_files": [str(path) for path in args.exclude_files],
        "windows": [asdict(window) for window in windows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_base_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(args.python_bin)
    env.update(parse_env_overrides(args.launcher_env))
    return env


def profile_checkpoint(
    args: argparse.Namespace,
    checkpoint_step_dir: Path,
    artifact_dir: Path,
) -> Path:
    checkpoint_step = parse_checkpoint_step(checkpoint_step_dir)
    profile_dir = artifact_dir / "profile"
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
    ]
    profile_data_files = args.profile_data_files or args.train_parquets
    command.extend(["--data-files", *[str(path) for path in profile_data_files]])
    if args.exclude_files:
        command.extend(["--exclude-files", *[str(path) for path in args.exclude_files]])
    run_command(
        command,
        env=make_base_env(args),
        log_path=artifact_dir / "profile.log",
        dry_run=args.dry_run,
    )
    return profile_dir


def build_selected_parquet(
    args: argparse.Namespace,
    checkpoint_step_dir: Path,
    profile_dir: Path,
    artifact_dir: Path,
) -> Path:
    checkpoint_step = parse_checkpoint_step(checkpoint_step_dir)
    output_parquet = artifact_dir / "selected_train.parquet"
    selection_csv = artifact_dir / "selected_rows.csv"
    command = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "build_profile_selector_dataset.py"),
        "--profile-csv",
        str(profile_dir / "ground_truth_profile.csv"),
        "--checkpoint-step",
        str(checkpoint_step),
        "--metric",
        args.selector_metric,
        "--selector",
        args.selector,
        "--output-parquet",
        str(output_parquet),
        "--output-selection-csv",
        str(selection_csv),
        "--seed",
        str(args.profile_seed),
    ]
    command.extend(["--source-parquets", *[str(path) for path in (args.profile_data_files or args.train_parquets)]])
    if args.keep_count is not None:
        command.extend(["--keep-count", str(args.keep_count)])
    else:
        command.extend(["--keep-ratio", str(args.keep_ratio)])
    run_command(
        command,
        env=make_base_env(args),
        log_path=artifact_dir / "select.log",
        dry_run=args.dry_run,
    )
    return output_parquet


def branch_checkpoint(
    args: argparse.Namespace,
    source_step_dir: Path,
    destination_step_dir: Path,
    artifact_dir: Path,
) -> None:
    command = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "prepare_resume_checkpoint.py"),
        "--source-step-dir",
        str(source_step_dir),
        "--output-step-dir",
        str(destination_step_dir),
    ]
    run_command(
        command,
        env=make_base_env(args),
        log_path=artifact_dir / "branch.log",
        dry_run=args.dry_run,
    )


def expected_checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"global_step_{step}"


def launch_scratch_window(
    args: argparse.Namespace,
    *,
    window_index: int,
    end_step: int,
) -> WindowRecord:
    run_dir = args.run_root / f"window_{window_index:03d}_scratch_to_{end_step}"
    env = make_base_env(args)
    env["TRAIN_FILES"] = list_literal(args.train_parquets)
    env["RUN_DIR"] = str(run_dir)
    env["EXPERIMENT_NAME"] = f"{args.experiment_name}-window-{window_index:03d}"
    env["TOTAL_TRAINING_STEPS"] = str(end_step)
    env["SAVE_FREQ"] = str(end_step)
    env["TEST_FREQ"] = str(args.test_freq)
    log_path = run_dir / "launcher.log"
    run_command(
        ["bash", str(args.base_launcher)],
        env=env,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    checkpoint_path = expected_checkpoint_path(run_dir, end_step)
    return WindowRecord(
        kind="scratch",
        index=window_index,
        start_step=0,
        end_step=end_step,
        run_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        log_path=str(log_path),
    )


def launch_continuation_window(
    args: argparse.Namespace,
    *,
    window_index: int,
    start_checkpoint: Path,
    start_step: int,
    end_step: int,
    selected_train_parquet: Path,
    profile_dir: Path,
) -> WindowRecord:
    run_dir = args.run_root / f"window_{window_index:03d}_resume_{start_step}_to_{end_step}"
    branch_checkpoint_path = expected_checkpoint_path(run_dir, start_step)
    artifact_dir = run_dir / "artifacts"
    branch_checkpoint(args, start_checkpoint, branch_checkpoint_path, artifact_dir)

    env = make_base_env(args)
    env["TRAIN_PARQUET"] = str(selected_train_parquet)
    env["RESUME_FROM_PATH"] = str(branch_checkpoint_path)
    env["RUN_DIR"] = str(run_dir)
    env["EXPERIMENT_NAME"] = f"{args.experiment_name}-window-{window_index:03d}"
    env["EXTRA_TRAINING_STEPS"] = str(end_step - start_step)
    env["SAVE_FREQ"] = str(end_step - start_step)
    env["TEST_FREQ"] = str(args.test_freq)
    log_path = run_dir / "launcher.log"
    run_command(
        ["bash", str(args.continuation_launcher)],
        env=env,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    checkpoint_path = expected_checkpoint_path(run_dir, end_step)
    return WindowRecord(
        kind="continuation",
        index=window_index,
        start_step=start_step,
        end_step=end_step,
        run_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        log_path=str(log_path),
        profile_dir=str(profile_dir),
        selected_train_parquet=str(selected_train_parquet),
        branch_checkpoint_path=str(branch_checkpoint_path),
    )


def main() -> None:
    args = parse_args()
    ensure_required_paths(args)
    args.run_root.mkdir(parents=True, exist_ok=True)

    windows: List[WindowRecord] = []
    manifest_path = args.run_root / "periodic_gradient_selector_manifest.json"

    if args.start_from_path is None:
        first_end = min(args.window_steps, args.total_training_steps)
        scratch_record = launch_scratch_window(args, window_index=0, end_step=first_end)
        windows.append(scratch_record)
        write_manifest(manifest_path, args, windows)
        current_checkpoint = Path(scratch_record.checkpoint_path)
        current_step = first_end
        next_window_index = 1
    else:
        current_checkpoint = args.start_from_path
        current_step = parse_checkpoint_step(current_checkpoint)
        next_window_index = 0

    while current_step < args.total_training_steps:
        artifact_dir = args.run_root / f"refresh_step_{current_step}"
        profile_dir = profile_checkpoint(args, current_checkpoint, artifact_dir)
        selected_train_parquet = build_selected_parquet(args, current_checkpoint, profile_dir, artifact_dir)
        next_step = min(args.total_training_steps, current_step + args.window_steps)
        record = launch_continuation_window(
            args,
            window_index=next_window_index,
            start_checkpoint=current_checkpoint,
            start_step=current_step,
            end_step=next_step,
            selected_train_parquet=selected_train_parquet,
            profile_dir=profile_dir,
        )
        windows.append(record)
        write_manifest(manifest_path, args, windows)
        current_checkpoint = Path(record.checkpoint_path)
        current_step = next_step
        next_window_index += 1


if __name__ == "__main__":
    main()
