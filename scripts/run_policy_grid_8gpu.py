#!/usr/bin/env python3
"""Launch one policy per GPU plus gradient-policy grid variants."""

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
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class PolicySpec:
    name: str
    family: str
    metric: str
    selector: str
    keep_ratio: float | None = None
    keep_count: int | None = None


def default_python_bin() -> Path:
    venv_python = ROOT_DIR / ".venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["continuation", "scratch"],
        default="continuation",
        help="continuation: branch from --start-from-path; scratch: run warmup to --checkpoint-step first.",
    )
    parser.add_argument("--start-from-path", type=Path, default=None, help="global_step_* checkpoint directory.")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Required in scratch mode. Warmup run trains to this step before policy branching.",
    )
    parser.add_argument("--arm-training-steps", type=int, default=100)
    parser.add_argument(
        "--policy-launch-mode",
        choices=["continuation", "fresh"],
        default="continuation",
        help="How policy arms are launched after selector dataset build.",
    )
    parser.add_argument(
        "--gradient-update-mode",
        choices=["static", "periodic"],
        default="static",
        help="static: one-shot selector pool; periodic: refresh gradient direction every window for gradient families.",
    )
    parser.add_argument("--window-steps", type=int, default=5)
    parser.add_argument("--test-freq", type=int, default=25)
    parser.add_argument("--experiment-name", type=str, default="qwen2.5-1.5b-grpo-policy-grid")
    parser.add_argument("--python-bin", type=Path, default=default_python_bin())
    parser.add_argument(
        "--base-launcher",
        type=Path,
        default=ROOT_DIR / "scripts" / "run_verl_math_1p5b_single_gpu_vllm.sh",
    )
    parser.add_argument(
        "--continuation-launcher",
        type=Path,
        default=ROOT_DIR / "scripts" / "run_grpo_gradient_selector_continuation.sh",
    )
    parser.add_argument(
        "--train-parquets",
        type=Path,
        nargs="+",
        default=[ROOT_DIR / "data" / "verl" / "gsm8k" / "train.parquet", ROOT_DIR / "data" / "verl" / "math" / "train.parquet"],
    )
    parser.add_argument(
        "--exclude-files",
        type=Path,
        nargs="*",
        default=[ROOT_DIR / "data" / "verl" / "validation_500.parquet"],
    )
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--profile-seed", type=int, default=7)
    parser.add_argument("--profile-window-multiplier", type=float, default=4.0)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--profile-group-size", type=int, default=5)
    parser.add_argument("--profile-rollout-batch-size", type=int, default=2)
    parser.add_argument("--profile-max-prompt-tokens", type=int, default=512)
    parser.add_argument("--profile-max-new-tokens", type=int, default=4096)
    parser.add_argument("--profile-temperature", type=float, default=1.0)
    parser.add_argument("--profile-top-k", type=int, default=0)
    parser.add_argument("--profile-base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--warmup-gpu-id", type=str, default=None, help="GPU id used for scratch warmup run.")
    parser.add_argument("--include-baselines", action="store_true", default=True)
    parser.add_argument(
        "--gradient-keep-ratios",
        type=str,
        default="0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated keep ratios for gradient-top grid.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="NAME:FAMILY:METRIC:SELECTOR:KEEP_RATIO",
        help="Optional custom policy spec (repeatable).",
    )
    parser.add_argument(
        "--launcher-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra env vars forwarded to continuation launcher.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_env_overrides(items: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid launcher-env entry: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid launcher-env key: {item}")
        overrides[key] = value
    return overrides


def parse_checkpoint_step(path: Path) -> int:
    if not path.name.startswith("global_step_"):
        raise ValueError(f"--start-from-path must point to global_step_* dir, got {path}")
    return int(path.name.split("global_step_", 1)[1])


def effective_profile_sizes(train_batch_size: int, window_steps: int, multiplier: float) -> tuple[int, int]:
    derived = max(1, math.ceil(train_batch_size * window_steps * multiplier))
    return derived, derived


def run_cmd(command: list[str], *, env: Dict[str, str], log_path: Path | None, dry_run: bool) -> None:
    rendered = " ".join(shlex.quote(x) for x in command)
    if dry_run:
        print(f"[dry-run] {rendered}")
        if log_path is not None:
            print(f"[dry-run] log -> {log_path}")
        return
    if log_path is None:
        subprocess.run(command, cwd=ROOT_DIR, env=env, check=True, text=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {rendered}\n\n")
        handle.flush()
        try:
            subprocess.run(command, cwd=ROOT_DIR, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as exc:
            tail_text = ""
            try:
                tail_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                tail_text = "\n".join(tail_lines)
            except Exception:
                tail_text = "<failed to read log tail>"
            raise RuntimeError(
                f"Command failed with exit code {exc.returncode}\n"
                f"log_path: {log_path}\n"
                f"command: {rendered}\n"
                f"--- log tail ---\n{tail_text}"
            ) from exc


def effective_test_freq(total_steps: int, requested: int) -> int:
    if requested <= 0 or requested > total_steps or total_steps % requested != 0:
        return total_steps
    return requested


def build_policy_specs(args: argparse.Namespace) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    if args.include_baselines:
        specs.extend(
            [
                PolicySpec("dapo_top", "dapo", "dapo_statistical_efficiency", "top", keep_ratio=0.5),
                PolicySpec("pods_top", "pods", "pods_statistical_efficiency", "top", keep_ratio=0.5),
                PolicySpec("random", "random", "gradient_statistical_efficiency", "random", keep_ratio=0.5),
            ]
        )
    ratios = [x.strip() for x in args.gradient_keep_ratios.split(",") if x.strip()]
    for ratio_raw in ratios:
        ratio = float(ratio_raw)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"Invalid gradient keep ratio: {ratio}")
        ratio_tag = str(int(round(ratio * 100))).zfill(3)
        specs.append(
            PolicySpec(
                name=f"gradient_top_r{ratio_tag}",
                family="gradient_grid",
                metric="gradient_statistical_efficiency",
                selector="top",
                keep_ratio=ratio,
            )
        )
    for raw in args.policy:
        parts = raw.split(":")
        if len(parts) != 5:
            raise ValueError(f"Invalid --policy spec: {raw}")
        name, family, metric, selector, keep_ratio_raw = [x.strip() for x in parts]
        keep_ratio = float(keep_ratio_raw)
        specs.append(
            PolicySpec(
                name=name,
                family=family,
                metric=metric,
                selector=selector,
                keep_ratio=keep_ratio,
            )
        )
    # keep insertion order, remove duplicates by name (last one wins)
    dedup: dict[str, PolicySpec] = {}
    for spec in specs:
        dedup[spec.name] = spec
    return list(dedup.values())


def launch_job(
    *,
    args: argparse.Namespace,
    gpu_id: str,
    policy: PolicySpec,
    start_checkpoint: Path,
    start_step: int,
    selected_train_parquet: Path,
    run_dir: Path,
) -> int:
    branch_path = run_dir / f"global_step_{start_step}"
    branch_cmd = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "prepare_resume_checkpoint.py"),
        "--source-step-dir",
        str(start_checkpoint),
        "--output-step-dir",
        str(branch_path),
    ]
    run_cmd(
        branch_cmd,
        env=os.environ.copy(),
        log_path=run_dir / "branch.log",
        dry_run=args.dry_run,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHON_BIN"] = str(args.python_bin)
    env["TRAIN_PARQUET"] = str(selected_train_parquet)
    env["RESUME_FROM_PATH"] = str(branch_path)
    env["RUN_DIR"] = str(run_dir)
    env["EXPERIMENT_NAME"] = f"{args.experiment_name}-{policy.name}"
    env["EXTRA_TRAINING_STEPS"] = str(args.arm_training_steps)
    env["SAVE_FREQ"] = str(args.arm_training_steps)
    env["TEST_FREQ"] = str(args.test_freq if 0 < args.test_freq <= args.arm_training_steps else args.arm_training_steps)
    env["TRAINER_LOGGER"] = env.get("TRAINER_LOGGER", "[\"console\",\"tensorboard\"]")
    env["TENSORBOARD_DIR"] = env.get("TENSORBOARD_DIR", str(run_dir / "tensorboard_log"))
    env["TRAIN_BATCH_SIZE"] = env.get("TRAIN_BATCH_SIZE", str(args.train_batch_size))
    env["PYTORCH_ALLOC_CONF"] = env.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.update(parse_env_overrides(args.launcher_env))

    if args.dry_run:
        rendered = "bash " + shlex.quote(str(args.continuation_launcher))
        print(f"[dry-run] launch policy={policy.name} gpu={gpu_id}: {rendered}")
        return -1

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "launcher.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ bash {args.continuation_launcher}\n\n")
        handle.flush()
        proc = subprocess.Popen(
            ["bash", str(args.continuation_launcher)],
            cwd=ROOT_DIR,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return proc.pid


def launch_fresh_job(
    *,
    args: argparse.Namespace,
    gpu_id: str,
    policy: PolicySpec,
    selected_train_parquet: Path,
    run_dir: Path,
) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHON_BIN"] = str(args.python_bin)
    env["TRAIN_FILES"] = str([str(selected_train_parquet)])
    env["RUN_DIR"] = str(run_dir)
    env["EXPERIMENT_NAME"] = f"{args.experiment_name}-{policy.name}"
    env["TOTAL_TRAINING_STEPS"] = str(args.arm_training_steps)
    env["SAVE_FREQ"] = str(args.arm_training_steps)
    env["TEST_FREQ"] = str(args.test_freq if 0 < args.test_freq <= args.arm_training_steps else args.arm_training_steps)
    env["TRAINER_LOGGER"] = env.get("TRAINER_LOGGER", "[\"console\",\"tensorboard\"]")
    env["TENSORBOARD_DIR"] = env.get("TENSORBOARD_DIR", str(run_dir / "tensorboard_log"))
    env["TRAIN_BATCH_SIZE"] = env.get("TRAIN_BATCH_SIZE", str(args.train_batch_size))
    env["PYTORCH_ALLOC_CONF"] = env.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.update(parse_env_overrides(args.launcher_env))

    if args.dry_run:
        rendered = "bash " + shlex.quote(str(args.base_launcher))
        print(f"[dry-run] launch(fresh) policy={policy.name} gpu={gpu_id}: {rendered}")
        return -1

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "launcher.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ bash {args.base_launcher}\n\n")
        handle.flush()
        proc = subprocess.Popen(
            ["bash", str(args.base_launcher)],
            cwd=ROOT_DIR,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return proc.pid


def launch_periodic_job(
    *,
    args: argparse.Namespace,
    gpu_id: str,
    policy: PolicySpec,
    start_checkpoint: Path,
    start_step: int,
    run_dir: Path,
) -> int:
    periodic_root = run_dir / "periodic"
    target_total_steps = start_step + args.arm_training_steps
    command = [
        str(args.python_bin),
        str(ROOT_DIR / "scripts" / "run_periodic_gradient_selector.py"),
        "--run-root",
        str(periodic_root),
        "--experiment-name",
        f"{args.experiment_name}-{policy.name}-periodic",
        "--window-steps",
        str(args.window_steps),
        "--total-training-steps",
        str(target_total_steps),
        "--start-from-path",
        str(start_checkpoint),
        "--selector-metric",
        policy.metric,
        "--selector",
        policy.selector,
        "--profile-window-multiplier",
        str(args.profile_window_multiplier),
        "--profile-seed",
        str(args.profile_seed),
        "--profile-group-size",
        str(args.profile_group_size),
        "--profile-rollout-batch-size",
        str(args.profile_rollout_batch_size),
        "--profile-max-prompt-tokens",
        str(args.profile_max_prompt_tokens),
        "--profile-max-new-tokens",
        str(args.profile_max_new_tokens),
        "--profile-temperature",
        str(args.profile_temperature),
        "--profile-top-k",
        str(args.profile_top_k),
        "--profile-base-model",
        str(args.profile_base_model),
    ]
    if policy.keep_count is not None:
        command.extend(["--keep-count", str(policy.keep_count)])
    else:
        command.extend(["--keep-ratio", str(policy.keep_ratio if policy.keep_ratio is not None else 0.5)])

    command.extend(["--launcher-env", f"TRAIN_BATCH_SIZE={args.train_batch_size}"])
    command.extend(["--launcher-env", "TRAINER_LOGGER=[\"console\",\"tensorboard\"]"])
    command.extend(["--launcher-env", f"TENSORBOARD_DIR={run_dir / 'tensorboard_log'}"])
    command.extend(["--launcher-env", "PYTORCH_ALLOC_CONF=expandable_segments:True"])
    for item in args.launcher_env:
        command.extend(["--launcher-env", item])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHON_BIN"] = str(args.python_bin)
    env["PYTORCH_ALLOC_CONF"] = env.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    if args.dry_run:
        rendered = " ".join(shlex.quote(x) for x in command)
        print(f"[dry-run] launch(periodic) policy={policy.name} gpu={gpu_id}: {rendered}")
        return -1

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "launcher.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(shlex.quote(x) for x in command)}\n\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return proc.pid


def main() -> None:
    args = parse_args()
    if args.mode == "continuation":
        if args.start_from_path is None:
            raise ValueError("--start-from-path is required in continuation mode.")
        if not args.start_from_path.exists():
            raise FileNotFoundError(f"Missing start checkpoint: {args.start_from_path}")
    else:
        if args.checkpoint_step is None:
            raise ValueError("--checkpoint-step is required in scratch mode.")
        if args.checkpoint_step <= 0:
            raise ValueError("--checkpoint-step must be positive.")
    if args.policy_launch_mode == "continuation" and not args.continuation_launcher.exists():
        raise FileNotFoundError(
            f"Missing continuation launcher: {args.continuation_launcher}. "
            "Use --policy-launch-mode fresh to avoid this dependency."
        )
    if args.gradient_update_mode == "periodic" and not args.continuation_launcher.exists():
        raise FileNotFoundError(
            f"Missing continuation launcher: {args.continuation_launcher}. "
            "Periodic gradient refresh requires continuation windows."
        )
    if not args.base_launcher.exists():
        raise FileNotFoundError(f"Missing base launcher: {args.base_launcher}")
    if not args.python_bin.exists():
        raise FileNotFoundError(f"Missing python binary: {args.python_bin}")
    for path in args.train_parquets:
        if not path.exists():
            raise FileNotFoundError(f"Missing train parquet: {path}")

    policies = build_policy_specs(args)
    gpu_ids = [x.strip() for x in args.gpu_ids.split(",") if x.strip()]
    if len(policies) > len(gpu_ids):
        raise ValueError(f"Need >= {len(policies)} GPUs for one-policy-per-GPU, got {len(gpu_ids)}")
    if not gpu_ids:
        raise ValueError("--gpu-ids must not be empty.")

    if args.mode == "continuation":
        assert args.start_from_path is not None
        start_checkpoint = args.start_from_path
        start_step = parse_checkpoint_step(start_checkpoint)
    else:
        warmup_gpu = args.warmup_gpu_id or gpu_ids[0]
        warmup_dir = args.run_root / "warmup"
        env = os.environ.copy()
        env["PYTHON_BIN"] = str(args.python_bin)
        env["CUDA_VISIBLE_DEVICES"] = warmup_gpu
        env["TRAIN_FILES"] = str([str(path) for path in args.train_parquets])
        env["RUN_DIR"] = str(warmup_dir)
        env["EXPERIMENT_NAME"] = f"{args.experiment_name}-warmup"
        env["TOTAL_TRAINING_STEPS"] = str(args.checkpoint_step)
        env["SAVE_FREQ"] = str(args.checkpoint_step)
        env["TEST_FREQ"] = str(effective_test_freq(args.checkpoint_step, args.test_freq))
        env["TRAINER_LOGGER"] = env.get("TRAINER_LOGGER", "[\"console\",\"tensorboard\"]")
        env["TENSORBOARD_DIR"] = env.get("TENSORBOARD_DIR", str(warmup_dir / "tensorboard_log"))
        env["TRAIN_BATCH_SIZE"] = env.get("TRAIN_BATCH_SIZE", str(args.train_batch_size))
        env.update(parse_env_overrides(args.launcher_env))
        run_cmd(
            ["bash", str(args.base_launcher)],
            env=env,
            log_path=warmup_dir / "launcher.log",
            dry_run=args.dry_run,
        )
        start_step = int(args.checkpoint_step)
        start_checkpoint = warmup_dir / f"global_step_{start_step}"

    args.run_root.mkdir(parents=True, exist_ok=True)
    selector_pool_dir = args.run_root / "selector_pools"
    selector_pool_dir.mkdir(parents=True, exist_ok=True)

    static_policies: list[PolicySpec] = []
    periodic_gradient_policies: list[PolicySpec] = []
    for policy in policies:
        if args.gradient_update_mode == "periodic" and policy.family.startswith("gradient"):
            periodic_gradient_policies.append(policy)
        else:
            static_policies.append(policy)

    v1_size, v2_size = effective_profile_sizes(args.train_batch_size, args.window_steps, args.profile_window_multiplier)
    profile_dir = args.run_root / "profile"
    if static_policies:
        profile_cmd = [
            str(args.python_bin),
            str(ROOT_DIR / "scripts" / "profile_grpo_ground_truth.py"),
            "--checkpoint-dir",
            str(start_checkpoint.parent),
            "--checkpoint-steps",
            str(start_step),
            "--output-dir",
            str(profile_dir),
            "--seed",
            str(args.profile_seed),
            "--v1-size",
            str(v1_size),
            "--v2-size",
            str(v2_size),
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
            *[str(path) for path in args.train_parquets],
        ]
        if args.exclude_files:
            profile_cmd.extend(["--exclude-files", *[str(path) for path in args.exclude_files]])
        run_cmd(profile_cmd, env=os.environ.copy(), log_path=args.run_root / "profile.log", dry_run=args.dry_run)

    jobs: list[dict[str, object]] = []
    for idx, policy in enumerate(policies):
        if policy in periodic_gradient_policies:
            gpu_id = gpu_ids[idx]
            run_dir = args.run_root / "policies" / policy.name
            pid = launch_periodic_job(
                args=args,
                gpu_id=gpu_id,
                policy=policy,
                start_checkpoint=start_checkpoint,
                start_step=start_step,
                run_dir=run_dir,
            )
            jobs.append(
                {
                    "policy": asdict(policy),
                    "job_type": "periodic_gradient",
                    "gpu_id": gpu_id,
                    "pid": pid,
                    "run_dir": str(run_dir),
                    "log_path": str(run_dir / "launcher.log"),
                    "periodic_run_root": str(run_dir / "periodic"),
                    "tensorboard_dir": str(run_dir / "tensorboard_log"),
                }
            )
            continue

        keep_args: list[str]
        if policy.keep_count is not None:
            keep_args = ["--keep-count", str(policy.keep_count)]
        else:
            keep_args = ["--keep-ratio", str(policy.keep_ratio if policy.keep_ratio is not None else 0.5)]

        selected_dir = selector_pool_dir / policy.name
        selected_dir.mkdir(parents=True, exist_ok=True)
        selected_train = selected_dir / "selected_train.parquet"
        selection_csv = selected_dir / "selected_rows.csv"
        build_cmd = [
            str(args.python_bin),
            str(ROOT_DIR / "scripts" / "build_profile_selector_dataset.py"),
            "--profile-csv",
            str(profile_dir / "ground_truth_profile.csv"),
            "--source-parquets",
            *[str(path) for path in args.train_parquets],
            "--checkpoint-step",
            str(start_step),
            "--metric",
            policy.metric,
            "--selector",
            policy.selector,
            "--seed",
            str(args.profile_seed),
            "--output-parquet",
            str(selected_train),
            "--output-selection-csv",
            str(selection_csv),
            *keep_args,
        ]
        run_cmd(
            build_cmd,
            env=os.environ.copy(),
            log_path=selected_dir / "build.log",
            dry_run=args.dry_run,
        )

        gpu_id = gpu_ids[idx]
        run_dir = args.run_root / "policies" / policy.name
        if args.policy_launch_mode == "continuation":
            pid = launch_job(
                args=args,
                gpu_id=gpu_id,
                policy=policy,
                start_checkpoint=start_checkpoint,
                start_step=start_step,
                selected_train_parquet=selected_train,
                run_dir=run_dir,
            )
        else:
            pid = launch_fresh_job(
                args=args,
                gpu_id=gpu_id,
                policy=policy,
                selected_train_parquet=selected_train,
                run_dir=run_dir,
            )
        jobs.append(
            {
                "policy": asdict(policy),
                "job_type": "static_selector",
                "gpu_id": gpu_id,
                "pid": pid,
                "run_dir": str(run_dir),
                "log_path": str(run_dir / "launcher.log"),
                "selected_train_parquet": str(selected_train),
                "selection_csv": str(selection_csv),
                "tensorboard_dir": str(run_dir / "tensorboard_log"),
            }
        )

    manifest = {
        "run_root": str(args.run_root),
        "mode": args.mode,
        "policy_launch_mode": args.policy_launch_mode,
        "gradient_update_mode": args.gradient_update_mode,
        "start_from_path": str(start_checkpoint),
        "start_step": start_step,
        "arm_training_steps": args.arm_training_steps,
        "profile": {
            "v1_size": v1_size,
            "v2_size": v2_size,
            "profile_window_multiplier": args.profile_window_multiplier,
            "profile_rollout_batch_size": args.profile_rollout_batch_size,
            "profile_max_new_tokens": args.profile_max_new_tokens,
        },
        "jobs": jobs,
    }
    manifest_path = args.run_root / "policy_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    for job in jobs:
        print(f"policy={job['policy']['name']} gpu={job['gpu_id']} pid={job['pid']} run_dir={job['run_dir']}")


if __name__ == "__main__":
    main()
