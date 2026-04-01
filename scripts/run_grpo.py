#!/usr/bin/env python3
"""Plan a canonical GRPO run from a lightweight experiment config.

This script does not yet launch `verl` directly. Its job in the first scaffold
is to validate the config, make the intended experiment explicit, and emit a
launch plan that later can be translated into the exact training command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "PyYAML is required to read config files. Install `pyyaml` first."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise SystemExit(f"Config {path} must decode to a mapping.")
    return data


def build_launch_plan(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    training = config.get("training", {})
    model = config.get("model", {})
    data = config.get("data", {})
    profiling = config.get("profiling", {})
    runtime = config.get("runtime", {})

    return {
        "experiment_name": experiment.get("name"),
        "seed": experiment.get("seed"),
        "output_dir": experiment.get("output_dir"),
        "framework": training.get("framework"),
        "algorithm": training.get("algorithm"),
        "train_steps": training.get("train_steps"),
        "eval_every": training.get("eval_every"),
        "save_every": training.get("save_every"),
        "group_size": training.get("group_size"),
        "max_prompt_length": training.get("max_prompt_length"),
        "max_completion_length": training.get("max_completion_length"),
        "policy_model": model.get("policy_model"),
        "reference_model": model.get("reference_model"),
        "train_dataset": data.get("train_dataset"),
        "train_files": data.get("train_files"),
        "train_split": data.get("train_split"),
        "eval_dataset": data.get("eval_dataset"),
        "eval_files": data.get("eval_files"),
        "recipe_origin": config.get("recipe_origin"),
        "launcher_hint": runtime.get("launcher_hint"),
        "profiling_enabled": {
            "rollout": profiling.get("log_rollout_records", False),
            "training": profiling.get("log_training_records", False),
        },
        "translation_todo": [
            "map this plan to the exact verl CLI or Python launcher",
            "attach reward/verifier config",
            "attach cluster, distributed, and checkpoint settings",
            "ensure sample ids survive rollout and training logs",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Write launch_plan.json next to the experiment output dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    plan = build_launch_plan(config)

    print(json.dumps(plan, indent=2, sort_keys=True))

    if not args.write_plan:
        return

    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "launch_plan.json"
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
