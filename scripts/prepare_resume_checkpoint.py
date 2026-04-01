#!/usr/bin/env python3
"""Copy a verl checkpoint for branching, optionally dropping dataloader state."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-step-dir",
        type=Path,
        required=True,
        help="Source checkpoint directory such as .../global_step_200",
    )
    parser.add_argument(
        "--output-step-dir",
        type=Path,
        required=True,
        help="Destination checkpoint directory such as .../branch_seed/global_step_200",
    )
    parser.add_argument(
        "--keep-data-state",
        action="store_true",
        help="Keep data.pt in the copied checkpoint. Default drops it so a new train dataset can start cleanly.",
    )
    return parser.parse_args()


def copy_step_dir(source_step_dir: Path, output_step_dir: Path, keep_data_state: bool) -> dict[str, object]:
    if not source_step_dir.exists():
        raise FileNotFoundError(f"Missing source checkpoint directory: {source_step_dir}")
    if source_step_dir.name.startswith("global_step_") is False:
        raise ValueError(f"Source path must point to a global_step_* directory, got {source_step_dir}")
    if output_step_dir.exists():
        raise FileExistsError(f"Destination already exists: {output_step_dir}")

    ignore = None
    if not keep_data_state:
        ignore = shutil.ignore_patterns("data.pt")

    output_step_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_step_dir, output_step_dir, ignore=ignore)

    step = int(source_step_dir.name.split("global_step_")[-1])
    latest_path = output_step_dir.parent / "latest_checkpointed_iteration.txt"
    latest_path.write_text(str(step), encoding="utf-8")

    manifest = {
        "source_step_dir": str(source_step_dir),
        "output_step_dir": str(output_step_dir),
        "step": step,
        "kept_data_state": keep_data_state,
        "latest_checkpointed_iteration": str(latest_path),
    }
    return manifest


def main() -> None:
    args = parse_args()
    manifest = copy_step_dir(args.source_step_dir, args.output_step_dir, args.keep_data_state)
    manifest_path = args.output_step_dir.parent / "branch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Copied checkpoint branch to {args.output_step_dir}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
