#!/usr/bin/env python3
"""Build an apple-to-apple comparison report for policy-grid runs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--collect-rollouts", action="store_true")
    parser.add_argument("--rollout-eval-files", type=Path, nargs="*", default=[])
    parser.add_argument("--rollout-eval-limit", type=int, default=200)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=512)
    return parser.parse_args()


def run_script(script: str, args: list[str]) -> None:
    py = ROOT_DIR / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    cmd = [str(py), str(ROOT_DIR / "scripts" / script), *args]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or (args.run_root / "policy_grid_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or (args.run_root / "analysis" / "policy_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Core progress + curves + deltas
    run_script("track_policy_grid_progress.py", ["--run-root", str(args.run_root)])
    run_script("plot_policy_grid_curves.py", ["--run-root", str(args.run_root)])
    run_script("export_policy_learning_delta.py", ["--run-root", str(args.run_root), "--output-dir", str(output_dir)])

    # 2) Selected-data overlap / source composition
    selection_sets: dict[str, set[str]] = {}
    source_counts_rows: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        policy = job.get("policy", {}).get("name")
        selection_csv_raw = job.get("selection_csv")
        if not isinstance(policy, str) or not selection_csv_raw:
            continue
        selection_csv = Path(selection_csv_raw)
        rows = read_csv(selection_csv)
        keys = {row.get("example_key", "") for row in rows if row.get("example_key")}
        if not keys:
            continue
        selection_sets[policy] = keys
        src = Counter(row.get("data_source", "unknown") for row in rows)
        for data_source, count in sorted(src.items()):
            source_counts_rows.append(
                {"policy": policy, "data_source": data_source, "count": count, "selection_csv": str(selection_csv)}
            )

    overlap_rows: list[dict[str, Any]] = []
    policies = sorted(selection_sets.keys())
    for a in policies:
        for b in policies:
            overlap_rows.append(
                {
                    "policy_a": a,
                    "policy_b": b,
                    "jaccard": jaccard(selection_sets[a], selection_sets[b]),
                    "intersection": len(selection_sets[a] & selection_sets[b]),
                    "union": len(selection_sets[a] | selection_sets[b]),
                }
            )
    write_csv(output_dir / "selection_overlap_matrix.csv", overlap_rows)
    write_csv(output_dir / "selection_source_counts.csv", source_counts_rows)

    # 3) Optional rollout snapshot collection
    rollout_note = "not collected"
    if args.collect_rollouts:
        if not args.rollout_eval_files:
            raise ValueError("--collect-rollouts requires --rollout-eval-files")
        run_script(
            "collect_policy_rollout_snapshots.py",
            [
                "--run-root",
                str(args.run_root),
                "--eval-files",
                *[str(p) for p in args.rollout_eval_files],
                "--eval-limit",
                str(args.rollout_eval_limit),
                "--max-new-tokens",
                str(args.rollout_max_new_tokens),
                "--output-dir",
                str(output_dir / "rollout_snapshots"),
            ],
        )
        rollout_note = "collected in rollout_snapshots/"

    # 4) Markdown summary
    md = output_dir / "report.md"
    lines = [
        "# Policy Grid Report",
        "",
        f"Run root: `{args.run_root}`",
        "",
        "## Generated artifacts",
        "",
        "- Loss/accuracy/length curves:",
        f"  - `{args.run_root / 'progress' / 'policy_grid_curves.png'}`",
        f"  - `{args.run_root / 'progress' / 'policy_grid_curves.svg'}`",
        "- Progress tables:",
        f"  - `{args.run_root / 'progress' / 'latest_status.csv'}`",
        f"  - `{args.run_root / 'progress' / 'step_metrics_long.csv'}`",
        f"  - `{args.run_root / 'progress' / 'family_step_summary.csv'}`",
        "- Learned deltas:",
        f"  - `{output_dir / 'per_run_delta.csv'}`",
        f"  - `{output_dir / 'family_delta_summary.csv'}`",
        "- Selected-data overlap:",
        f"  - `{output_dir / 'selection_overlap_matrix.csv'}`",
        f"  - `{output_dir / 'selection_source_counts.csv'}`",
        f"- Rollout snapshots: {rollout_note}",
        "",
        "## Apple-to-apple comparison checklist",
        "",
        "1. Compare `family_step_summary.csv` at matched `global_step`.",
        "2. Compare selected-data overlap (`selection_overlap_matrix.csv`).",
        "3. Compare selected-data composition (`selection_source_counts.csv`).",
        "4. Use rollout snapshots for per-example qualitative differences.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {md}")
    print(f"Wrote: {output_dir / 'selection_overlap_matrix.csv'}")
    print(f"Wrote: {output_dir / 'selection_source_counts.csv'}")


if __name__ == "__main__":
    main()
