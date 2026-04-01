#!/usr/bin/env python3
"""Plot multi-policy curves from policy-grid progress CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required. Install with: ./.venv/bin/pip install matplotlib"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ["actor_loss", "gsm8k_acc", "math_acc", "response_length_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    parser.add_argument("--policies", type=str, default="", help="Comma-separated policy filter.")
    parser.add_argument("--smooth", type=float, default=0.0, help="EMA alpha in [0,1], 0 disables smoothing.")
    parser.add_argument("--output-prefix", type=Path, default=None)
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


def ema(values: list[float], alpha: float) -> list[float]:
    if not values:
        return values
    out: list[float] = []
    state = values[0]
    out.append(state)
    for value in values[1:]:
        state = alpha * value + (1.0 - alpha) * state
        out.append(state)
    return out


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv or (args.run_root / "progress" / "step_metrics_long.csv")
    ensure_progress_csv(args.run_root, input_csv)
    rows = load_rows(input_csv)
    if not rows:
        raise ValueError(f"No rows in {input_csv}")

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    policy_filter = {x.strip() for x in args.policies.split(",") if x.strip()}
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        policy = row.get("policy")
        step = row.get("global_step")
        if not isinstance(policy, str) or not isinstance(step, int):
            continue
        if policy_filter and policy not in policy_filter:
            continue
        by_policy[policy].append(row)

    if not by_policy:
        raise ValueError("No policy series found after filtering.")

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics, strict=True):
        drawn = 0
        for policy in sorted(by_policy.keys()):
            series = sorted(by_policy[policy], key=lambda r: r["global_step"])
            xs: list[int] = []
            ys: list[float] = []
            for row in series:
                value = row.get(metric)
                step = row.get("global_step")
                if isinstance(value, (int, float)) and isinstance(step, int):
                    xs.append(step)
                    ys.append(float(value))
            if not xs:
                continue
            if args.smooth > 0:
                ys = ema(ys, args.smooth)
            ax.plot(xs, ys, label=policy, linewidth=1.6)
            drawn += 1
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        if drawn == 0:
            ax.text(0.5, 0.5, f"no data for {metric}", transform=ax.transAxes, ha="center", va="center")
    axes[-1].set_xlabel("global_step")
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle("Policy Grid Curves")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    output_prefix = args.output_prefix or (args.run_root / "progress" / "policy_grid_curves")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    print(f"Wrote: {png_path}")
    print(f"Wrote: {svg_path}")


if __name__ == "__main__":
    main()
