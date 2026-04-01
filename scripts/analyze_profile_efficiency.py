#!/usr/bin/env python3
"""Analyze a profiling run and render summary SVGs/CSVs."""

from __future__ import annotations

import argparse
import colorsys
import csv
import html
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


BG = "#f6f1e8"
PANEL_BG = "#fffdf8"
PANEL_BORDER = "#d9d2c3"
GRID = "#ece7de"
AXIS = "#555"
TEXT = "#222"
MUTED = "#666"
BLUE = "#0a6c8f"
ORANGE = "#d95f02"
GREEN = "#2a9d8f"
ROSE = "#b56576"
RED = "#c0392b"
CKPT_COLORS = {50: "#0a6c8f", 100: "#d95f02", 150: "#2a9d8f", 200: "#b56576"}
BATCH_SIZE_COLORS = {2: "#0a6c8f", 4: "#d95f02", 8: "#2a9d8f", 16: "#b56576", 32: "#6c757d"}
REMOVAL_CRITERIA = [
    ("gradient_goodput", "goodput", "Remove Lowest Goodput"),
    ("gradient_statistical_efficiency", "stat_effi", "Remove Lowest Stat Effi"),
    ("computational_efficiency", "comp_effi", "Remove Lowest Comp Effi"),
]

PANEL_W = 300
PANEL_H = 230
MARGIN_L = 56
MARGIN_R = 16
MARGIN_T = 26
MARGIN_B = 36


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-csv", type=Path, required=True)
    parser.add_argument("--rollout-detail-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", type=str, default="Profiling")
    parser.add_argument(
        "--mini-batch-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
        help="Mini-batch sizes used for sampled V2 subset aggregation.",
    )
    parser.add_argument(
        "--mini-batch-samples",
        type=int,
        default=100,
        help="Maximum number of unique random subsets sampled per mini-batch size.",
    )
    parser.add_argument(
        "--mini-batch-seed",
        type=int,
        default=0,
        help="Random seed for mini-batch subset sampling.",
    )
    parser.add_argument(
        "--stat-vs-compute-plot-batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of mini-batch sizes to display in the stat-vs-compute figure.",
    )
    parser.add_argument(
        "--stat-vs-compute-mode",
        type=str,
        choices=["per_example", "mini_batch"],
        default="per_example",
        help="Whether stat-vs-compute renders the original per-example view or the mini-batched view.",
    )
    parser.add_argument(
        "--removal-batch-size",
        type=int,
        default=16,
        help="Batch size used for retained-pool goodput-vs-removal analysis.",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv_rows(rows: list[dict[str, object]], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_tick(value: float) -> str:
    if abs(value) >= 100 or value.is_integer():
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def bounds(values: list[float], clamp_zero: bool = False, pad_frac: float = 0.08) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    lo = min(values)
    hi = max(values)
    if clamp_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if lo == hi:
        pad = 0.1 if lo == 0 else abs(lo) * 0.1
        return (lo - pad, hi + pad)
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def map_point(
    x: float,
    y: float,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    left: float,
    top: float,
    panel_w: float,
    panel_h: float,
) -> tuple[float, float]:
    plot_w = panel_w - MARGIN_L - MARGIN_R
    plot_h = panel_h - MARGIN_T - MARGIN_B
    px = left + MARGIN_L + (x - x_lo) / (x_hi - x_lo) * plot_w
    py = top + MARGIN_T + plot_h - (y - y_lo) / (y_hi - y_lo) * plot_h
    return px, py


def draw_axes(
    left: float,
    top: float,
    title: str,
    x_label: str,
    y_label: str,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    panel_w: float,
    panel_h: float,
) -> str:
    plot_w = panel_w - MARGIN_L - MARGIN_R
    plot_h = panel_h - MARGIN_T - MARGIN_B
    x0 = left + MARGIN_L
    y0 = top + MARGIN_T + plot_h
    parts = [
        f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" rx="14" fill="{PANEL_BG}" stroke="{PANEL_BORDER}" />',
        f'<text x="{left + 14}" y="{top + 20}" font-size="14" font-weight="600" fill="{TEXT}">{html.escape(title)}</text>',
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{left + MARGIN_L + plot_w:.2f}" y2="{y0:.2f}" stroke="{AXIS}" stroke-width="1" />',
        f'<line x1="{x0:.2f}" y1="{top + MARGIN_T:.2f}" x2="{x0:.2f}" y2="{y0:.2f}" stroke="{AXIS}" stroke-width="1" />',
    ]
    for tick in ticks(y_lo, y_hi):
        _, py = map_point(x_lo, tick, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
        parts.append(
            f'<line x1="{x0:.2f}" y1="{py:.2f}" x2="{left + MARGIN_L + plot_w:.2f}" y2="{py:.2f}" stroke="{GRID}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{left + MARGIN_L - 8}" y="{py + 4:.2f}" text-anchor="end" font-size="10" fill="{AXIS}">{html.escape(fmt_tick(tick))}</text>'
        )
    for tick in ticks(x_lo, x_hi):
        px, _ = map_point(tick, y_lo, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
        parts.append(
            f'<line x1="{px:.2f}" y1="{top + MARGIN_T:.2f}" x2="{px:.2f}" y2="{y0:.2f}" stroke="#f3eee7" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{top + MARGIN_T + plot_h + 18:.2f}" text-anchor="middle" font-size="10" fill="{AXIS}">{html.escape(fmt_tick(tick))}</text>'
        )
    parts.append(
        f'<text x="{left + MARGIN_L + plot_w / 2:.2f}" y="{top + panel_h - 8:.2f}" text-anchor="middle" font-size="11" fill="{AXIS}">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="{left + 14:.2f}" y="{top + panel_h / 2:.2f}" transform="rotate(-90 {left + 14:.2f} {top + panel_h / 2:.2f})" text-anchor="middle" font-size="11" fill="{AXIS}">{html.escape(y_label)}</text>'
    )
    return "".join(parts)


def caption_block(left: float, top: float, width: float, lines: list[str]) -> str:
    height = 20 + 15 * len(lines)
    parts = [f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="12" fill="{PANEL_BG}" stroke="{PANEL_BORDER}" />']
    for idx, line in enumerate(lines):
        parts.append(f'<text x="{left + 12}" y="{top + 18 + idx * 15}" font-size="11" fill="{MUTED}">{html.escape(line)}</text>')
    return "".join(parts)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return 0.0, ys[0] if ys else 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def example_style(example_key: str) -> tuple[str, str, str]:
    suffix = example_key.split("::")[-1]
    try:
        seed = int(suffix)
    except ValueError:
        seed = sum(ord(ch) for ch in example_key)
    hue = ((seed * 0.61803398875) % 1.0)
    sat = 0.65
    val = 0.82
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    shape = ["circle", "square", "diamond", "triangle"][seed % 4]
    return color, shape, suffix


def draw_marker(px: float, py: float, shape: str, size: float, color: str, opacity: float = 0.88) -> str:
    if shape == "square":
        return f'<rect x="{px - size:.2f}" y="{py - size:.2f}" width="{2*size:.2f}" height="{2*size:.2f}" fill="{color}" opacity="{opacity}" />'
    if shape == "diamond":
        points = f"{px:.2f},{py-size:.2f} {px+size:.2f},{py:.2f} {px:.2f},{py+size:.2f} {px-size:.2f},{py:.2f}"
        return f'<polygon points="{points}" fill="{color}" opacity="{opacity}" />'
    if shape == "triangle":
        points = f"{px:.2f},{py-size:.2f} {px+size:.2f},{py+size:.2f} {px-size:.2f},{py+size:.2f}"
        return f'<polygon points="{points}" fill="{color}" opacity="{opacity}" />'
    return f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{size:.2f}" fill="{color}" opacity="{opacity}" />'


def summarize_profile_rows(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, object]]:
    by_ckpt: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ckpt[int(row["checkpoint_step"])].append(row)

    summary_rows: list[dict[str, object]] = []
    fieldnames = [
        "checkpoint_step",
        "n_points",
        "gradient_stat_eff_mean",
        "gradient_stat_eff_std",
        "gradient_stat_eff_min",
        "gradient_stat_eff_max",
        "pods_stat_eff_mean",
        "pods_stat_eff_std",
        "pods_stat_eff_min",
        "pods_stat_eff_max",
        "compute_eff_mean",
        "compute_eff_std",
        "compute_eff_min",
        "compute_eff_max",
        "dapo_partial_mean",
        "dapo_keep_mean",
        "rollout_cost_mean",
        "grad_cost_mean",
        "total_cost_mean",
    ]
    for ckpt in sorted(by_ckpt):
        subset = by_ckpt[ckpt]
        grad = [float(row["gradient_statistical_efficiency"]) for row in subset]
        pods = [float(row["pods_statistical_efficiency"]) for row in subset]
        comp = [float(row["computational_efficiency"]) for row in subset]
        dapo_partial = [float(row["dapo_statistical_efficiency"]) for row in subset]
        dapo_keep = [float(row["dapo_keep_efficiency"]) for row in subset]
        rollout = [float(row["C_roll"]) for row in subset]
        grad_cost = [float(row["T_grad"]) for row in subset]
        total = [float(row["total_cost_s"]) for row in subset]
        summary_rows.append(
            {
                "checkpoint_step": ckpt,
                "n_points": len(subset),
                "gradient_stat_eff_mean": statistics.mean(grad),
                "gradient_stat_eff_std": statistics.pstdev(grad),
                "gradient_stat_eff_min": min(grad),
                "gradient_stat_eff_max": max(grad),
                "pods_stat_eff_mean": statistics.mean(pods),
                "pods_stat_eff_std": statistics.pstdev(pods),
                "pods_stat_eff_min": min(pods),
                "pods_stat_eff_max": max(pods),
                "compute_eff_mean": statistics.mean(comp),
                "compute_eff_std": statistics.pstdev(comp),
                "compute_eff_min": min(comp),
                "compute_eff_max": max(comp),
                "dapo_partial_mean": statistics.mean(dapo_partial),
                "dapo_keep_mean": statistics.mean(dapo_keep),
                "rollout_cost_mean": statistics.mean(rollout),
                "grad_cost_mean": statistics.mean(grad_cost),
                "total_cost_mean": statistics.mean(total),
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def summarize_correlations(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, object]]:
    by_ckpt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ckpt[row["checkpoint_step"]].append(row)
    by_ckpt["overall"] = rows

    summary_rows: list[dict[str, object]] = []
    fieldnames = [
        "checkpoint_step",
        "n_points",
        "pearson_gradient_vs_dapo_partial",
        "pearson_gradient_vs_dapo_keep",
    ]
    for key, subset in by_ckpt.items():
        grad = [float(row["gradient_statistical_efficiency"]) for row in subset]
        dapo_partial = [float(row["dapo_statistical_efficiency"]) for row in subset]
        dapo_keep = [float(row["dapo_keep_efficiency"]) for row in subset]
        summary_rows.append(
            {
                "checkpoint_step": key,
                "n_points": len(subset),
                "pearson_gradient_vs_dapo_partial": pearson(grad, dapo_partial),
                "pearson_gradient_vs_dapo_keep": pearson(grad, dapo_keep),
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def summarize_stat_efficiency_correlations(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, object]]:
    by_ckpt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ckpt[row["checkpoint_step"]].append(row)
    by_ckpt["overall"] = rows

    fieldnames = [
        "checkpoint_step",
        "n_points",
        "pearson_gradient_vs_dapo_partial",
        "pearson_gradient_vs_pods",
        "pearson_dapo_partial_vs_pods",
    ]
    summary_rows: list[dict[str, object]] = []
    for key, subset in by_ckpt.items():
        gradient = [float(row["gradient_statistical_efficiency"]) for row in subset]
        dapo_partial = [float(row["dapo_statistical_efficiency"]) for row in subset]
        pods = [float(row["pods_statistical_efficiency"]) for row in subset]
        summary_rows.append(
            {
                "checkpoint_step": key,
                "n_points": len(subset),
                "pearson_gradient_vs_dapo_partial": pearson(gradient, dapo_partial),
                "pearson_gradient_vs_pods": pearson(gradient, pods),
                "pearson_dapo_partial_vs_pods": pearson(dapo_partial, pods),
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def summarize_direction_changes(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, object]]:
    by_example: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_example[row["example_key"]][int(row["checkpoint_step"])] = row

    checkpoints = sorted({int(row["checkpoint_step"]) for row in rows})
    metrics = [
        "gradient_statistical_efficiency",
        "computational_efficiency",
        "dapo_statistical_efficiency",
    ]
    metric_ranges = {
        metric: max(float(row[metric]) for row in rows) - min(float(row[metric]) for row in rows)
        for metric in metrics
    }
    fieldnames = [
        "metric",
        "from_checkpoint",
        "to_checkpoint",
        "n_examples",
        "metric_range",
        "similarity_threshold",
        "up_count",
        "down_count",
        "flat_count",
        "up_pct",
        "down_pct",
        "flat_pct",
    ]
    summary_rows: list[dict[str, object]] = []
    for metric in metrics:
        similarity_threshold = 0.05 * metric_ranges[metric]
        for start_ckpt, end_ckpt in zip(checkpoints, checkpoints[1:]):
            up = down = flat = 0
            for values in by_example.values():
                delta = float(values[end_ckpt][metric]) - float(values[start_ckpt][metric])
                if abs(delta) <= similarity_threshold:
                    flat += 1
                elif delta > 0:
                    up += 1
                else:
                    down += 1
            n_examples = up + down + flat
            summary_rows.append(
                {
                    "metric": metric,
                    "from_checkpoint": start_ckpt,
                    "to_checkpoint": end_ckpt,
                    "n_examples": n_examples,
                    "metric_range": metric_ranges[metric],
                    "similarity_threshold": similarity_threshold,
                    "up_count": up,
                    "down_count": down,
                    "flat_count": flat,
                    "up_pct": 100.0 * up / n_examples,
                    "down_pct": 100.0 * down / n_examples,
                    "flat_pct": 100.0 * flat / n_examples,
                }
            )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def summarize_accuracy_buckets(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, object]]:
    by_ckpt: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ckpt[int(row["checkpoint_step"])].append(row)

    fieldnames = [
        "checkpoint_step",
        "bucket",
        "n_points",
        "mean_gradient_statistical_efficiency",
        "mean_total_cost_s",
        "mean_computational_efficiency",
    ]
    summary_rows: list[dict[str, object]] = []
    for ckpt in sorted(by_ckpt):
        subset = by_ckpt[ckpt]
        buckets = {
            "solved": [row for row in subset if float(row["group_accuracy"]) == 1.0],
            "partial": [row for row in subset if 0.0 < float(row["group_accuracy"]) < 1.0],
            "zero": [row for row in subset if float(row["group_accuracy"]) == 0.0],
        }
        for bucket_name, bucket_rows in buckets.items():
            if not bucket_rows:
                continue
            summary_rows.append(
                {
                    "checkpoint_step": ckpt,
                    "bucket": bucket_name,
                    "n_points": len(bucket_rows),
                    "mean_gradient_statistical_efficiency": statistics.mean(
                        float(row["gradient_statistical_efficiency"]) for row in bucket_rows
                    ),
                    "mean_total_cost_s": statistics.mean(float(row["total_cost_s"]) for row in bucket_rows),
                    "mean_computational_efficiency": statistics.mean(
                        float(row["computational_efficiency"]) for row in bucket_rows
                    ),
                }
            )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def summarize_goodput(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, object]]:
    by_ckpt: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ckpt[int(row["checkpoint_step"])].append(row)

    fieldnames = [
        "checkpoint_step",
        "n_points",
        "mean_gradient_goodput",
        "std_gradient_goodput",
        "min_gradient_goodput",
        "max_gradient_goodput",
    ]
    summary_rows: list[dict[str, object]] = []
    for ckpt in sorted(by_ckpt):
        values = [float(row["gradient_goodput"]) for row in by_ckpt[ckpt]]
        summary_rows.append(
            {
                "checkpoint_step": ckpt,
                "n_points": len(values),
                "mean_gradient_goodput": statistics.mean(values),
                "std_gradient_goodput": statistics.pstdev(values),
                "min_gradient_goodput": min(values),
                "max_gradient_goodput": max(values),
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def summarize_goodput_correlations(rows: list[dict[str, str]], output_path: Path) -> list[dict[str, object]]:
    by_ckpt: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ckpt[row["checkpoint_step"]].append(row)
    by_ckpt["overall"] = rows

    fieldnames = [
        "checkpoint_step",
        "n_points",
        "pearson_goodput_vs_stat_eff",
        "pearson_goodput_vs_compute_eff",
    ]
    summary_rows: list[dict[str, object]] = []
    for key, subset in by_ckpt.items():
        goodput = [float(row["gradient_goodput"]) for row in subset]
        stat_eff = [float(row["gradient_statistical_efficiency"]) for row in subset]
        compute_eff = [float(row["computational_efficiency"]) for row in subset]
        summary_rows.append(
            {
                "checkpoint_step": key,
                "n_points": len(subset),
                "pearson_goodput_vs_stat_eff": pearson(goodput, stat_eff),
                "pearson_goodput_vs_compute_eff": pearson(goodput, compute_eff),
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def sample_mini_batch_subsets(
    example_keys: list[str],
    batch_sizes: list[int],
    num_samples: int,
    seed: int,
) -> dict[int, list[tuple[str, ...]]]:
    rng = random.Random(seed)
    sampled: dict[int, list[tuple[str, ...]]] = {}
    for batch_size in sorted(set(batch_sizes)):
        if batch_size <= 0 or batch_size > len(example_keys):
            continue
        max_unique = math.comb(len(example_keys), batch_size)
        target = min(num_samples, max_unique)
        if target <= 0:
            continue
        if max_unique <= 5000 and target == max_unique:
            subsets = list(itertools.combinations(example_keys, batch_size))
            rng.shuffle(subsets)
            sampled[batch_size] = [tuple(subset) for subset in subsets[:target]]
            continue
        seen: set[tuple[str, ...]] = set()
        chosen: list[tuple[str, ...]] = []
        while len(chosen) < target:
            subset = tuple(sorted(rng.sample(example_keys, batch_size)))
            if subset in seen:
                continue
            seen.add(subset)
            chosen.append(subset)
        sampled[batch_size] = chosen
    return sampled


def build_mini_batched_rows(
    rows: list[dict[str, str]],
    batch_sizes: list[int],
    num_samples: int,
    seed: int,
    output_path: Path,
) -> tuple[list[dict[str, object]], dict[int, int], int]:
    by_ckpt: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_ckpt[int(row["checkpoint_step"])][row["example_key"]] = row

    checkpoint_keys = [set(mapping) for mapping in by_ckpt.values()]
    if not checkpoint_keys:
        raise SystemExit("no checkpoint rows found for mini-batched analysis")
    shared_example_keys = sorted(set.intersection(*checkpoint_keys))
    if not shared_example_keys:
        raise SystemExit("no shared V2 examples across checkpoints for mini-batched analysis")

    sampled_subsets = sample_mini_batch_subsets(shared_example_keys, batch_sizes, num_samples, seed)
    batch_counts = {batch_size: len(subsets) for batch_size, subsets in sampled_subsets.items()}

    fieldnames = [
        "checkpoint_step",
        "batch_size",
        "batch_index",
        "example_keys",
        "mini_batched_stat_efficiency",
        "mini_batched_compute_efficiency",
        "mini_batched_goodput",
    ]
    batched_rows: list[dict[str, object]] = []
    for batch_size in sorted(sampled_subsets):
        for batch_index, subset_keys in enumerate(sampled_subsets[batch_size], start=1):
            subset_label = "|".join(subset_keys)
            for ckpt in sorted(by_ckpt):
                subset_rows = [by_ckpt[ckpt][key] for key in subset_keys]
                stat_eff = statistics.mean(float(row["gradient_statistical_efficiency"]) for row in subset_rows)
                compute_eff = min(float(row["computational_efficiency"]) for row in subset_rows)
                goodput = stat_eff * compute_eff
                batched_rows.append(
                    {
                        "checkpoint_step": ckpt,
                        "batch_size": batch_size,
                        "batch_index": batch_index,
                        "example_keys": subset_label,
                        "mini_batched_stat_efficiency": stat_eff,
                        "mini_batched_compute_efficiency": compute_eff,
                        "mini_batched_goodput": goodput,
                    }
                )

    write_csv_rows(batched_rows, output_path, fieldnames)
    return batched_rows, batch_counts, len(shared_example_keys)


def build_removal_goodput_curve_rows(
    rows: list[dict[str, str]],
    batch_size: int,
    num_samples: int,
    seed: int,
    curve_output_path: Path,
    summary_output_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    by_ckpt: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_ckpt[int(row["checkpoint_step"])][row["example_key"]] = row

    checkpoint_keys = [set(mapping) for mapping in by_ckpt.values()]
    if not checkpoint_keys:
        raise SystemExit("no checkpoint rows found for removal analysis")
    shared_example_keys = sorted(set.intersection(*checkpoint_keys))
    if batch_size > len(shared_example_keys):
        raise SystemExit(f"removal batch size {batch_size} exceeds shared V2 size {len(shared_example_keys)}")

    curve_fieldnames = [
        "checkpoint_step",
        "ranking_metric",
        "ranking_field",
        "removed_count",
        "removed_pct",
        "retained_count",
        "batch_size",
        "n_batches_sampled",
        "mean_mini_batched_goodput",
        "std_mini_batched_goodput",
        "min_mini_batched_goodput",
        "max_mini_batched_goodput",
    ]
    summary_fieldnames = [
        "checkpoint_step",
        "ranking_metric",
        "ranking_field",
        "baseline_goodput",
        "best_goodput",
        "absolute_gain",
        "best_removed_count",
        "best_removed_pct",
    ]

    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    max_removed = len(shared_example_keys) - batch_size

    for ckpt in sorted(by_ckpt):
        ckpt_row_map = by_ckpt[ckpt]
        ckpt_rows = [ckpt_row_map[key] for key in shared_example_keys]
        for criterion_index, (ranking_field, ranking_metric, _title) in enumerate(REMOVAL_CRITERIA):
            ranked_rows = sorted(ckpt_rows, key=lambda row: (float(row[ranking_field]), row["example_key"]))
            baseline_goodput = None
            best_goodput = None
            best_removed_count = 0
            best_removed_pct = 0.0
            for removed_count in range(max_removed + 1):
                retained_rows = ranked_rows[removed_count:]
                retained_keys = sorted(row["example_key"] for row in retained_rows)
                sampled_subsets = sample_mini_batch_subsets(
                    retained_keys,
                    [batch_size],
                    num_samples,
                    seed + ckpt * 1000 + criterion_index * 100 + removed_count,
                )
                subsets = sampled_subsets.get(batch_size, [])
                goodputs: list[float] = []
                for subset_keys in subsets:
                    subset_rows = [ckpt_row_map[key] for key in subset_keys]
                    stat_eff = statistics.mean(float(row["gradient_statistical_efficiency"]) for row in subset_rows)
                    compute_eff = min(float(row["computational_efficiency"]) for row in subset_rows)
                    goodputs.append(stat_eff * compute_eff)
                if not goodputs:
                    continue
                mean_goodput = statistics.mean(goodputs)
                removed_pct = 100.0 * removed_count / len(shared_example_keys)
                curve_rows.append(
                    {
                        "checkpoint_step": ckpt,
                        "ranking_metric": ranking_metric,
                        "ranking_field": ranking_field,
                        "removed_count": removed_count,
                        "removed_pct": removed_pct,
                        "retained_count": len(retained_keys),
                        "batch_size": batch_size,
                        "n_batches_sampled": len(goodputs),
                        "mean_mini_batched_goodput": mean_goodput,
                        "std_mini_batched_goodput": statistics.pstdev(goodputs) if len(goodputs) > 1 else 0.0,
                        "min_mini_batched_goodput": min(goodputs),
                        "max_mini_batched_goodput": max(goodputs),
                    }
                )
                if removed_count == 0:
                    baseline_goodput = mean_goodput
                if best_goodput is None or mean_goodput > best_goodput:
                    best_goodput = mean_goodput
                    best_removed_count = removed_count
                    best_removed_pct = removed_pct

            summary_rows.append(
                {
                    "checkpoint_step": ckpt,
                    "ranking_metric": ranking_metric,
                    "ranking_field": ranking_field,
                    "baseline_goodput": baseline_goodput,
                    "best_goodput": best_goodput,
                    "absolute_gain": None if baseline_goodput is None or best_goodput is None else best_goodput - baseline_goodput,
                    "best_removed_count": best_removed_count,
                    "best_removed_pct": best_removed_pct,
                }
            )

    write_csv_rows(curve_rows, curve_output_path, curve_fieldnames)
    write_csv_rows(summary_rows, summary_output_path, summary_fieldnames)
    return curve_rows, summary_rows, len(shared_example_keys)


def render_per_example_stat_vs_compute(rows: list[dict[str, str]], output_path: Path, title_prefix: str) -> None:
    checkpoints = sorted({int(row["checkpoint_step"]) for row in rows})
    comp_values = [float(row["computational_efficiency"]) for row in rows]
    grad_values = [float(row["gradient_statistical_efficiency"]) for row in rows]
    x_lo, x_hi = bounds(comp_values, clamp_zero=True)
    grad_lo, grad_hi = bounds(grad_values, clamp_zero=True)
    dapo_lo, dapo_hi = (-0.1, 1.1)

    width = len(checkpoints) * PANEL_W + (len(checkpoints) - 1) * 8 + 40
    height = 2 * PANEL_H + 170
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Statistical vs Computational Efficiency</text>',
    ]
    for row_index, (metric_key, y_lo, y_hi, row_title, color) in enumerate(
        [
            ("gradient_statistical_efficiency", grad_lo, grad_hi, "gradient-based", BLUE),
            ("dapo_statistical_efficiency", dapo_lo, dapo_hi, "DAPO-partial", ORANGE),
        ]
    ):
        for col_index, ckpt in enumerate(checkpoints):
            left = 20 + col_index * (PANEL_W + 8)
            top = 50 + row_index * (PANEL_H + 12)
            subset = [row for row in rows if int(row["checkpoint_step"]) == ckpt]
            parts.append(
                draw_axes(
                    left,
                    top,
                    f"{row_title} @ ckpt {ckpt}",
                    "computational_efficiency (sample / s)",
                    "statistical_efficiency (Gain / sample)",
                    x_lo,
                    x_hi,
                    y_lo,
                    y_hi,
                    PANEL_W,
                    PANEL_H,
                )
            )
            xs = [float(row["computational_efficiency"]) for row in subset]
            ys = [float(row[metric_key]) for row in subset]
            slope, intercept = linear_fit(xs, ys)
            x1, y1 = x_lo, slope * x_lo + intercept
            x2, y2 = x_hi, slope * x_hi + intercept
            px1, py1 = map_point(x1, y1, x_lo, x_hi, y_lo, y_hi, left, top, PANEL_W, PANEL_H)
            px2, py2 = map_point(x2, y2, x_lo, x_hi, y_lo, y_hi, left, top, PANEL_W, PANEL_H)
            parts.append(
                f'<line x1="{px1:.2f}" y1="{py1:.2f}" x2="{px2:.2f}" y2="{py2:.2f}" stroke="{color}" stroke-width="2.2" stroke-dasharray="6 4" opacity="0.95" />'
            )
            parts.append(
                f'<text x="{left + PANEL_W - 14:.2f}" y="{top + PANEL_H - 18:.2f}" text-anchor="end" font-size="9" fill="{color}">y = {slope:.2f}x + {intercept:.3f}</text>'
            )
            for point_index, row in enumerate(subset):
                x = float(row["computational_efficiency"])
                y = float(row[metric_key])
                px, py = map_point(x, y, x_lo, x_hi, y_lo, y_hi, left, top, PANEL_W, PANEL_H)
                jitter = ((point_index % 4) - 1.5) * 1.8 if metric_key == "dapo_statistical_efficiency" else 0.0
                example_color, example_shape, example_label = example_style(row["example_key"])
                parts.append(draw_marker(px + jitter, py, example_shape, 3.9, example_color))
                if metric_key == "gradient_statistical_efficiency":
                    parts.append(
                        f'<text x="{px + jitter + 4.5:.2f}" y="{py - 4.5:.2f}" font-size="8.5" fill="{example_color}">{html.escape(example_label)}</text>'
                    )
    caption_lines = [
        f"gradient_statistical_efficiency (Gain/sample) observed range: [{min(grad_values):.4f}, {max(grad_values):.4f}]. Positive is good, negative is bad, near zero is weak/no immediate gain.",
        f"computational_efficiency (sample/s) observed range: [{min(comp_values):.4f}, {max(comp_values):.4f}]. Higher means cheaper examples.",
        "DAPO-partial is 1[0 < group_accuracy < 1], so it ranges from 0 to 1: 1 means partially solved, 0 means either zero-acc or fully solved.",
        "Each panel shows an ordinary least-squares linear fit for that checkpoint.",
        "This is the original per-example bsz=1 view; the same example keeps the same marker color and shape across all panels.",
    ]
    parts.append(caption_block(20, 2 * PANEL_H + 76, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_stat_vs_compute(
    batched_rows: list[dict[str, object]],
    batch_counts: dict[int, int],
    v2_size: int,
    requested_num_samples: int,
    plot_batch_sizes: list[int] | None,
    output_path: Path,
    title_prefix: str,
) -> None:
    if plot_batch_sizes is not None:
        allowed_batch_sizes = set(plot_batch_sizes)
        batched_rows = [row for row in batched_rows if int(row["batch_size"]) in allowed_batch_sizes]
        batch_counts = {batch_size: count for batch_size, count in batch_counts.items() if batch_size in allowed_batch_sizes}
    if not batched_rows:
        raise SystemExit("no mini-batched rows available for the requested stat-vs-compute plot batch sizes")
    checkpoints = sorted({int(row["checkpoint_step"]) for row in batched_rows})
    batch_sizes = sorted({int(row["batch_size"]) for row in batched_rows})
    stat_values = [float(row["mini_batched_stat_efficiency"]) for row in batched_rows]
    comp_values = [float(row["mini_batched_compute_efficiency"]) for row in batched_rows]
    goodput_values = [float(row["mini_batched_goodput"]) for row in batched_rows]
    stat_lo, stat_hi = bounds(stat_values, clamp_zero=True)
    comp_lo, comp_hi = bounds(comp_values, clamp_zero=True)
    goodput_lo, goodput_hi = bounds(goodput_values, clamp_zero=True)

    panel_specs = [
        (
            "mini_batched_compute_efficiency",
            "mini_batched_stat_efficiency",
            "mini-batched stat vs compute",
            "mini_batched_compute_efficiency (min sample / s)",
            "mini_batched_stat_efficiency (mean Gain / sample)",
            comp_lo,
            comp_hi,
            stat_lo,
            stat_hi,
        ),
        (
            "mini_batched_stat_efficiency",
            "mini_batched_goodput",
            "mini-batched goodput vs stat",
            "mini_batched_stat_efficiency (mean Gain / sample)",
            "mini_batched_goodput (Gain / s)",
            stat_lo,
            stat_hi,
            goodput_lo,
            goodput_hi,
        ),
        (
            "mini_batched_compute_efficiency",
            "mini_batched_goodput",
            "mini-batched goodput vs compute",
            "mini_batched_compute_efficiency (min sample / s)",
            "mini_batched_goodput (Gain / s)",
            comp_lo,
            comp_hi,
            goodput_lo,
            goodput_hi,
        ),
    ]

    width = len(checkpoints) * PANEL_W + (len(checkpoints) - 1) * 8 + 40
    legend_y = 50 + len(panel_specs) * PANEL_H + (len(panel_specs) - 1) * 12 + 26
    caption_lines = [
        f"Shared V2 pool size = {v2_size}. For each batch size in {{{', '.join(str(size) for size in batch_sizes)}}}, the script samples up to {requested_num_samples} unique random V2 subsets and reuses the same subsets across checkpoints.",
        f"mini_batched_stat_efficiency = mean gradient_statistical_efficiency over the subset. Observed range: [{min(stat_values):.4f}, {max(stat_values):.4f}].",
        f"mini_batched_compute_efficiency = min computational_efficiency over the subset. Observed range: [{min(comp_values):.4f}, {max(comp_values):.4f}].",
        f"mini_batched_goodput = mini_batched_stat_efficiency * mini_batched_compute_efficiency. Observed range: [{min(goodput_values):.4f}, {max(goodput_values):.4f}].",
        "Small points are individual sampled subsets; larger outlined points are the mean over sampled subsets at that batch size within the checkpoint panel.",
    ]
    caption_top = legend_y + len(batch_sizes) * 16 + 18
    height = caption_top + 20 + 15 * len(caption_lines) + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Mini-Batched V2 Efficiency Tradeoffs</text>',
    ]
    for row_idx, (x_key, y_key, row_title, x_label, y_label, x_lo, x_hi, y_lo, y_hi) in enumerate(panel_specs):
        for col_idx, ckpt in enumerate(checkpoints):
            left = 20 + col_idx * (PANEL_W + 8)
            top = 50 + row_idx * (PANEL_H + 12)
            subset = [row for row in batched_rows if int(row["checkpoint_step"]) == ckpt]
            parts.append(draw_axes(left, top, f"{row_title} @ ckpt {ckpt}", x_label, y_label, x_lo, x_hi, y_lo, y_hi, PANEL_W, PANEL_H))
            for batch_size in batch_sizes:
                color = BATCH_SIZE_COLORS.get(batch_size, BLUE)
                batch_subset = [row for row in subset if int(row["batch_size"]) == batch_size]
                if not batch_subset:
                    continue
                for row in batch_subset:
                    x = float(row[x_key])
                    y = float(row[y_key])
                    px, py = map_point(x, y, x_lo, x_hi, y_lo, y_hi, left, top, PANEL_W, PANEL_H)
                    parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.0" fill="{color}" opacity="0.34" />')
                mean_x = statistics.mean(float(row[x_key]) for row in batch_subset)
                mean_y = statistics.mean(float(row[y_key]) for row in batch_subset)
                px, py = map_point(mean_x, mean_y, x_lo, x_hi, y_lo, y_hi, left, top, PANEL_W, PANEL_H)
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.0" fill="{color}" stroke="{AXIS}" stroke-width="1.1" opacity="0.98" />')

    legend_x = width - 190
    for idx, batch_size in enumerate(batch_sizes):
        color = BATCH_SIZE_COLORS.get(batch_size, BLUE)
        yy = legend_y + idx * 16
        parts.append(f'<circle cx="{legend_x:.2f}" cy="{yy:.2f}" r="4.2" fill="{color}" />')
        parts.append(
            f'<text x="{legend_x + 10:.2f}" y="{yy + 4:.2f}" font-size="10" fill="{AXIS}">batch {batch_size} (n={batch_counts.get(batch_size, 0)})</text>'
        )
    parts.append(caption_block(20, caption_top, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_removal_goodput_curves(
    curve_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    v2_size: int,
    batch_size: int,
    requested_num_samples: int,
    output_path: Path,
    title_prefix: str,
) -> None:
    panel_w = 400
    panel_h = 280
    width = len(REMOVAL_CRITERIA) * panel_w + (len(REMOVAL_CRITERIA) - 1) * 10 + 40
    removal_values = [float(row["removed_pct"]) for row in curve_rows]
    goodput_values = [float(row["mean_mini_batched_goodput"]) for row in curve_rows]
    checkpoints = sorted({int(row["checkpoint_step"]) for row in curve_rows})
    x_lo, x_hi = min(removal_values), max(removal_values)
    y_lo, y_hi = bounds(goodput_values, clamp_zero=True)
    legend_y = panel_h + 86
    best_by_metric = {
        ranking_metric: max(
            [row for row in summary_rows if str(row["ranking_metric"]) == ranking_metric],
            key=lambda row: float(row["absolute_gain"] or 0.0),
        )
        for _ranking_field, ranking_metric, _title in REMOVAL_CRITERIA
    }
    caption_lines = [
        f"Shared V2 pool size = {v2_size}. At each checkpoint, examples are ranked by the chosen per-example metric and the lowest k are removed before sampling retained bsz={batch_size} batches.",
        f"Each point is the mean mini-batched goodput over up to {requested_num_samples} unique sampled retained-pool subsets. The x-axis stops at 50% removal because V2=32 and bsz={batch_size}.",
    ]
    for _ranking_field, ranking_metric, title in REMOVAL_CRITERIA:
        best_row = best_by_metric[ranking_metric]
        caption_lines.append(
            f"{title}: best gain is ckpt {int(best_row['checkpoint_step'])} at {float(best_row['best_removed_pct']):.1f}% removed, delta {float(best_row['absolute_gain'] or 0.0):+.4f}."
        )
    caption_top = legend_y + len(checkpoints) * 16 + 18
    height = caption_top + 20 + 15 * len(caption_lines) + 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Mini-Batch Goodput vs Removed Data</text>',
    ]

    for panel_index, (_ranking_field, ranking_metric, title) in enumerate(REMOVAL_CRITERIA):
        left = 20 + panel_index * (panel_w + 10)
        top = 50
        parts.append(
            draw_axes(
                left,
                top,
                title,
                "removed data (%)",
                f"mean mini_batched_goodput @ bsz {batch_size}",
                x_lo,
                x_hi,
                y_lo,
                y_hi,
                panel_w,
                panel_h,
            )
        )
        metric_rows = [row for row in curve_rows if str(row["ranking_metric"]) == ranking_metric]
        for ckpt in checkpoints:
            color = CKPT_COLORS.get(ckpt, BLUE)
            line_rows = sorted(
                [row for row in metric_rows if int(row["checkpoint_step"]) == ckpt],
                key=lambda row: int(row["removed_count"]),
            )
            points = [
                map_point(
                    float(row["removed_pct"]),
                    float(row["mean_mini_batched_goodput"]),
                    x_lo,
                    x_hi,
                    y_lo,
                    y_hi,
                    left,
                    top,
                    panel_w,
                    panel_h,
                )
                for row in line_rows
            ]
            coords = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{coords}" />')
            for px, py in points:
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.2" fill="{color}" opacity="0.92" />')
            best_row = max(line_rows, key=lambda row: float(row["mean_mini_batched_goodput"]))
            best_px, best_py = map_point(
                float(best_row["removed_pct"]),
                float(best_row["mean_mini_batched_goodput"]),
                x_lo,
                x_hi,
                y_lo,
                y_hi,
                left,
                top,
                panel_w,
                panel_h,
            )
            parts.append(f'<circle cx="{best_px:.2f}" cy="{best_py:.2f}" r="5.4" fill="none" stroke="{color}" stroke-width="1.6" />')

    legend_x = 40
    for idx, ckpt in enumerate(checkpoints):
        color = CKPT_COLORS.get(ckpt, BLUE)
        yy = legend_y + idx * 16
        parts.append(f'<circle cx="{legend_x:.2f}" cy="{yy:.2f}" r="4.0" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 10:.2f}" y="{yy + 4:.2f}" font-size="10" fill="{AXIS}">checkpoint {ckpt}</text>')

    parts.append(caption_block(20, caption_top, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_stat_efficiency_correlations(
    rows: list[dict[str, str]],
    correlation_rows: list[dict[str, object]],
    output_path: Path,
    title_prefix: str,
) -> None:
    panel_w = 400
    panel_h = 280
    width = 3 * panel_w + 60
    gradient_values = [float(row["gradient_statistical_efficiency"]) for row in rows]
    dapo_values = [float(row["dapo_statistical_efficiency"]) for row in rows]
    pods_values = [float(row["pods_statistical_efficiency"]) for row in rows]
    grad_lo, grad_hi = bounds(gradient_values, clamp_zero=True)
    dapo_lo, dapo_hi = bounds(dapo_values, clamp_zero=True)
    pods_lo, pods_hi = bounds(pods_values, clamp_zero=True)
    overall_row = next(row for row in correlation_rows if str(row["checkpoint_step"]) == "overall")

    panel_specs = [
        (
            "gradient_statistical_efficiency",
            "dapo_statistical_efficiency",
            "gradient vs DAPO-partial",
            "gradient_statistical_efficiency (Gain / sample)",
            "dapo_partial_indicator",
            grad_lo,
            grad_hi,
            dapo_lo,
            dapo_hi,
            "pearson_gradient_vs_dapo_partial",
        ),
        (
            "gradient_statistical_efficiency",
            "pods_statistical_efficiency",
            "gradient vs PODS",
            "gradient_statistical_efficiency (Gain / sample)",
            "pods_statistical_efficiency",
            grad_lo,
            grad_hi,
            pods_lo,
            pods_hi,
            "pearson_gradient_vs_pods",
        ),
        (
            "dapo_statistical_efficiency",
            "pods_statistical_efficiency",
            "DAPO-partial vs PODS",
            "dapo_partial_indicator",
            "pods_statistical_efficiency",
            dapo_lo,
            dapo_hi,
            pods_lo,
            pods_hi,
            "pearson_dapo_partial_vs_pods",
        ),
    ]
    checkpoints = sorted({int(row["checkpoint_step"]) for row in rows})
    legend_y = panel_h + 86
    caption_lines = [
        "PODS statistical efficiency is defined here as p_acc * (1 - p_acc), where p_acc is group_accuracy.",
        "DAPO-partial is 1[0 < group_accuracy < 1]. Gradient statistical efficiency is the projection onto the reference gradient.",
        "The dashed red line in each panel is an overall least-squares fit across all checkpoints.",
    ]
    caption_top = legend_y + len(checkpoints) * 16 + 18
    height = caption_top + 20 + 15 * len(caption_lines) + 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Stat Efficiency Correlations</text>',
    ]
    for panel_index, (x_key, y_key, title, x_label, y_label, x_lo, x_hi, y_lo, y_hi, corr_key) in enumerate(panel_specs):
        left = 20 + panel_index * (panel_w + 10)
        top = 50
        parts.append(draw_axes(left, top, title, x_label, y_label, x_lo, x_hi, y_lo, y_hi, panel_w, panel_h))
        xs = [float(row[x_key]) for row in rows]
        ys = [float(row[y_key]) for row in rows]
        slope, intercept = linear_fit(xs, ys)
        x1, y1 = x_lo, slope * x_lo + intercept
        x2, y2 = x_hi, slope * x_hi + intercept
        px1, py1 = map_point(x1, y1, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
        px2, py2 = map_point(x2, y2, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
        parts.append(
            f'<line x1="{px1:.2f}" y1="{py1:.2f}" x2="{px2:.2f}" y2="{py2:.2f}" stroke="{RED}" stroke-width="2.2" stroke-dasharray="6 4" />'
        )
        parts.append(
            f'<text x="{left + panel_w - 14:.2f}" y="{top + 22:.2f}" text-anchor="end" font-size="10" fill="{RED}">Pearson = {float(overall_row[corr_key]):.3f}</text>'
        )
        for row in rows:
            ckpt = int(row["checkpoint_step"])
            x = float(row[x_key])
            y = float(row[y_key])
            px, py = map_point(x, y, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" fill="{CKPT_COLORS.get(ckpt, BLUE)}" opacity="0.82" />')

    legend_x = 40
    for idx, ckpt in enumerate(checkpoints):
        color = CKPT_COLORS.get(ckpt, BLUE)
        yy = legend_y + idx * 16
        parts.append(f'<circle cx="{legend_x:.2f}" cy="{yy:.2f}" r="4.0" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 10:.2f}" y="{yy + 4:.2f}" font-size="10" fill="{AXIS}">checkpoint {ckpt}</text>')

    parts.append(caption_block(20, caption_top, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_correlation(rows: list[dict[str, str]], correlations: list[dict[str, object]], output_path: Path, title_prefix: str) -> None:
    grad_values = [float(row["gradient_statistical_efficiency"]) for row in rows]
    x_lo, x_hi = bounds(grad_values, clamp_zero=True)
    y_lo, y_hi = (-0.2, 1.2)
    panel_w = 540
    panel_h = 280
    width = 2 * panel_w + 60
    height = panel_h + 150
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Gradient vs DAPO Correlation</text>',
    ]
    parts.append(
        draw_axes(
            20,
            50,
            "Gradient Efficiency vs DAPO-partial",
            "gradient_statistical_efficiency",
            "dapo_partial_indicator",
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            panel_w,
            panel_h,
        )
    )
    xs = [float(row["gradient_statistical_efficiency"]) for row in rows]
    ys = [float(row["dapo_statistical_efficiency"]) for row in rows]
    slope, intercept = linear_fit(xs, ys)
    x1, y1 = x_lo, slope * x_lo + intercept
    x2, y2 = x_hi, slope * x_hi + intercept
    px1, py1 = map_point(x1, y1, x_lo, x_hi, y_lo, y_hi, 20, 50, panel_w, panel_h)
    px2, py2 = map_point(x2, y2, x_lo, x_hi, y_lo, y_hi, 20, 50, panel_w, panel_h)
    parts.append(
        f'<line x1="{px1:.2f}" y1="{py1:.2f}" x2="{px2:.2f}" y2="{py2:.2f}" stroke="{RED}" stroke-width="2.4" stroke-dasharray="6 4" />'
    )
    overall_row = next(row for row in correlations if str(row["checkpoint_step"]) == "overall")
    parts.append(
        f'<text x="{20 + panel_w - 14:.2f}" y="{50 + 22:.2f}" text-anchor="end" font-size="10" fill="{RED}">y = {slope:.2f}x + {intercept:.3f}; Pearson = {float(overall_row["pearson_gradient_vs_dapo_partial"]):.3f}</text>'
    )
    for idx, row in enumerate(rows):
        x = float(row["gradient_statistical_efficiency"])
        y = float(row["dapo_statistical_efficiency"])
        jitter = ((idx % 5) - 2) * 0.03
        ckpt = int(row["checkpoint_step"])
        px, py = map_point(x, y + jitter, x_lo, x_hi, y_lo, y_hi, 20, 50, panel_w, panel_h)
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.6" fill="{CKPT_COLORS.get(ckpt, BLUE)}" opacity="0.82" />')

    bar_left = 40 + panel_w
    bar_top = 50
    bar_w = panel_w
    bar_h = panel_h
    corr_values = [float(row["pearson_gradient_vs_dapo_partial"]) for row in correlations]
    corr_lo, corr_hi = bounds(corr_values, clamp_zero=True)
    parts.append(
        draw_axes(
            bar_left,
            bar_top,
            "Pearson Correlation by Checkpoint",
            "checkpoint",
            "corr(grad, dapo_partial)",
            -0.5,
            len(correlations) - 0.5,
            corr_lo,
            corr_hi,
            bar_w,
            bar_h,
        )
    )
    for idx, row in enumerate(correlations):
        corr = float(row["pearson_gradient_vs_dapo_partial"])
        px0, py0 = map_point(idx, 0.0, -0.5, len(correlations) - 0.5, corr_lo, corr_hi, bar_left, bar_top, bar_w, bar_h)
        _, py1 = map_point(idx, corr, -0.5, len(correlations) - 0.5, corr_lo, corr_hi, bar_left, bar_top, bar_w, bar_h)
        bar_x = px0 - 18
        bar_y = min(py0, py1)
        bar_height = abs(py1 - py0)
        label = str(row["checkpoint_step"])
        color = RED if label == "overall" else CKPT_COLORS.get(int(label), BLUE)
        parts.append(f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="36" height="{bar_height:.2f}" fill="{color}" opacity="0.85" />')
        parts.append(f'<text x="{px0:.2f}" y="{bar_top + panel_h - 20:.2f}" text-anchor="middle" font-size="10" fill="{AXIS}">{html.escape(label)}</text>')
    caption_lines = [
        "Because DAPO-partial is binary, the Pearson value here is a point-biserial correlation in practice.",
        "The dashed red line is an overall least-squares fit; with a binary y-axis it should be read as a linear probability approximation.",
        "Positive correlation means partially solved examples also tend to have larger positive gradient alignment; near zero means the two criteria disagree.",
    ]
    parts.append(caption_block(20, panel_h + 70, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_trajectories(rows: list[dict[str, str]], output_path: Path, title_prefix: str) -> None:
    by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_example[row["example_key"]].append(row)
    for values in by_example.values():
        values.sort(key=lambda row: int(row["checkpoint_step"]))

    checkpoints = sorted({int(row["checkpoint_step"]) for row in rows})
    x_lo, x_hi = min(checkpoints), max(checkpoints)
    metric_specs = [
        ("gradient_statistical_efficiency", "Gradient Stat Efficiency", "gradient_statistical_efficiency (Gain / sample)", BLUE),
        ("computational_efficiency", "Computational Efficiency", "computational_efficiency (sample / s)", GREEN),
        ("dapo_statistical_efficiency", "DAPO-partial Efficiency", "dapo_partial_indicator", ORANGE),
    ]
    width = 3 * 410 + 60
    panel_w = 410
    panel_h = 280
    height = panel_h + 130
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Per-Example Metric Change vs Checkpoint</text>',
    ]
    for panel_idx, (metric_key, title, y_label, mean_color) in enumerate(metric_specs):
        left = 20 + panel_idx * (panel_w + 10)
        top = 50
        values = [float(row[metric_key]) for row in rows]
        if metric_key == "dapo_statistical_efficiency":
            y_lo, y_hi = -0.1, 1.1
        else:
            y_lo, y_hi = bounds(values, clamp_zero=True)
        parts.append(
            draw_axes(
                left,
                top,
                title,
                "checkpoint_step",
                y_label,
                x_lo,
                x_hi,
                y_lo,
                y_hi,
                panel_w,
                panel_h,
            )
        )
        mean_by_ckpt = {}
        for ckpt in checkpoints:
            ckpt_vals = [float(row[metric_key]) for row in rows if int(row["checkpoint_step"]) == ckpt]
            mean_by_ckpt[ckpt] = statistics.mean(ckpt_vals)
        for ex_idx, example_key in enumerate(sorted(by_example)):
            subset = by_example[example_key]
            points = []
            for row in subset:
                x = int(row["checkpoint_step"])
                y = float(row[metric_key])
                points.append(map_point(x, y, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h))
            coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(f'<polyline fill="none" stroke="#7f8c8d" stroke-opacity="0.17" stroke-width="1.4" points="{coords}" />')
        mean_points = [map_point(ckpt, mean_by_ckpt[ckpt], x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h) for ckpt in checkpoints]
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in mean_points)
        parts.append(f'<polyline fill="none" stroke="{mean_color}" stroke-width="3.2" points="{coords}" />')
        for x, y in mean_points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="{mean_color}" />')
    caption_lines = [
        "Thin lines are individual V2 examples tracked across checkpoints. The bold line is the checkpoint mean.",
        "Examples are the same V2 prompts at each checkpoint, so vertical movement reflects metric change rather than a different sample.",
    ]
    parts.append(caption_block(20, panel_h + 70, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_time_breakdown(rows: list[dict[str, str]], summaries: list[dict[str, object]], output_path: Path, title_prefix: str) -> None:
    checkpoints = [int(row["checkpoint_step"]) for row in summaries]
    rollout_mean = [float(row["mean_v2_rollout_cost_s"]) for row in summaries]
    grad_mean = [float(row["mean_v2_grad_cost_s"]) for row in summaries]
    total_mean = [r + g for r, g in zip(rollout_mean, grad_mean, strict=True)]
    y_lo, y_hi = bounds(total_mean, clamp_zero=True)
    width = 720
    height = 420
    panel_w = 680
    panel_h = 260
    left = 20
    top = 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Mean Time Breakdown by Checkpoint</text>',
        draw_axes(left, top, "Rollout vs Gradient Time", "checkpoint_step", "seconds", min(checkpoints) - 10, max(checkpoints) + 10, y_lo, y_hi, panel_w, panel_h),
    ]
    bar_width = 42
    for ckpt, roll, grad in zip(checkpoints, rollout_mean, grad_mean, strict=True):
        px, py0 = map_point(ckpt, 0.0, min(checkpoints) - 10, max(checkpoints) + 10, y_lo, y_hi, left, top, panel_w, panel_h)
        _, py_roll = map_point(ckpt, roll, min(checkpoints) - 10, max(checkpoints) + 10, y_lo, y_hi, left, top, panel_w, panel_h)
        _, py_total = map_point(ckpt, roll + grad, min(checkpoints) - 10, max(checkpoints) + 10, y_lo, y_hi, left, top, panel_w, panel_h)
        parts.append(f'<rect x="{px - bar_width/2:.2f}" y="{py_roll:.2f}" width="{bar_width}" height="{py0 - py_roll:.2f}" fill="{BLUE}" opacity="0.86" />')
        parts.append(f'<rect x="{px - bar_width/2:.2f}" y="{py_total:.2f}" width="{bar_width}" height="{py_roll - py_total:.2f}" fill="{ROSE}" opacity="0.86" />')
        parts.append(f'<text x="{px:.2f}" y="{py_total - 6:.2f}" text-anchor="middle" font-size="10" fill="{TEXT}">{roll + grad:.2f}s</text>')
    legend_y = 64
    parts.append(f'<rect x="520" y="{legend_y - 9}" width="12" height="12" fill="{BLUE}" />')
    parts.append(f'<text x="538" y="{legend_y + 1}" font-size="11" fill="{AXIS}">mean rollout cost</text>')
    parts.append(f'<rect x="520" y="{legend_y + 11}" width="12" height="12" fill="{ROSE}" />')
    parts.append(f'<text x="538" y="{legend_y + 21}" font-size="11" fill="{AXIS}">mean gradient time</text>')
    caption_lines = [
        "C_roll is rollout wall time attributed from batched generation lengths. T_grad is per-group gradient/backward time.",
        "In this profiler revision C_prefill is still approximated as 0 and C_decode = C_roll, so this is a rollout-vs-other split rather than a prefill/decode decomposition.",
    ]
    parts.append(caption_block(20, panel_h + 70, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_direction_counts(direction_rows: list[dict[str, object]], output_path: Path, title_prefix: str) -> None:
    metric_titles = {
        "gradient_statistical_efficiency": "Gradient Stat Efficiency (Gain / sample)",
        "computational_efficiency": "Computational Efficiency (sample / s)",
        "dapo_statistical_efficiency": "DAPO-partial Efficiency",
    }
    panel_w = 410
    panel_h = 270
    width = 3 * panel_w + 60
    height = panel_h + 140
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Up / Down / Similar Example Shares</text>',
    ]
    metrics = [
        "gradient_statistical_efficiency",
        "computational_efficiency",
        "dapo_statistical_efficiency",
    ]
    for panel_idx, metric in enumerate(metrics):
        subset = [row for row in direction_rows if row["metric"] == metric]
        left = 20 + panel_idx * (panel_w + 10)
        top = 50
        transitions = [f'{row["from_checkpoint"]}->{row["to_checkpoint"]}' for row in subset]
        x_lo, x_hi = -0.5, len(subset) - 0.5
        y_lo, y_hi = 0.0, 100.0
        parts.append(
            draw_axes(
                left,
                top,
                metric_titles[metric],
                "checkpoint transition",
                "example share (%)",
                x_lo,
                x_hi,
                y_lo,
                y_hi,
                panel_w,
                panel_h,
            )
        )
        bar_w = 0.62
        stack_specs = [
            ("up_count", "up", GREEN),
            ("down_count", "down", RED),
            ("flat_count", "similar", "#7f8c8d"),
        ]
        for idx, row in enumerate(subset):
            cumulative = 0.0
            plot_w = panel_w - MARGIN_L - MARGIN_R
            rect_w = plot_w * (bar_w / (x_hi - x_lo))
            px_center, _ = map_point(idx, 0.0, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
            for count_key, pct_label, color in stack_specs:
                pct_key = count_key.replace("_count", "_pct")
                value = float(row[pct_key])
                _, py_bottom = map_point(idx, cumulative, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
                cumulative += value
                _, py_top = map_point(idx, cumulative, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
                parts.append(
                    f'<rect x="{px_center - rect_w/2:.2f}" y="{py_top:.2f}" width="{rect_w:.2f}" height="{py_bottom - py_top:.2f}" fill="{color}" opacity="0.88" />'
                )
            px_tick, _ = map_point(idx, 0.0, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
            parts.append(f'<text x="{px_tick:.2f}" y="{top + panel_h - 20:.2f}" text-anchor="middle" font-size="10" fill="{AXIS}">{html.escape(transitions[idx])}</text>')
        legend_x = left + panel_w - 112
        legend_y = top + 18
        for idx, (_key, label, color) in enumerate(stack_specs):
            parts.append(f'<rect x="{legend_x:.2f}" y="{legend_y + idx*16 - 8:.2f}" width="10" height="10" fill="{color}" />')
            parts.append(f'<text x="{legend_x + 16:.2f}" y="{legend_y + idx*16:.2f}" font-size="10" fill="{AXIS}">{label}</text>')
        threshold = float(subset[0]["similarity_threshold"]) if subset else 0.0
        parts.append(
            f'<text x="{left + 14:.2f}" y="{top + 34:.2f}" font-size="10" fill="{MUTED}">similar if |delta| &lt;= {threshold:.4f}</text>'
        )
    caption_lines = [
        "Each transition is one stacked bar over the same 32 V2 examples. The y-axis is percentage share, not raw count.",
        "For each metric, similar means the checkpoint-to-checkpoint change is within 5% of that metric's observed overall range in this run.",
        "For DAPO-partial, many transitions stay similar because it is binary: examples often stay partial or stay non-partial across adjacent checkpoints.",
    ]
    parts.append(caption_block(20, panel_h + 70, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_bucket_summary(bucket_rows: list[dict[str, object]], output_path: Path, title_prefix: str) -> None:
    panel_w = 520
    panel_h = 290
    width = 2 * panel_w + 60
    height = panel_h + 150
    bucket_colors = {"solved": BLUE, "partial": ORANGE, "zero": ROSE}
    bucket_labels = {"solved": "solved", "partial": "partial", "zero": "zero-acc"}
    checkpoints = sorted({int(row["checkpoint_step"]) for row in bucket_rows})

    def by_bucket(metric_key: str) -> dict[str, list[tuple[int, float]]]:
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in bucket_rows:
            grouped[str(row["bucket"])].append((int(row["checkpoint_step"]), float(row[metric_key])))
        for values in grouped.values():
            values.sort(key=lambda pair: pair[0])
        return grouped

    grad_series = by_bucket("mean_gradient_statistical_efficiency")
    cost_series = by_bucket("mean_total_cost_s")
    grad_vals = [value for values in grad_series.values() for _, value in values]
    cost_vals = [value for values in cost_series.values() for _, value in values]
    grad_lo, grad_hi = bounds(grad_vals, clamp_zero=True)
    cost_lo, cost_hi = bounds(cost_vals, clamp_zero=True)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Accuracy-Bucket Summary</text>',
    ]

    for panel_idx, (title, metric_key, y_lo, y_hi, y_label, series_map) in enumerate(
        [
            ("Mean Gradient Efficiency by Bucket", "mean_gradient_statistical_efficiency", grad_lo, grad_hi, "mean statistical_efficiency (Gain / sample)", grad_series),
            ("Mean Total Cost by Bucket", "mean_total_cost_s", cost_lo, cost_hi, "mean total_cost_s", cost_series),
        ]
    ):
        left = 20 + panel_idx * (panel_w + 20)
        top = 50
        x_lo, x_hi = min(checkpoints), max(checkpoints)
        parts.append(draw_axes(left, top, title, "checkpoint_step", y_label, x_lo, x_hi, y_lo, y_hi, panel_w, panel_h))
        for bucket in ["solved", "partial", "zero"]:
            values = series_map.get(bucket, [])
            if not values:
                continue
            points = [
                map_point(ckpt, val, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
                for ckpt, val in values
            ]
            coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            color = bucket_colors[bucket]
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3.0" points="{coords}" />')
            for (ckpt, val), (x, y) in zip(values, points, strict=True):
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.0" fill="{color}" />')
                parts.append(f'<text x="{x:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="9" fill="{color}">{val:.3f}</text>')
        legend_x = left + panel_w - 112
        legend_y = top + 18
        for idx, bucket in enumerate(["solved", "partial", "zero"]):
            color = bucket_colors[bucket]
            parts.append(f'<line x1="{legend_x:.2f}" y1="{legend_y + idx*16:.2f}" x2="{legend_x + 14:.2f}" y2="{legend_y + idx*16:.2f}" stroke="{color}" stroke-width="3" />')
            parts.append(f'<text x="{legend_x + 20:.2f}" y="{legend_y + idx*16 + 4:.2f}" font-size="10" fill="{AXIS}">{bucket_labels[bucket]}</text>')

    caption_lines = [
        "Buckets are defined per checkpoint from group_accuracy: solved = 1.0, partial = 0<a<1, zero = 0.0.",
        "This isolates the key pattern: solved and zero-accuracy groups have near-zero gradient efficiency, while partial groups carry most of the usable learning signal.",
    ]
    parts.append(caption_block(20, panel_h + 70, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_goodput_distribution(rows: list[dict[str, str]], output_path: Path, title_prefix: str) -> None:
    by_ckpt: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_ckpt[int(row["checkpoint_step"])].append(float(row["gradient_goodput"]))

    checkpoints = sorted(by_ckpt)
    all_values = [value for values in by_ckpt.values() for value in values]
    y_lo, y_hi = bounds(all_values, clamp_zero=True)
    panel_w = 640
    panel_h = 320
    width = panel_w + 40
    height = panel_h + 140
    left = 20
    top = 50
    x_lo, x_hi = float(min(checkpoints) - 10), float(max(checkpoints) + 10)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Gradient Goodput Distribution</text>',
        draw_axes(
            left,
            top,
            "goodput = statistical_efficiency * computational_efficiency",
            "checkpoint_step",
            "gradient_goodput (Gain / s)",
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            panel_w,
            panel_h,
        ),
    ]

    for idx, ckpt in enumerate(checkpoints):
        values = by_ckpt[ckpt]
        mean = statistics.mean(values)
        for point_idx, value in enumerate(values):
            jitter = ((point_idx % 7) - 3) * 0.65
            px, py = map_point(ckpt + jitter, value, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.7" fill="{CKPT_COLORS.get(ckpt, BLUE)}" opacity="0.82" />')
        mean_left, mean_y = map_point(ckpt - 2.4, mean, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
        mean_right, _ = map_point(ckpt + 2.4, mean, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
        parts.append(f'<line x1="{mean_left:.2f}" y1="{mean_y:.2f}" x2="{mean_right:.2f}" y2="{mean_y:.2f}" stroke="{AXIS}" stroke-width="2.6" />')
        parts.append(f'<text x="{mean_right + 6:.2f}" y="{mean_y + 4:.2f}" font-size="9" fill="{AXIS}">{mean:.4f}</text>')

    caption_lines = [
        "gradient_goodput (Gain/s) = gradient_statistical_efficiency (Gain/sample) * computational_efficiency (sample/s).",
        "Each dot is one V2 example at one checkpoint; the short black bar is the checkpoint mean.",
    ]
    parts.append(caption_block(20, panel_h + 70, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def render_goodput_correlations(
    rows: list[dict[str, str]],
    correlation_rows: list[dict[str, object]],
    output_path: Path,
    title_prefix: str,
) -> None:
    panel_w = 540
    panel_h = 300
    width = 2 * panel_w + 60
    height = panel_h + 150
    goodput_values = [float(row["gradient_goodput"]) for row in rows]
    goodput_lo, goodput_hi = bounds(goodput_values, clamp_zero=True)
    stat_values = [float(row["gradient_statistical_efficiency"]) for row in rows]
    stat_lo, stat_hi = bounds(stat_values, clamp_zero=True)
    comp_values = [float(row["computational_efficiency"]) for row in rows]
    comp_lo, comp_hi = bounds(comp_values, clamp_zero=True)

    overall_row = next(row for row in correlation_rows if str(row["checkpoint_step"]) == "overall")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Goodput Correlations</text>',
    ]

    panel_specs = [
        (
            20,
            "goodput vs statistical_efficiency",
            "gradient_statistical_efficiency (Gain / sample)",
            stat_lo,
            stat_hi,
            "pearson_goodput_vs_stat_eff",
            [float(row["gradient_statistical_efficiency"]) for row in rows],
        ),
        (
            40 + panel_w,
            "goodput vs computational_efficiency",
            "computational_efficiency (sample / s)",
            comp_lo,
            comp_hi,
            "pearson_goodput_vs_compute_eff",
            [float(row["computational_efficiency"]) for row in rows],
        ),
    ]

    for left, title, x_label, x_lo, x_hi, corr_key, xs in panel_specs:
        top = 50
        parts.append(
            draw_axes(
                left,
                top,
                title,
                x_label,
                "gradient_goodput (Gain / s)",
                x_lo,
                x_hi,
                goodput_lo,
                goodput_hi,
                panel_w,
                panel_h,
            )
        )
        ys = [float(row["gradient_goodput"]) for row in rows]
        slope, intercept = linear_fit(xs, ys)
        x1, y1 = x_lo, slope * x_lo + intercept
        x2, y2 = x_hi, slope * x_hi + intercept
        px1, py1 = map_point(x1, y1, x_lo, x_hi, goodput_lo, goodput_hi, left, top, panel_w, panel_h)
        px2, py2 = map_point(x2, y2, x_lo, x_hi, goodput_lo, goodput_hi, left, top, panel_w, panel_h)
        parts.append(
            f'<line x1="{px1:.2f}" y1="{py1:.2f}" x2="{px2:.2f}" y2="{py2:.2f}" stroke="{RED}" stroke-width="2.4" stroke-dasharray="6 4" />'
        )
        parts.append(
            f'<text x="{left + panel_w - 14:.2f}" y="{top + 22:.2f}" text-anchor="end" font-size="10" fill="{RED}">Pearson = {float(overall_row[corr_key]):.3f}</text>'
        )
        for row in rows:
            x = float(row["gradient_statistical_efficiency"]) if "stat" in corr_key else float(row["computational_efficiency"])
            y = float(row["gradient_goodput"])
            ckpt = int(row["checkpoint_step"])
            px, py = map_point(x, y, x_lo, x_hi, goodput_lo, goodput_hi, left, top, panel_w, panel_h)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.7" fill="{CKPT_COLORS.get(ckpt, BLUE)}" opacity="0.82" />')

    legend_x = width - 140
    legend_y = panel_h + 92
    for idx, ckpt in enumerate(sorted({int(row["checkpoint_step"]) for row in rows})):
        color = CKPT_COLORS.get(ckpt, BLUE)
        parts.append(f'<circle cx="{legend_x:.2f}" cy="{legend_y + idx * 16:.2f}" r="4" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 10:.2f}" y="{legend_y + idx * 16 + 4:.2f}" font-size="10" fill="{AXIS}">checkpoint {ckpt}</text>')

    caption_lines = [
        "gradient_goodput (Gain/s) = gradient_statistical_efficiency (Gain/sample) * computational_efficiency (sample/s).",
        "Points are colored by checkpoint; the dashed red line is an overall least-squares fit across all checkpoints.",
    ]
    parts.append(caption_block(20, panel_h + 70, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.profile_csv)
    _ = load_csv(args.rollout_detail_csv)
    checkpoint_summaries = load_json(args.checkpoint_summary_json)
    mini_batched_rows, mini_batch_counts, v2_size = build_mini_batched_rows(
        rows,
        args.mini_batch_sizes,
        args.mini_batch_samples,
        args.mini_batch_seed,
        args.output_dir / "mini_batched_v2_efficiency.csv",
    )
    removal_curve_rows, removal_summary_rows, _ = build_removal_goodput_curve_rows(
        rows,
        args.removal_batch_size,
        args.mini_batch_samples,
        args.mini_batch_seed,
        args.output_dir / f"mini_batch_goodput_vs_removed_data_batch{args.removal_batch_size}.csv",
        args.output_dir / f"mini_batch_goodput_removal_gain_summary_batch{args.removal_batch_size}.csv",
    )

    metric_summary = summarize_profile_rows(rows, args.output_dir / "checkpoint_metric_summary.csv")
    correlation_summary = summarize_correlations(rows, args.output_dir / "correlation_summary.csv")
    stat_eff_correlation_summary = summarize_stat_efficiency_correlations(
        rows,
        args.output_dir / "stat_efficiency_correlation_summary.csv",
    )
    direction_summary = summarize_direction_changes(rows, args.output_dir / "trajectory_direction_summary.csv")
    bucket_summary = summarize_accuracy_buckets(rows, args.output_dir / "accuracy_bucket_summary.csv")
    goodput_summary = summarize_goodput(rows, args.output_dir / "goodput_summary.csv")
    goodput_correlation_summary = summarize_goodput_correlations(rows, args.output_dir / "goodput_correlation_summary.csv")

    if args.stat_vs_compute_mode == "per_example":
        render_per_example_stat_vs_compute(
            rows,
            args.output_dir / "stat_vs_compute_by_checkpoint.svg",
            args.title_prefix,
        )
    else:
        render_stat_vs_compute(
            mini_batched_rows,
            mini_batch_counts,
            v2_size,
            args.mini_batch_samples,
            args.stat_vs_compute_plot_batch_sizes,
            args.output_dir / "stat_vs_compute_by_checkpoint.svg",
            args.title_prefix,
        )
    render_removal_goodput_curves(
        removal_curve_rows,
        removal_summary_rows,
        v2_size,
        args.removal_batch_size,
        args.mini_batch_samples,
        args.output_dir / f"mini_batch_goodput_vs_removed_data_batch{args.removal_batch_size}.svg",
        args.title_prefix,
    )
    render_stat_efficiency_correlations(
        rows,
        stat_eff_correlation_summary,
        args.output_dir / "stat_efficiency_correlations.svg",
        args.title_prefix,
    )
    render_correlation(rows, correlation_summary, args.output_dir / "gradient_dapo_correlation.svg", args.title_prefix)
    render_trajectories(rows, args.output_dir / "per_data_metric_trajectories.svg", args.title_prefix)
    render_direction_counts(direction_summary, args.output_dir / "trajectory_direction_counts.svg", args.title_prefix)
    render_bucket_summary(bucket_summary, args.output_dir / "accuracy_bucket_summary.svg", args.title_prefix)
    render_goodput_distribution(rows, args.output_dir / "goodput_distribution.svg", args.title_prefix)
    render_goodput_correlations(rows, goodput_correlation_summary, args.output_dir / "goodput_correlations.svg", args.title_prefix)
    render_time_breakdown(rows, checkpoint_summaries, args.output_dir / "time_breakdown_by_checkpoint.svg", args.title_prefix)

    print(args.output_dir / "stat_vs_compute_by_checkpoint.svg")
    print(args.output_dir / "gradient_dapo_correlation.svg")
    print(args.output_dir / "per_data_metric_trajectories.svg")
    print(args.output_dir / "trajectory_direction_counts.svg")
    print(args.output_dir / "accuracy_bucket_summary.svg")
    print(args.output_dir / "goodput_distribution.svg")
    print(args.output_dir / "goodput_correlations.svg")
    print(args.output_dir / "time_breakdown_by_checkpoint.svg")
    print(args.output_dir / "mini_batched_v2_efficiency.csv")
    print(args.output_dir / "stat_efficiency_correlations.svg")
    print(args.output_dir / "stat_efficiency_correlation_summary.csv")
    print(args.output_dir / f"mini_batch_goodput_vs_removed_data_batch{args.removal_batch_size}.svg")
    print(args.output_dir / f"mini_batch_goodput_vs_removed_data_batch{args.removal_batch_size}.csv")
    print(args.output_dir / f"mini_batch_goodput_removal_gain_summary_batch{args.removal_batch_size}.csv")
    print(args.output_dir / "checkpoint_metric_summary.csv")
    print(args.output_dir / "correlation_summary.csv")
    print(args.output_dir / "trajectory_direction_summary.csv")
    print(args.output_dir / "accuracy_bucket_summary.csv")
    print(args.output_dir / "goodput_summary.csv")
    print(args.output_dir / "goodput_correlation_summary.csv")


if __name__ == "__main__":
    main()
