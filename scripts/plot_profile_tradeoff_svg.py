#!/usr/bin/env python3
"""Render per-example compute/accuracy trajectories from profiling CSV."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


PANEL_W = 620
PANEL_H = 320
MARGIN_L = 64
MARGIN_R = 18
MARGIN_T = 30
MARGIN_B = 46
COLORS = [
    "#0a6c8f",
    "#d95f02",
    "#2a9d8f",
    "#b56576",
    "#6c757d",
    "#3a86ff",
    "#7a5c61",
    "#588157",
    "#bc6c25",
    "#8338ec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Profiling CSV path.")
    parser.add_argument("--output-svg", type=Path, required=True, help="Output SVG path.")
    parser.add_argument(
        "--x-field",
        type=str,
        default="total_cost_s",
        help="CSV field used on the x-axis of the trajectory panel.",
    )
    parser.add_argument(
        "--y-field",
        type=str,
        default="reward_mean",
        help="CSV field used on the y-axis.",
    )
    parser.add_argument(
        "--time-field",
        type=str,
        default="checkpoint_step",
        help="CSV field used on the x-axis of the time panel.",
    )
    parser.add_argument("--x-label", type=str, default=None, help="Display label for x-field.")
    parser.add_argument("--y-label", type=str, default=None, help="Display label for y-field.")
    parser.add_argument("--time-label", type=str, default=None, help="Display label for time-field.")
    parser.add_argument(
        "--title",
        type=str,
        default="Per-Example Compute vs Accuracy",
        help="Figure title.",
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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_tick(value: float) -> str:
    if abs(value) >= 100 or value.is_integer():
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def bounds(values: list[float], clamp_01: bool = False) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    lo = min(values)
    hi = max(values)
    if clamp_01:
        lo = min(0.0, lo)
        hi = max(1.0, hi)
    if lo == hi:
        pad = 0.1 if lo == 0 else abs(lo) * 0.1
        return (lo - pad, hi + pad)
    pad = (hi - lo) * 0.08
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
) -> tuple[float, float]:
    plot_w = PANEL_W - MARGIN_L - MARGIN_R
    plot_h = PANEL_H - MARGIN_T - MARGIN_B
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
) -> str:
    plot_w = PANEL_W - MARGIN_L - MARGIN_R
    plot_h = PANEL_H - MARGIN_T - MARGIN_B
    x0 = left + MARGIN_L
    y0 = top + MARGIN_T + plot_h

    parts = [
        f'<rect x="{left}" y="{top}" width="{PANEL_W}" height="{PANEL_H}" rx="14" fill="#fffdf8" stroke="#d9d2c3" />',
        f'<text x="{left + 16}" y="{top + 20}" font-size="16" font-weight="600" fill="#222">{html.escape(title)}</text>',
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{left + MARGIN_L + plot_w:.2f}" y2="{y0:.2f}" stroke="#555" stroke-width="1" />',
        f'<line x1="{x0:.2f}" y1="{top + MARGIN_T:.2f}" x2="{x0:.2f}" y2="{y0:.2f}" stroke="#555" stroke-width="1" />',
    ]

    for tick in ticks(y_lo, y_hi):
        _, py = map_point(x_lo, tick, x_lo, x_hi, y_lo, y_hi, left, top)
        parts.append(
            f'<line x1="{x0:.2f}" y1="{py:.2f}" x2="{left + MARGIN_L + plot_w:.2f}" y2="{py:.2f}" stroke="#ece7de" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{left + MARGIN_L - 8}" y="{py + 4:.2f}" text-anchor="end" font-size="11" fill="#555">{html.escape(fmt_tick(tick))}</text>'
        )

    for tick in ticks(x_lo, x_hi):
        px, _ = map_point(tick, y_lo, x_lo, x_hi, y_lo, y_hi, left, top)
        parts.append(
            f'<line x1="{px:.2f}" y1="{top + MARGIN_T:.2f}" x2="{px:.2f}" y2="{y0:.2f}" stroke="#f3eee7" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{top + MARGIN_T + plot_h + 18:.2f}" text-anchor="middle" font-size="11" fill="#555">{html.escape(fmt_tick(tick))}</text>'
        )

    parts.append(
        f'<text x="{left + MARGIN_L + plot_w / 2:.2f}" y="{top + PANEL_H - 10:.2f}" text-anchor="middle" font-size="12" fill="#444">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="{left + 16:.2f}" y="{top + PANEL_H / 2:.2f}" transform="rotate(-90 {left + 16:.2f} {top + PANEL_H / 2:.2f})" text-anchor="middle" font-size="12" fill="#444">{html.escape(y_label)}</text>'
    )
    return "".join(parts)


def legend(entries: list[tuple[str, str]], x: float, y: float) -> str:
    parts: list[str] = []
    for idx, (label, color) in enumerate(entries):
        yy = y + idx * 16
        parts.append(f'<line x1="{x:.2f}" y1="{yy:.2f}" x2="{x + 16:.2f}" y2="{yy:.2f}" stroke="{color}" stroke-width="3" />')
        parts.append(
            f'<text x="{x + 22:.2f}" y="{yy + 4:.2f}" font-size="11" fill="#444">{html.escape(label)}</text>'
        )
    return "".join(parts)


def short_label(example_key: str) -> str:
    return example_key.split("::")[-1]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    if not rows:
        raise SystemExit("no profiling rows found")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["example_key"]].append(row)

    for value in grouped.values():
        value.sort(key=lambda row: int(row["checkpoint_step"]))

    examples = sorted(grouped)
    x_field = args.x_field
    y_field = args.y_field
    time_field = args.time_field

    cost_values = [parse_float(row.get(x_field)) or 0.0 for row in rows]
    acc_values = [parse_float(row.get(y_field)) or 0.0 for row in rows]
    step_values = [parse_float(row.get(time_field)) or 0.0 for row in rows]

    cost_lo, cost_hi = bounds(cost_values)
    clamp_01 = all(0.0 <= value <= 1.0 for value in acc_values)
    acc_lo, acc_hi = bounds(acc_values, clamp_01=clamp_01)
    step_lo, step_hi = bounds(step_values)
    x_label = args.x_label or x_field
    y_label = args.y_label or y_field
    time_label = args.time_label or time_field

    width = 2 * PANEL_W + 56
    height = PANEL_H + 90

    left_panel = draw_axes(
        left=20,
        top=50,
        title="Per-Example Trajectory",
        x_label=x_label,
        y_label=y_label,
        x_lo=cost_lo,
        x_hi=cost_hi,
        y_lo=acc_lo,
        y_hi=acc_hi,
    )
    right_panel = draw_axes(
        left=20 + PANEL_W + 16,
        top=50,
        title=f"{y_label} Over {time_label}",
        x_label=time_label,
        y_label=y_label,
        x_lo=step_lo,
        x_hi=step_hi,
        y_lo=acc_lo,
        y_hi=acc_hi,
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f1e8" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="#1e1e1e">{html.escape(args.title)}</text>',
        f'<text x="{width - 20}" y="28" text-anchor="end" font-size="12" fill="#666">{html.escape(args.input_csv.name)}</text>',
        left_panel,
        right_panel,
    ]

    legend_entries: list[tuple[str, str]] = []
    for idx, example_key in enumerate(examples):
        color = COLORS[idx % len(COLORS)]
        label = short_label(example_key)
        legend_entries.append((label, color))
        series = grouped[example_key]

        left_points: list[tuple[float, float, str]] = []
        right_points: list[tuple[float, float, str]] = []
        for row in series:
            cost = parse_float(row.get(x_field)) or 0.0
            acc = parse_float(row.get(y_field)) or 0.0
            step = parse_float(row.get(time_field)) or 0.0
            left_px, left_py = map_point(cost, acc, cost_lo, cost_hi, acc_lo, acc_hi, 20, 50)
            right_px, right_py = map_point(step, acc, step_lo, step_hi, acc_lo, acc_hi, 20 + PANEL_W + 16, 50)
            left_points.append((left_px, left_py, str(int(step))))
            right_points.append((right_px, right_py, str(int(step))))

        left_coords = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in left_points)
        right_coords = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in right_points)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{left_coords}" />'
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{right_coords}" />'
        )

        for point_idx, (px, py, step_label) in enumerate(left_points):
            radius = 4.5 if point_idx == len(left_points) - 1 else 3.5
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}" fill="{color}" />')
            parts.append(
                f'<text x="{px + 5:.2f}" y="{py - 6:.2f}" font-size="10" fill="{color}">{html.escape(step_label)}</text>'
            )

        for point_idx, (px, py, step_label) in enumerate(right_points):
            radius = 4.5 if point_idx == len(right_points) - 1 else 3.5
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius}" fill="{color}" />')
            parts.append(
                f'<text x="{px + 5:.2f}" y="{py - 6:.2f}" font-size="10" fill="{color}">{html.escape(step_label)}</text>'
            )

    parts.append(legend(legend_entries, x=20 + PANEL_W - 110, y=70))
    parts.append(legend(legend_entries, x=20 + PANEL_W + 16 + PANEL_W - 110, y=70))
    parts.append("</svg>")

    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote SVG to {args.output_svg}")


if __name__ == "__main__":
    main()
