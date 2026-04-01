#!/usr/bin/env python3
"""Render checkpoint-wise efficiency and length-variance dashboards from profiling CSV."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PANEL_W = 320
PANEL_H = 240
MARGIN_L = 56
MARGIN_R = 16
MARGIN_T = 28
MARGIN_B = 36
COLORS = ["#0a6c8f", "#d95f02", "#2a9d8f", "#b56576"]
BG = "#f6f1e8"
PANEL_BG = "#fffdf8"
PANEL_BORDER = "#d9d2c3"
GRID = "#ece7de"
AXIS = "#555"
TEXT = "#222"
MUTED = "#666"
RED = "#c0392b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--scatter-svg", type=Path, required=True)
    parser.add_argument("--distribution-svg", type=Path, required=True)
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="GRPO Profiling",
        help="Prefix used in figure titles.",
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
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def bounds(values: list[float], clamp_zero: bool = False) -> tuple[float, float]:
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
    panel_w: float = PANEL_W,
    panel_h: float = PANEL_H,
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
    panel_w: float = PANEL_W,
    panel_h: float = PANEL_H,
) -> str:
    plot_w = panel_w - MARGIN_L - MARGIN_R
    plot_h = panel_h - MARGIN_T - MARGIN_B
    x0 = left + MARGIN_L
    y0 = top + MARGIN_T + plot_h
    parts = [
        f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" rx="14" fill="{PANEL_BG}" stroke="{PANEL_BORDER}" />',
        f'<text x="{left + 14}" y="{top + 20}" font-size="15" font-weight="600" fill="{TEXT}">{html.escape(title)}</text>',
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


def draw_strip_panel(
    left: float,
    top: float,
    title: str,
    x_label: str,
    y_label: str,
    grouped_values: dict[int, list[float]],
    global_line: dict[int, float] | None = None,
    panel_w: float = 480,
    panel_h: float = 280,
) -> str:
    checkpoints = sorted(grouped_values)
    x_lo = float(min(checkpoints) - 10)
    x_hi = float(max(checkpoints) + 10)
    all_values = [value for values in grouped_values.values() for value in values]
    if global_line:
        all_values.extend(global_line.values())
    y_lo, y_hi = bounds(all_values, clamp_zero=True)
    parts = [draw_axes(left, top, title, x_label, y_label, x_lo, x_hi, y_lo, y_hi, panel_w, panel_h)]

    if global_line:
        line_points = []
        for checkpoint in checkpoints:
            x, y = map_point(checkpoint, global_line[checkpoint], x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
            line_points.append((x, y))
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in line_points)
        parts.append(
            f'<polyline fill="none" stroke="{RED}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{coords}" />'
        )
        for x, y in line_points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{RED}" />')

    for idx, checkpoint in enumerate(checkpoints):
        values = grouped_values[checkpoint]
        color = COLORS[idx % len(COLORS)]
        mean = statistics.mean(values)
        for point_idx, value in enumerate(values):
            jitter = ((point_idx % 5) - 2) * 5.0
            px, py = map_point(checkpoint + jitter * 0.08, value, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.6" fill="{color}" opacity="0.88" />')

        mean_left, mean_y = map_point(checkpoint - 3, mean, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
        mean_right, _ = map_point(checkpoint + 3, mean, x_lo, x_hi, y_lo, y_hi, left, top, panel_w, panel_h)
        parts.append(
            f'<line x1="{mean_left:.2f}" y1="{mean_y:.2f}" x2="{mean_right:.2f}" y2="{mean_y:.2f}" stroke="{AXIS}" stroke-width="2.5" />'
        )

    if global_line:
        parts.append(
            f'<text x="{left + panel_w - 14:.2f}" y="{top + 20:.2f}" text-anchor="end" font-size="10" fill="{RED}">red line = global variance</text>'
        )

    return "".join(parts)


def caption_block(left: float, top: float, width: float, lines: list[str]) -> str:
    parts = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{22 + 15 * len(lines)}" rx="12" fill="{PANEL_BG}" stroke="{PANEL_BORDER}" />'
    ]
    for idx, line in enumerate(lines):
        parts.append(
            f'<text x="{left + 14}" y="{top + 20 + idx * 15}" font-size="11" fill="{MUTED}">{html.escape(line)}</text>'
        )
    return "".join(parts)


def render_scatter_dashboard(rows: list[dict[str, str]], output_path: Path, title_prefix: str) -> None:
    checkpoints = sorted({int(row["checkpoint_step"]) for row in rows})
    stat_values = [parse_float(row["statistical_efficiency"]) or 0.0 for row in rows]
    comp_values = [parse_float(row["computational_efficiency"]) or 0.0 for row in rows]
    x_lo, x_hi = bounds(comp_values, clamp_zero=True)
    y_lo, y_hi = bounds(stat_values, clamp_zero=True)

    width = 4 * PANEL_W + 70
    caption_h = 104
    height = PANEL_H + 80 + caption_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Per-Checkpoint Efficiency Scatter</text>',
        f'<text x="{width - 20}" y="28" text-anchor="end" font-size="12" fill="{MUTED}">{html.escape(output_path.name)}</text>',
    ]

    for idx, checkpoint in enumerate(checkpoints):
        left = 20 + idx * (PANEL_W + 10)
        top = 50
        panel_rows = [row for row in rows if int(row["checkpoint_step"]) == checkpoint]
        parts.append(
            draw_axes(
                left,
                top,
                title=f"checkpoint {checkpoint}",
                x_label="computational_efficiency",
                y_label="statistical_efficiency",
                x_lo=x_lo,
                x_hi=x_hi,
                y_lo=y_lo,
                y_hi=y_hi,
            )
        )
        for point_idx, row in enumerate(panel_rows):
            x = parse_float(row["computational_efficiency"]) or 0.0
            y = parse_float(row["statistical_efficiency"]) or 0.0
            px, py = map_point(x, y, x_lo, x_hi, y_lo, y_hi, left, top)
            color = COLORS[point_idx % len(COLORS)]
            label = row["example_key"].split("::")[-1]
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.1" fill="{color}" />')
            parts.append(f'<text x="{px + 5:.2f}" y="{py - 5:.2f}" font-size="9" fill="{color}">{html.escape(label)}</text>')

    caption_lines = [
        "statistical_efficiency = stat_eff_proj = (g_u^T g_ref) / (||g_ref||_2 + eps)",
        "computational_efficiency = 1 / (C_roll + T_grad + eps), where C_roll is rollout cost attributed from batched generation lengths",
        "goodput = statistical_efficiency * computational_efficiency = stat_eff_proj / (total_cost_s + eps)",
        "This pilot uses GSM8K only, checkpoints 50/100/150/200, V1=4, V2=8, group_size=5, max_new_tokens=128, excluding validation_500.",
    ]
    parts.append(caption_block(20, PANEL_H + 64, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(parts), encoding="utf-8")


def render_distribution_dashboard(rows: list[dict[str, str]], output_path: Path, title_prefix: str) -> None:
    checkpoints = sorted({int(row["checkpoint_step"]) for row in rows})
    stat_by_ckpt: dict[int, list[float]] = defaultdict(list)
    comp_by_ckpt: dict[int, list[float]] = defaultdict(list)
    lenvar_by_ckpt: dict[int, list[float]] = defaultdict(list)
    global_lenvar: dict[int, float] = {}

    by_ckpt_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        checkpoint = int(row["checkpoint_step"])
        by_ckpt_rows[checkpoint].append(row)
        stat_by_ckpt[checkpoint].append(parse_float(row["statistical_efficiency"]) or 0.0)
        comp_by_ckpt[checkpoint].append(parse_float(row["computational_efficiency"]) or 0.0)
        lengths = json.loads(row["completion_lengths"])
        lenvar_by_ckpt[checkpoint].append(float(statistics.pvariance(lengths)))

    for checkpoint in checkpoints:
        all_lengths: list[int] = []
        for row in by_ckpt_rows[checkpoint]:
            all_lengths.extend(json.loads(row["completion_lengths"]))
        global_lenvar[checkpoint] = float(statistics.pvariance(all_lengths))

    panel_w = 410
    panel_h = 290
    width = 3 * panel_w + 60
    height = panel_h + 150
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}" />',
        f'<text x="20" y="28" font-size="22" font-weight="700" fill="{TEXT}">{html.escape(title_prefix)}: Distribution and Variance</text>',
    ]

    parts.append(
        draw_strip_panel(
            left=20,
            top=50,
            title="Statistical Efficiency Distribution",
            x_label="checkpoint_step",
            y_label="statistical_efficiency",
            grouped_values=stat_by_ckpt,
            panel_w=panel_w,
            panel_h=panel_h,
        )
    )
    parts.append(
        draw_strip_panel(
            left=30 + panel_w,
            top=50,
            title="Computational Efficiency Distribution",
            x_label="checkpoint_step",
            y_label="computational_efficiency",
            grouped_values=comp_by_ckpt,
            panel_w=panel_w,
            panel_h=panel_h,
        )
    )
    parts.append(
        draw_strip_panel(
            left=40 + 2 * panel_w,
            top=50,
            title="Completion-Length Variance",
            x_label="checkpoint_step",
            y_label="variance(lengths)",
            grouped_values=lenvar_by_ckpt,
            global_line=global_lenvar,
            panel_w=panel_w,
            panel_h=panel_h,
        )
    )

    caption_lines = [
        "Each dot in the first two panels is one prompt-group from V2 at a fixed checkpoint; the short black bar is the checkpoint mean.",
        "For the length panel, each dot is the within-problem variance of the 5 completion lengths for that prompt-group.",
        "The red line is the global variance over all 40 completion lengths at that checkpoint.",
        "If per-problem variance is much lower than the red line, most length spread comes from between-problem differences rather than within-problem stochasticity.",
    ]
    parts.append(caption_block(20, panel_h + 70, width - 40, caption_lines))
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    if not rows:
        raise SystemExit("no profiling rows found")

    render_scatter_dashboard(rows, args.scatter_svg, args.title_prefix)
    render_distribution_dashboard(rows, args.distribution_svg, args.title_prefix)
    print(f"Wrote {args.scatter_svg}")
    print(f"Wrote {args.distribution_svg}")


if __name__ == "__main__":
    main()
