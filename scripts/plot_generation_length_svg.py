#!/usr/bin/env python3
"""Render response-length trends over training into an SVG."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


PANEL_W = 620
PANEL_H = 260
MARGIN_L = 64
MARGIN_R = 18
MARGIN_T = 30
MARGIN_B = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
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


def load_rows(path: Path) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: parse_float(value) for key, value in row.items()})
    return rows


def ema(values: list[float | None], alpha: float = 0.18) -> list[float | None]:
    out: list[float | None] = []
    state: float | None = None
    for value in values:
        if value is None:
            out.append(state)
            continue
        if state is None:
            state = value
        else:
            state = alpha * value + (1.0 - alpha) * state
        out.append(state)
    return out


def bounds(values: list[float], clamp_zero: bool = False) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    lo = min(values)
    hi = max(values)
    if clamp_zero:
        lo = min(lo, 0.0)
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.1
        return (lo - pad, hi + pad)
    pad = (hi - lo) * 0.08
    return (lo - pad, hi + pad)


def ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def fmt_tick(value: float) -> str:
    if abs(value) >= 100 or value.is_integer():
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


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


def svg_polyline(points: list[tuple[float, float]], color: str, width: float, opacity: float = 1.0) -> str:
    if len(points) < 2:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}" points="{coords}" />'
    )


def draw_panel(
    left: float,
    top: float,
    title: str,
    x_values: list[float],
    x_label: str,
    y_label: str,
    series: list[dict[str, object]],
    clamp_zero: bool = False,
) -> str:
    x_lo = min(x_values)
    x_hi = max(x_values)
    if x_lo == x_hi:
        x_hi = x_lo + 1.0
    y_vals = [y for spec in series for y in spec["y"] if y is not None and math.isfinite(y)]  # type: ignore[index]
    y_lo, y_hi = bounds(y_vals, clamp_zero=clamp_zero)
    if y_lo == y_hi:
        y_hi = y_lo + 1.0

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
        f'<text x="{left + 14:.2f}" y="{top + PANEL_H / 2:.2f}" transform="rotate(-90 {left + 14:.2f} {top + PANEL_H / 2:.2f})" text-anchor="middle" font-size="12" fill="#444">{html.escape(y_label)}</text>'
    )

    legend_x = left + PANEL_W - 170
    legend_y = top + 18
    for idx, spec in enumerate(series):
        color = spec["color"]  # type: ignore[index]
        label = spec["label"]  # type: ignore[index]
        parts.append(
            f'<line x1="{legend_x:.2f}" y1="{legend_y + idx * 16:.2f}" x2="{legend_x + 16:.2f}" y2="{legend_y + idx * 16:.2f}" stroke="{color}" stroke-width="3" />'
        )
        parts.append(
            f'<text x="{legend_x + 22:.2f}" y="{legend_y + idx * 16 + 4:.2f}" font-size="11" fill="#444">{html.escape(str(label))}</text>'
        )

    for spec in series:
        color: str = spec["color"]  # type: ignore[index]
        width: float = spec.get("width", 2.0)  # type: ignore[assignment]
        opacity: float = spec.get("opacity", 1.0)  # type: ignore[assignment]
        y_series: list[float | None] = spec["y"]  # type: ignore[index]
        segment: list[tuple[float, float]] = []
        for x, y in zip(x_values, y_series):
            if y is None or not math.isfinite(y):
                if segment:
                    parts.append(svg_polyline(segment, color, width, opacity))
                    segment = []
                continue
            segment.append(map_point(x, y, x_lo, x_hi, y_lo, y_hi, left, top))
        if segment:
            parts.append(svg_polyline(segment, color, width, opacity))

    return "".join(parts)


def caption_block(left: float, top: float, width: float, lines: list[str]) -> str:
    height = 22 + 15 * len(lines)
    parts = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="12" fill="#fffdf8" stroke="#d9d2c3" />'
    ]
    for idx, line in enumerate(lines):
        parts.append(
            f'<text x="{left + 14}" y="{top + 20 + idx * 15}" font-size="11" fill="#666">{html.escape(line)}</text>'
        )
    return "".join(parts)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    if not rows:
        raise SystemExit("no rows found in metrics CSV")

    steps = [row.get("training/global_step") or row.get("step") or 0.0 for row in rows]
    mean_response = [row.get("response_length/mean") for row in rows]
    mean_response_ema = ema(mean_response)
    max_response = [row.get("response_length/max") for row in rows]
    slowest_response = [row.get("timing_s/agent_loop/slowest/response_length") for row in rows]

    width = 2 * PANEL_W + 56
    height = PANEL_H + 150
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f1e8" />',
        '<text x="20" y="28" font-size="22" font-weight="700" fill="#222">Generation Length Across Training</text>',
    ]

    parts.append(
        draw_panel(
            20,
            50,
            "Mean Generation Length vs Step",
            steps,
            "training/global_step",
            "tokens",
            [
                {"label": "response_length/mean", "y": mean_response, "color": "#0a6c8f", "width": 2.0},
                {"label": "EMA(mean)", "y": mean_response_ema, "color": "#d95f02", "width": 3.0},
            ],
            clamp_zero=True,
        )
    )
    parts.append(
        draw_panel(
            36 + PANEL_W,
            50,
            "Tail Indicators vs Step",
            steps,
            "training/global_step",
            "tokens",
            [
                {"label": "response_length/mean", "y": mean_response, "color": "#0a6c8f", "width": 2.0, "opacity": 0.85},
                {"label": "response_length/max", "y": max_response, "color": "#b56576", "width": 2.5},
                {"label": "slowest response", "y": slowest_response, "color": "#2a9d8f", "width": 2.0},
            ],
            clamp_zero=True,
        )
    )

    caption_lines = [
        "Mean generation length uses training metric response_length/mean from the verl step log.",
        "Tail indicators use response_length/max and timing_s/agent_loop/slowest/response_length from the same step logs.",
        "If the max or slowest curve stays high while the mean is lower, rollout cost is being driven by long-tail stragglers.",
    ]
    parts.append(caption_block(20, PANEL_H + 70, width - 40, caption_lines))
    parts.append("</svg>")

    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {args.output_svg}")


if __name__ == "__main__":
    main()
