#!/usr/bin/env python3
"""Render training metrics CSV into a lightweight SVG dashboard."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


PANEL_W = 620
PANEL_H = 250
MARGIN_L = 64
MARGIN_R = 18
MARGIN_T = 30
MARGIN_B = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Extracted metrics CSV path.")
    parser.add_argument("--output-svg", type=Path, required=True, help="Output SVG path.")
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


def valid_values(series_list: list[list[float | None]]) -> list[float]:
    values: list[float] = []
    for series in series_list:
        values.extend(v for v in series if v is not None and math.isfinite(v))
    return values


def bounds(values: list[float], clamp_01: bool = False) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    lo = min(values)
    hi = max(values)
    if clamp_01:
        lo = min(lo, 0.0)
        hi = max(hi, 1.0)
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


def svg_polyline(points: list[tuple[float, float]], color: str, width: float, opacity: float = 1.0, dash: str | None = None) -> str:
    if len(points) < 2:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash_attr} points="{coords}" />'
    )


def svg_points(points: list[tuple[float, float]], color: str, radius: float = 3.5) -> str:
    return "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" />' for x, y in points
    )


def draw_panel(
    left: float,
    top: float,
    title: str,
    x_values: list[float],
    x_label: str,
    series: list[dict[str, object]],
    clamp_01: bool = False,
) -> str:
    x_lo = min(x_values)
    x_hi = max(x_values)
    if x_lo == x_hi:
        x_hi = x_lo + 1.0
    y_vals = valid_values([s["y"] for s in series])  # type: ignore[index]
    y_lo, y_hi = bounds(y_vals, clamp_01=clamp_01)
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

    legend_x = left + PANEL_W - 150
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
        y_series: list[float | None] = spec["y"]  # type: ignore[index]
        color: str = spec["color"]  # type: ignore[index]
        width: float = spec.get("width", 2.0)  # type: ignore[assignment]
        opacity: float = spec.get("opacity", 1.0)  # type: ignore[assignment]
        dash: str | None = spec.get("dash")  # type: ignore[assignment]
        show_points: bool = bool(spec.get("points", False))  # type: ignore[arg-type]

        segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        point_list: list[tuple[float, float]] = []
        for x, y in zip(x_values, y_series):
            if y is None or not math.isfinite(y):
                if current:
                    segments.append(current)
                    current = []
                continue
            px, py = map_point(x, y, x_lo, x_hi, y_lo, y_hi, left, top)
            current.append((px, py))
            point_list.append((px, py))
        if current:
            segments.append(current)

        for segment in segments:
            parts.append(svg_polyline(segment, color=color, width=width, opacity=opacity, dash=dash))
        if show_points:
            parts.append(svg_points(point_list, color=color))

    return "".join(parts)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    if not rows:
        raise SystemExit("no rows found in metrics CSV")

    steps = [row.get("training/global_step") or row.get("step") or 0.0 for row in rows]
    elapsed_minutes = [(row.get("elapsed_total_s") or 0.0) / 60.0 for row in rows]
    train_reward = [row.get("critic/rewards/mean") for row in rows]
    train_loss = [row.get("actor/loss") for row in rows]
    gsm_acc = [row.get("val-core/openai/gsm8k/acc/mean@1") for row in rows]
    math_acc = [row.get("val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1") for row in rows]

    reward_ema = ema(train_reward)
    loss_ema = ema(train_loss)

    width = 2 * PANEL_W + 56
    height = 3 * PANEL_H + 80
    latest_step = int(steps[-1])
    latest_minutes = elapsed_minutes[-1]

    panels = [
        draw_panel(
            20,
            50,
            "Train Reward vs Step",
            steps,
            "step",
            [
                {"label": "raw reward", "y": train_reward, "color": "#9fb7c9", "width": 1.5, "opacity": 0.9},
                {"label": "EMA reward", "y": reward_ema, "color": "#0a6c8f", "width": 3.0},
            ],
            clamp_01=True,
        ),
        draw_panel(
            20 + PANEL_W + 16,
            50,
            "Train Reward vs Time",
            elapsed_minutes,
            "elapsed time (min)",
            [
                {"label": "raw reward", "y": train_reward, "color": "#9fb7c9", "width": 1.5, "opacity": 0.9},
                {"label": "EMA reward", "y": reward_ema, "color": "#0a6c8f", "width": 3.0},
            ],
            clamp_01=True,
        ),
        draw_panel(
            20,
            50 + PANEL_H + 16,
            "Actor Loss vs Step",
            steps,
            "step",
            [
                {"label": "raw loss", "y": train_loss, "color": "#e2a66f", "width": 1.5, "opacity": 0.9},
                {"label": "EMA loss", "y": loss_ema, "color": "#c14d00", "width": 3.0},
            ],
        ),
        draw_panel(
            20 + PANEL_W + 16,
            50 + PANEL_H + 16,
            "Actor Loss vs Time",
            elapsed_minutes,
            "elapsed time (min)",
            [
                {"label": "raw loss", "y": train_loss, "color": "#e2a66f", "width": 1.5, "opacity": 0.9},
                {"label": "EMA loss", "y": loss_ema, "color": "#c14d00", "width": 3.0},
            ],
        ),
        draw_panel(
            20,
            50 + 2 * (PANEL_H + 16),
            "Validation Accuracy vs Step",
            steps,
            "step",
            [
                {"label": "gsm8k acc", "y": gsm_acc, "color": "#2f7d32", "width": 2.5, "points": True},
                {"label": "math acc", "y": math_acc, "color": "#7b4aa6", "width": 2.5, "points": True},
            ],
            clamp_01=True,
        ),
        draw_panel(
            20 + PANEL_W + 16,
            50 + 2 * (PANEL_H + 16),
            "Validation Accuracy vs Time",
            elapsed_minutes,
            "elapsed time (min)",
            [
                {"label": "gsm8k acc", "y": gsm_acc, "color": "#2f7d32", "width": 2.5, "points": True},
                {"label": "math acc", "y": math_acc, "color": "#7b4aa6", "width": 2.5, "points": True},
            ],
            clamp_01=True,
        ),
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f1e8" />',
        '<text x="20" y="26" font-size="22" font-weight="700" fill="#222">Qwen2.5-1.5B GRPO Training Dashboard</text>',
        (
            f'<text x="20" y="44" font-size="12" fill="#555">latest step {latest_step} | '
            f'elapsed {latest_minutes:.1f} min | source {html.escape(str(args.input_csv))}</text>'
        ),
    ]
    svg.extend(panels)
    svg.append("</svg>")

    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text("".join(svg), encoding="utf-8")
    print(f"Wrote dashboard to {args.output_svg}")


if __name__ == "__main__":
    main()
