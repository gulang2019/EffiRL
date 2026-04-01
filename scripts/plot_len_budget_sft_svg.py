#!/usr/bin/env python3
"""Render len-budget SFT training logs into a lightweight SVG curve plot."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from pathlib import Path


W = 980
H = 420
MARGIN_L = 72
MARGIN_R = 24
MARGIN_T = 42
MARGIN_B = 52
TRAIN_RE = re.compile(r"\[train epoch (\d+) step (\d+)/(\d+)\] loss=([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    return parser.parse_args()


def parse_train_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TRAIN_RE.search(line)
        if not match:
            continue
        epoch = int(match.group(1))
        step = int(match.group(2))
        epoch_total = int(match.group(3))
        global_step = (epoch - 1) * epoch_total + step
        loss = float(match.group(4))
        rows.append(
            {
                "epoch": float(epoch),
                "step": float(step),
                "epoch_total": float(epoch_total),
                "global_step": float(global_step),
                "loss": loss,
            }
        )
    return rows


def parse_val_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        "epoch": float(row["epoch"]),
                        "global_step": float(row["global_step"]),
                        "val_loss": float(row["val_loss"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.1
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


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


def map_point(x: float, y: float, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> tuple[float, float]:
    plot_w = W - MARGIN_L - MARGIN_R
    plot_h = H - MARGIN_T - MARGIN_B
    px = MARGIN_L + (x - x_lo) / (x_hi - x_lo) * plot_w
    py = MARGIN_T + plot_h - (y - y_lo) / (y_hi - y_lo) * plot_h
    return px, py


def polyline(points: list[tuple[float, float]], color: str, width: float) -> str:
    if len(points) < 2:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{coords}" />'
    )


def circles(points: list[tuple[float, float]], color: str, radius: float) -> str:
    return "".join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" />' for x, y in points)


def main() -> None:
    args = parse_args()
    train_rows = parse_train_log(args.run_dir / "train.stdout.log")
    if not train_rows:
        raise SystemExit("no train loss points found in train.stdout.log")
    val_rows = parse_val_csv(args.run_dir / "training_log.csv")

    x_values = [row["global_step"] for row in train_rows]
    y_values = [row["loss"] for row in train_rows]
    val_x = [row["global_step"] for row in val_rows]
    val_y = [row["val_loss"] for row in val_rows]

    x_lo, x_hi = bounds(x_values + (val_x if val_x else []))
    y_lo, y_hi = bounds(y_values + (val_y if val_y else []))
    if x_lo == x_hi:
        x_hi = x_lo + 1.0
    if y_lo == y_hi:
        y_hi = y_lo + 1.0

    plot_w = W - MARGIN_L - MARGIN_R
    plot_h = H - MARGIN_T - MARGIN_B
    x0 = MARGIN_L
    y0 = MARGIN_T + plot_h

    train_points = [map_point(row["global_step"], row["loss"], x_lo, x_hi, y_lo, y_hi) for row in train_rows]
    val_points = [map_point(row["global_step"], row["val_loss"], x_lo, x_hi, y_lo, y_hi) for row in val_rows]

    first_loss = train_rows[0]["loss"]
    last_loss = train_rows[-1]["loss"]
    min_loss = min(y_values)
    summary = f"train loss {first_loss:.4f} -> {last_loss:.4f} | min {min_loss:.4f} | points {len(train_rows)}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#fffdf8" />',
        f'<text x="20" y="26" font-size="20" font-weight="700" fill="#222">Len-Budget SFT Loss Curve</text>',
        f'<text x="20" y="46" font-size="12" fill="#555">{html.escape(summary)}</text>',
        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{MARGIN_L + plot_w:.2f}" y2="{y0:.2f}" stroke="#555" stroke-width="1" />',
        f'<line x1="{x0:.2f}" y1="{MARGIN_T:.2f}" x2="{x0:.2f}" y2="{y0:.2f}" stroke="#555" stroke-width="1" />',
    ]

    for tick in ticks(y_lo, y_hi):
        _, py = map_point(x_lo, tick, x_lo, x_hi, y_lo, y_hi)
        parts.append(
            f'<line x1="{x0:.2f}" y1="{py:.2f}" x2="{MARGIN_L + plot_w:.2f}" y2="{py:.2f}" stroke="#ece7de" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{MARGIN_L - 8}" y="{py + 4:.2f}" text-anchor="end" font-size="11" fill="#555">{html.escape(fmt_tick(tick))}</text>'
        )

    for tick in ticks(x_lo, x_hi):
        px, _ = map_point(tick, y_lo, x_lo, x_hi, y_lo, y_hi)
        parts.append(
            f'<line x1="{px:.2f}" y1="{MARGIN_T:.2f}" x2="{px:.2f}" y2="{y0:.2f}" stroke="#f3eee7" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{px:.2f}" y="{H - 18:.2f}" text-anchor="middle" font-size="11" fill="#555">{html.escape(fmt_tick(tick))}</text>'
        )

    parts.append(f'<text x="{W/2:.2f}" y="{H - 6:.2f}" text-anchor="middle" font-size="12" fill="#444">global step</text>')
    parts.append(f'<text x="20" y="{H/2:.2f}" transform="rotate(-90 20,{H/2:.2f})" text-anchor="middle" font-size="12" fill="#444">loss</text>')

    parts.append(polyline(train_points, color="#b44f2a", width=2.5))
    if train_points:
        parts.append(circles([train_points[0], train_points[-1]], color="#b44f2a", radius=3.5))
    if val_points:
        parts.append(circles(val_points, color="#1f4e79", radius=4.0))

    legend_x = W - 180
    legend_y = 28
    parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+18}" y2="{legend_y}" stroke="#b44f2a" stroke-width="3" />')
    parts.append(f'<text x="{legend_x+24}" y="{legend_y+4}" font-size="11" fill="#444">train loss</text>')
    if val_points:
        parts.append(f'<circle cx="{legend_x+9}" cy="{legend_y+18}" r="4" fill="#1f4e79" />')
        parts.append(f'<text x="{legend_x+24}" y="{legend_y+22}" font-size="11" fill="#444">val loss</text>')

    parts.append("</svg>")
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
