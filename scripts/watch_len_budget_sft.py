#!/usr/bin/env python3
"""Watch a len-budget SFT run and print the latest streamed metrics."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path


STOP = False


def handle_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
    del signum, frame
    global STOP
    STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument(
        "--stop-when-complete",
        action="store_true",
        default=True,
        help="Exit automatically after the run reports state=completed.",
    )
    parser.add_argument(
        "--no-stop-when-complete",
        action="store_false",
        dest="stop_when_complete",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fmt_float(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    progress_path = args.run_dir / "progress.json"
    metrics_path = args.run_dir / "metrics.json"

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    last_payload = None
    while not STOP:
        payload = read_json(progress_path)
        if payload is not None and payload != last_payload:
            last_payload = payload
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"state: {payload.get('state', '-')}")
            print(f"phase: {payload.get('phase', '-')}")
            print(f"epoch: {payload.get('epoch', '-')}")
            print(f"global_step: {payload.get('global_step', '-')}")
            print(f"elapsed_seconds: {fmt_float(payload.get('elapsed_seconds'))}")
            if payload.get("phase") == "train":
                print(
                    "train: "
                    f"loss={fmt_float(payload.get('running_train_loss'))} "
                    f"prefix_acc={fmt_float(payload.get('running_train_prefix_token_accuracy'))} "
                    f"answer_acc={fmt_float(payload.get('running_train_answer_token_accuracy'))}"
                )
                print(
                    "latest_val: "
                    f"loss={fmt_float(payload.get('latest_val_loss'))} "
                    f"prefix_acc={fmt_float(payload.get('latest_val_prefix_token_accuracy'))} "
                    f"answer_acc={fmt_float(payload.get('latest_val_answer_token_accuracy'))}"
                )
                print(
                    "epoch_progress: "
                    f"{payload.get('epoch_batches_completed', '-')}/{payload.get('epoch_batches_total', '-')}"
                )
            else:
                print(
                    "epoch_metrics: "
                    f"train_loss={fmt_float(payload.get('train_loss'))} "
                    f"val_loss={fmt_float(payload.get('val_loss'))} "
                    f"val_prefix_acc={fmt_float(payload.get('val_prefix_token_accuracy'))} "
                    f"val_answer_acc={fmt_float(payload.get('val_answer_token_accuracy'))}"
                )
            print(f"run_dir: {args.run_dir}")
            print()
            sys.stdout.flush()

            if payload.get("state") == "completed" and args.stop_when_complete:
                break

        if payload is None and metrics_path.exists() and args.stop_when_complete:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
