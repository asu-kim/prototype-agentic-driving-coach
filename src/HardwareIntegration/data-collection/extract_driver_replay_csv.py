#!/usr/bin/env python3
"""Extract DriverReplayInput CSVs from logged driver input traces.

DriverReplayInput.lf expects these two files in DRIVER_REPLAY_LOG_DIR:

  - driver_control_replay_values.csv
  - driver_monitor_replay_values.csv

This script reads the logged driver_control_inputs.csv and
driver_monitor_inputs.csv files and writes exactly the columns consumed by
DriverReplayInput.lf. It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "fed-gen/HardwareIntegratedADC/logs-trace"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/HardwareIntegration/data-collection/logs-trace-2"

CONTROL_COLUMNS = [
    "logical_time_ns",
    "steer_present",
    "accelerator_present",
    "brake_present",
    "steer",
    "accelerator",
    "brake",
]

MONITOR_COLUMNS = [
    "logical_time_ns",
    "head_present",
    "eye_present",
    "head",
    "eye",
]


def to_int(value: str | None, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def read_rows(path: Path, event: str | None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if event is None or event == "all":
        return rows
    return [row for row in rows if str(row.get("event", "")) == event]


def require_columns(rows: list[dict[str, str]], path: Path, columns: list[str]) -> None:
    if not rows:
        return
    missing = [col for col in columns if col not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def write_replay_csv(rows: list[dict[str, str]], out_path: Path, columns: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: to_int(row.get("logical_time_ns"), 0))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    print(f"Wrote {len(rows)} rows to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DriverReplayInput CSVs from driver input logs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--control-input", type=Path, default=None)
    parser.add_argument("--monitor-input", type=Path, default=None)
    parser.add_argument("--control-event", default="output_tick", help="Event to extract from driver_control_inputs.csv. Use 'all' to keep all rows.")
    parser.add_argument("--monitor-event", default="output_tick", help="Event to extract from driver_monitor_inputs.csv. Use 'all' to keep all rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_path = args.control_input or args.input_dir / "driver_control_inputs.csv"
    monitor_path = args.monitor_input or args.input_dir / "driver_monitor_inputs.csv"

    control_rows = read_rows(control_path, args.control_event)
    monitor_rows = read_rows(monitor_path, args.monitor_event)

    require_columns(control_rows, control_path, CONTROL_COLUMNS)
    require_columns(monitor_rows, monitor_path, MONITOR_COLUMNS)

    write_replay_csv(
        control_rows,
        args.output_dir / "driver_control_replay_values.csv",
        CONTROL_COLUMNS,
    )
    write_replay_csv(
        monitor_rows,
        args.output_dir / "driver_monitor_replay_values.csv",
        MONITOR_COLUMNS,
    )


if __name__ == "__main__":
    main()
