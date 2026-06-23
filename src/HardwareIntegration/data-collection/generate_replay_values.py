#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_TRACE = Path("src/HardwareIntegration/data-collection/logs-trace-2")
START_NS_DEFAULT = int(7e9)
PERIOD_NS_DEFAULT = int(100e6)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: str | None, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def to_float_text(value: str | None, default: str = "0.0") -> str:
    if value is None or value == "":
        return default
    try:
        return str(float(value))
    except Exception:
        return default


def to_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def logical_time(row: dict[str, str]) -> int:
    return to_int(row.get("logical_time_ns"), 0)


def replay_ticks(start_ns: int, stop_ns: int, period_ns: int) -> list[int]:
    if stop_ns < start_ns:
        return []
    return list(range(start_ns, stop_ns + 1, period_ns))


def generate_control_values(trace_dir: Path, out_dir: Path, start_ns: int, period_ns: int) -> Path | None:
    rows = sorted(read_csv(trace_dir / "driver_control_inputs.csv"), key=logical_time)
    if not rows:
        return None

    stop_ns = max(logical_time(row) for row in rows)
    ticks = replay_ticks(start_ns, max(stop_ns, start_ns), period_ns)
    idx = 0
    steer = "1"
    accel = "0.0"
    brake = "0.0"
    out_rows: list[dict[str, object]] = []

    for t in ticks:
        steer_present = 0
        accel_present = 0
        brake_present = 0
        while idx < len(rows) and logical_time(rows[idx]) <= t:
            row = rows[idx]
            if to_int(row.get("steer_present"), 0):
                steer = str(to_int(row.get("steer"), to_int(steer, 1)))
                steer_present = 1
            if to_int(row.get("accelerator_present"), 0):
                accel = to_float_text(row.get("accelerator"), accel)
                accel_present = 1
            if to_int(row.get("brake_present"), 0):
                brake = to_float_text(row.get("brake"), brake)
                brake_present = 1
            idx += 1

        out_rows.append({
            "logical_time_ns": t,
            "logical_time_ms": t / 1e6,
            "steer_present": steer_present,
            "accelerator_present": accel_present,
            "brake_present": brake_present,
            "steer": steer,
            "accelerator": accel,
            "brake": brake,
        })

    path = out_dir / "driver_control_replay_values.csv"
    write_csv(path, [
        "logical_time_ns",
        "logical_time_ms",
        "steer_present",
        "accelerator_present",
        "brake_present",
        "steer",
        "accelerator",
        "brake",
    ], out_rows)
    return path


def generate_monitor_values(trace_dir: Path, out_dir: Path, start_ns: int, period_ns: int) -> Path | None:
    rows = sorted(read_csv(trace_dir / "driver_monitor_inputs.csv"), key=logical_time)
    if not rows:
        return None

    stop_ns = max(logical_time(row) for row in rows)
    ticks = replay_ticks(start_ns, max(stop_ns, start_ns), period_ns)
    idx = 0
    head = "1"
    eye = "1"
    out_rows: list[dict[str, object]] = []

    for t in ticks:
        head_present = 0
        eye_present = 0
        while idx < len(rows) and logical_time(rows[idx]) <= t:
            row = rows[idx]
            if to_int(row.get("head_present"), 0):
                head = str(to_int(row.get("head"), to_int(head, 1)))
                head_present = 1
            if to_int(row.get("eye_present"), 0):
                eye = str(to_int(row.get("eye"), to_int(eye, 1)))
                eye_present = 1
            idx += 1

        out_rows.append({
            "logical_time_ns": t,
            "logical_time_ms": t / 1e6,
            "head_present": head_present,
            "eye_present": eye_present,
            "head": head,
            "eye": eye,
        })

    path = out_dir / "driver_monitor_replay_values.csv"
    write_csv(path, [
        "logical_time_ns",
        "logical_time_ms",
        "head_present",
        "eye_present",
        "head",
        "eye",
    ], out_rows)
    return path


def generate_llm_values(trace_dir: Path, out_dir: Path) -> Path | None:
    rows = read_csv(trace_dir / "llm_behavior.csv")
    if not rows:
        return None

    out_rows = []
    for row in rows:
        event = (row.get("event") or "").strip()
        if event not in {"llm_inference", "missing_environment_state", "startup_warmup"}:
            continue
        control = (row.get("control") or "NONE").strip().upper()
        if control not in {"NONE", "WARNING", "ACTUATE"}:
            control = "NONE"
        instruction = (row.get("instruction") or "").strip()
        if control == "NONE":
            instruction = ""
        logical_ms = to_float(row.get("logical_time_ms"), to_int(row.get("logical_time_ns"), 0) / 1e6)
        physical_ms = to_float(row.get("physical_time_ms"), 0.0)
        out_rows.append({
            "event": event,
            "logical_time_ns": to_int(row.get("logical_time_ns"), 0),
            "logical_time_ms": row.get("logical_time_ms", logical_ms),
            "physical_time_ms": row.get("physical_time_ms", physical_ms),
            "lag_ms": physical_ms - logical_ms,
            "control": control,
            "instruction": instruction,
            "inference_ms": row.get("inference_ms", ""),
        })

    path = out_dir / "llm_replay_values.csv"
    write_csv(path, [
        "event",
        "logical_time_ns",
        "logical_time_ms",
        "physical_time_ms",
        "lag_ms",
        "control",
        "instruction",
        "inference_ms",
    ], out_rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic driver and LLM replay-value CSVs from original trace logs.")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE, help="Original trace directory containing driver_*_inputs.csv and llm_behavior.csv.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to --trace.")
    parser.add_argument("--start-ms", type=float, default=7000.0, help="Replay sensor start time in ms.")
    parser.add_argument("--period-ms", type=float, default=100.0, help="Replay driver sensor period in ms.")
    args = parser.parse_args()

    trace_dir = args.trace
    out_dir = args.out or trace_dir
    start_ns = int(args.start_ms * 1e6)
    period_ns = int(args.period_ms * 1e6)

    outputs = [
        generate_control_values(trace_dir, out_dir, start_ns, period_ns),
        generate_monitor_values(trace_dir, out_dir, start_ns, period_ns),
        generate_llm_values(trace_dir, out_dir),
    ]

    for path in outputs:
        if path is None:
            print("Skipped missing input for one replay-value file")
        else:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
