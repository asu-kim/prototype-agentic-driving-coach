#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_ORIGINAL = Path("/home/asurite.ad.asu.edu/dprahlad/deterministic-prototyping-agenticCPS/logs-trace")
DEFAULT_REPLAY = Path("src/HardwareIntegration/data-collection/logs-trace-2-replay")
IGNORE_COLUMNS = {"physical_time_ms", "tag_time_ns"}
KEY_COLUMNS = ["event", "logical_time_ns", "microstep"]
NUMERIC_TOL = 1e-6


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def is_number(value: str) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except Exception:
        return False


def values_equal(a: str, b: str, tol: float) -> bool:
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    if is_number(a) and is_number(b):
        fa = float(a)
        fb = float(b)
        return math.isclose(fa, fb, rel_tol=tol, abs_tol=tol)
    return a == b


def row_key(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def compare_csv(name: str, original: Path, replay: Path, out_dir: Path, tol: float) -> dict[str, object]:
    orig_header, orig_rows = read_csv(original)
    replay_header, replay_rows = read_csv(replay)
    common_header = [c for c in orig_header if c in replay_header and c not in IGNORE_COLUMNS]
    key_fields = [c for c in KEY_COLUMNS if c in common_header]
    if "logical_time_ns" in common_header and "logical_time_ns" not in key_fields:
        key_fields.append("logical_time_ns")
    if not key_fields:
        key_fields = ["__row_index__"]
        for i, row in enumerate(orig_rows):
            row["__row_index__"] = str(i)
        for i, row in enumerate(replay_rows):
            row["__row_index__"] = str(i)
        common_header.append("__row_index__")

    orig_by_key = {row_key(row, key_fields): row for row in orig_rows}
    replay_by_key = {row_key(row, key_fields): row for row in replay_rows}
    all_keys = sorted(set(orig_by_key) | set(replay_by_key))

    mismatch_path = out_dir / f"{name}_mismatches.csv"
    compared_cells = 0
    mismatched_cells = 0
    missing_original = 0
    missing_replay = 0

    with mismatch_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "key", "column", "original", "replay"])
        for key in all_keys:
            a = orig_by_key.get(key)
            b = replay_by_key.get(key)
            key_text = "|".join(key)
            if a is None:
                missing_original += 1
                writer.writerow([name, key_text, "__row__", "<missing>", "present"])
                continue
            if b is None:
                missing_replay += 1
                writer.writerow([name, key_text, "__row__", "present", "<missing>"])
                continue
            for col in common_header:
                if col in key_fields or col in IGNORE_COLUMNS:
                    continue
                compared_cells += 1
                av = a.get(col, "")
                bv = b.get(col, "")
                if not values_equal(av, bv, tol):
                    mismatched_cells += 1
                    writer.writerow([name, key_text, col, av, bv])

    return {
        "file": name,
        "original_rows": len(orig_rows),
        "replay_rows": len(replay_rows),
        "matched_keys": len(set(orig_by_key) & set(replay_by_key)),
        "missing_original_rows": missing_original,
        "missing_replay_rows": missing_replay,
        "compared_cells": compared_cells,
        "mismatched_cells": mismatched_cells,
        "mismatch_csv": str(mismatch_path),
    }


def compare_control_replay(original_dir: Path, replay_dir: Path, out_dir: Path, tol: float) -> dict[str, object] | None:
    control_path = original_dir / "driver_control_replay_values.csv"
    car_path = replay_dir / "car_inputs.csv"
    _, control_rows = read_csv(control_path)
    _, car_rows = read_csv(car_path)
    if not control_rows or not car_rows:
        return None

    car_by_time = {row.get("logical_time_ns", ""): row for row in car_rows}
    field_map = {
        "steer": "steer_in",
        "accelerator": "accel_in",
        "brake": "brake_in",
        "steer_present": "steer_in_present",
        "accelerator_present": "accel_in_present",
        "brake_present": "brake_in_present",
    }
    mismatch_path = out_dir / "driver_replay_to_car_input_mismatches.csv"
    checked = 0
    mismatches = 0
    missing_car = 0
    with mismatch_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["logical_time_ns", "field", "original_replay_value", "replay_car_input"])
        for row in control_rows:
            t = row.get("logical_time_ns", "")
            car = car_by_time.get(t)
            if car is None:
                missing_car += 1
                writer.writerow([t, "__row__", "present", "<missing car input>"])
                continue
            for a_field, b_field in field_map.items():
                checked += 1
                av = row.get(a_field, "")
                bv = car.get(b_field, "")
                if not values_equal(av, bv, tol):
                    mismatches += 1
                    writer.writerow([t, a_field, av, bv])
    return {
        "file": "driver_control_replay_values_vs_replay_car_inputs",
        "original_rows": len(control_rows),
        "replay_rows": len(car_rows),
        "matched_keys": len(control_rows) - missing_car,
        "missing_original_rows": 0,
        "missing_replay_rows": missing_car,
        "compared_cells": checked,
        "mismatched_cells": mismatches,
        "mismatch_csv": str(mismatch_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original data-collection inputs against replay inputs.")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tol", type=float, default=NUMERIC_TOL)
    args = parser.parse_args()

    out_dir = args.out or (args.replay / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    shared = sorted(
        p.name for p in args.original.glob("*inputs.csv")
        if (args.replay / p.name).exists()
    )

    summaries = []
    for name in shared:
        summaries.append(compare_csv(name, args.original / name, args.replay / name, out_dir, args.tol))

    extra = compare_control_replay(args.original, args.replay, out_dir, args.tol)
    if extra:
        summaries.append(extra)

    summary_path = out_dir / "input_comparison_summary.csv"
    fields = ["file", "original_rows", "replay_rows", "matched_keys", "missing_original_rows", "missing_replay_rows", "compared_cells", "mismatched_cells", "mismatch_csv"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote summary: {summary_path}")
    for row in summaries:
        print(
            f"{row['file']}: rows {row['original_rows']} vs {row['replay_rows']}, "
            f"missing replay {row['missing_replay_rows']}, mismatched cells {row['mismatched_cells']}"
        )


if __name__ == "__main__":
    main()
