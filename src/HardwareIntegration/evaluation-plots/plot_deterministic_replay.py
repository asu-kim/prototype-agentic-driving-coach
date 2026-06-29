#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import DATA_ROOT, NORMAL_TRACE, OUT_ROOT, PALETTE, apply_style, read_csv, save_figure, write_summary


IGNORE_COLUMNS = {"physical_time_ms", "tag_time_ns", "lag_ms", "start_physical_time_ms", "stop_physical_time_ms"}
DEFAULT_FILES = [
    "driver_control_replay_values.csv",
    "driver_monitor_replay_values.csv",
    "llm_replay_values.csv",
    "planner_events.csv",
]


def values_equal(a: object, b: object, tol: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except Exception:
        return str(a) == str(b)


def compare_file(name: str, reference_dir: Path, candidate_dir: Path, tol: float) -> dict[str, object]:
    ref = read_csv(reference_dir / name, required=False)
    cand = read_csv(candidate_dir / name, required=False)
    if ref.empty or cand.empty:
        return {
            "file": name,
            "reference_rows": len(ref),
            "candidate_rows": len(cand),
            "matched_rows": 0,
            "compared_cells": 0,
            "mismatched_cells": 0,
            "missing_reference_rows": 0 if not ref.empty else len(cand),
            "missing_candidate_rows": 0 if not cand.empty else len(ref),
            "match_percent": 0.0,
        }

    key_cols = [col for col in ("event", "logical_time_ns", "microstep") if col in ref.columns and col in cand.columns]
    if not key_cols:
        key_cols = ["logical_time_ns"] if "logical_time_ns" in ref.columns and "logical_time_ns" in cand.columns else []
    if key_cols:
        ref = ref.set_index(key_cols, drop=False)
        cand = cand.set_index(key_cols, drop=False)
        keys = ref.index.union(cand.index)
    else:
        ref.index = range(len(ref))
        cand.index = range(len(cand))
        keys = ref.index.union(cand.index)

    common_cols = [col for col in ref.columns if col in cand.columns and col not in IGNORE_COLUMNS and col not in key_cols]
    compared = 0
    mismatched = 0
    matched_rows = 0
    missing_ref = 0
    missing_cand = 0
    for key in keys:
        if key not in ref.index:
            missing_ref += 1
            continue
        if key not in cand.index:
            missing_cand += 1
            continue
        ref_row = ref.loc[key]
        cand_row = cand.loc[key]
        if isinstance(ref_row, pd.DataFrame):
            ref_row = ref_row.iloc[0]
        if isinstance(cand_row, pd.DataFrame):
            cand_row = cand_row.iloc[0]
        matched_rows += 1
        for col in common_cols:
            compared += 1
            if not values_equal(ref_row[col], cand_row[col], tol):
                mismatched += 1

    match_percent = 100.0 if compared == 0 else 100.0 * (compared - mismatched) / compared
    return {
        "file": name,
        "reference_rows": len(ref),
        "candidate_rows": len(cand),
        "matched_rows": matched_rows,
        "compared_cells": compared,
        "mismatched_cells": mismatched,
        "missing_reference_rows": missing_ref,
        "missing_candidate_rows": missing_cand,
        "match_percent": match_percent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot deterministic replay agreement across executions.")
    parser.add_argument("--reference-dir", type=Path, default=DATA_ROOT / NORMAL_TRACE)
    parser.add_argument("--candidate-dir", type=Path, default=Path("fed-gen/HardwareIntegratedADCReplay/logs-trace"))
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT / "deterministic-replay")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    apply_style()
    rows = [compare_file(name, args.reference_dir, args.candidate_dir, args.tol) for name in args.files]
    write_summary(args.out_dir / "deterministic_replay_summary.csv", rows)

    labels = [Path(row["file"]).stem.replace("_", "\n") for row in rows]
    match = [float(row["match_percent"]) for row in rows]
    mismatches = [int(row["mismatched_cells"]) + int(row["missing_candidate_rows"]) + int(row["missing_reference_rows"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(labels, match, color=PALETTE["Reference"], alpha=0.75)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Matching compared cells (%)")
    axes[0].set_title("Replay value agreement")
    axes[0].tick_params(axis="x", rotation=0)

    axes[1].bar(labels, mismatches, color=PALETTE["Replay"], alpha=0.75)
    axes[1].set_ylabel("Mismatched or missing items")
    axes[1].set_title("Non-deterministic differences")
    axes[1].tick_params(axis="x", rotation=0)

    save_figure(fig, args.out_dir, "deterministic_replay")
    print(f"Wrote {args.out_dir / 'deterministic_replay.png'}")
    print(f"Wrote {args.out_dir / 'deterministic_replay_summary.csv'}")


if __name__ == "__main__":
    main()
