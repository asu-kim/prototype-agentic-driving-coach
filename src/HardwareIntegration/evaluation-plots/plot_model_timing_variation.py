#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import (
    DATA_ROOT,
    MODEL_TRACES,
    OUT_ROOT,
    PALETTE,
    apply_style,
    boxplot_by_label,
    collect_tardy_rows,
    describe,
    load_execution_times,
    load_llm_behavior,
    save_figure,
    write_summary,
)


def _planner_tardy_count(tardy_df: pd.DataFrame, label: str) -> int:
    if tardy_df.empty:
        return 0
    subset = tardy_df[tardy_df["condition"] == label]
    text = (
        subset.get("reactor", pd.Series(dtype=str)).fillna("").astype(str)
        + " "
        + subset.get("reaction", pd.Series(dtype=str)).fillna("").astype(str)
    ).str.lower()
    return int(text.str.contains("planner|actionplanner").sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot timing variation across LLM models.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT / "model-timing-variation")
    parser.add_argument(
        "--maxwait-ms",
        type=float,
        default=None,
        help="Draw this maxwait value. Defaults to the largest measured worst-case inference latency.",
    )
    args = parser.parse_args()

    apply_style()
    trace_paths = {label: args.data_root / dirname for label, dirname in MODEL_TRACES.items()}
    llm_df = pd.concat(
        [load_llm_behavior(path, label) for label, path in trace_paths.items()],
        ignore_index=True,
    )
    exec_df = pd.concat(
        [load_execution_times(path, label) for label, path in trace_paths.items()],
        ignore_index=True,
    )
    tardy_df = pd.concat(
        [collect_tardy_rows(path, label) for label, path in trace_paths.items()],
        ignore_index=True,
    )

    rows = []
    for label in MODEL_TRACES:
        llm_subset = llm_df[llm_df["condition"] == label]
        rows.append({"model": label, "metric": "llm_inference_ms", **describe(llm_subset["inference_ms"])})
        rows.append({"model": label, "metric": "llm_lag_ms", **describe(llm_subset["lag_ms"])})
        exec_subset = exec_df[exec_df["condition"] == label]
        planner_subset = exec_subset[exec_subset["federate"].astype(str).str.contains("planner", case=False, na=False)]
        rows.append({"model": label, "metric": "planner_lag_ms", **describe(planner_subset.get("lag_ms", []))})
        rows.append({"model": label, "metric": "planner_execution_time_ms", **describe(planner_subset.get("execution_time_ms", []))})
        rows.append({
            "model": label,
            "metric": "planner_tardy_rows",
            "count": _planner_tardy_count(tardy_df, label),
            "mean": "",
            "median": "",
            "std": "",
            "p95": "",
            "max": "",
        })
    write_summary(args.out_dir / "model_timing_variation_summary.csv", rows)

    maxwait = args.maxwait_ms
    if maxwait is None:
        maxwait = float(llm_df["inference_ms"].max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    boxplot_by_label(
        axes[0],
        {label: llm_df.loc[llm_df["condition"] == label, "inference_ms"] for label in MODEL_TRACES},
        "Inference latency (ms)",
    )
    axes[0].axhline(maxwait, color="#111827", linestyle="--", linewidth=1.1, label=f"maxwait = {maxwait:.1f} ms")
    axes[0].set_title("Inference timing by model")
    axes[0].legend()

    summary = pd.DataFrame(rows)
    inference = summary[summary["metric"] == "llm_inference_ms"].set_index("model")
    x = range(len(inference.index))
    axes[1].bar(
        [i - 0.18 for i in x],
        inference["mean"],
        width=0.36,
        label="Mean",
        color=[PALETTE.get(label, "#64748b") for label in inference.index],
        alpha=0.65,
    )
    axes[1].bar([i + 0.18 for i in x], inference["max"], width=0.36, label="Worst case", color="#111827", alpha=0.75)
    axes[1].axhline(maxwait, color="#dc2626", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(list(x), list(inference.index))
    axes[1].set_ylabel("Inference latency (ms)")
    axes[1].set_title("Mean vs worst-case latency")
    axes[1].legend()

    save_figure(fig, args.out_dir, "model_timing_variation")

    planner = exec_df[exec_df["federate"].astype(str).str.contains("planner", case=False, na=False)]
    planner_counts = pd.Series({label: _planner_tardy_count(tardy_df, label) for label in MODEL_TRACES})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(planner_counts.index, planner_counts.values, color=[PALETTE.get(x, "#64748b") for x in planner_counts.index])
    axes[0].set_ylabel("Planner tardy handler invocations")
    axes[0].set_title("ActionPlanner tardy-handler demand")

    boxplot_by_label(
        axes[1],
        {label: planner.loc[planner["condition"] == label, "lag_ms"] for label in MODEL_TRACES},
        "Planner reaction lag (ms)",
    )
    axes[1].set_title("Planner scheduling lag by model")

    save_figure(fig, args.out_dir, "model_planner_tardy_impact")

    print(f"Wrote {args.out_dir / 'model_timing_variation.png'}")
    print(f"Wrote {args.out_dir / 'model_planner_tardy_impact.png'}")
    print(f"Wrote {args.out_dir / 'model_timing_variation_summary.csv'}")


if __name__ == "__main__":
    main()
