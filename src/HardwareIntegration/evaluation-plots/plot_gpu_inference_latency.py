#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import (
    DATA_ROOT,
    GPU_TRACE,
    NORMAL_TRACE,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LLM inference timing with and without added GPU workload.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--normal-trace", default=NORMAL_TRACE)
    parser.add_argument("--gpu-trace", default=GPU_TRACE)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT / "gpu-inference-latency")
    args = parser.parse_args()

    apply_style()
    traces = {
        "Reference": args.data_root / args.normal_trace,
        "GPU loaded": args.data_root / args.gpu_trace,
    }
    llm_df = pd.concat(
        [load_llm_behavior(path, label) for label, path in traces.items()],
        ignore_index=True,
    )
    exec_df = pd.concat(
        [load_execution_times(path, label) for label, path in traces.items()],
        ignore_index=True,
    )
    tardy_df = pd.concat(
        [collect_tardy_rows(path, label) for label, path in traces.items()],
        ignore_index=True,
    )

    rows = []
    for label in traces:
        llm_subset = llm_df[llm_df["condition"] == label]
        for metric in ("inference_ms", "lag_ms"):
            rows.append({"condition": label, "metric": f"llm_{metric}", **describe(llm_subset[metric])})
        exec_subset = exec_df[exec_df["condition"] == label]
        for metric in ("lag_ms", "execution_time_ms"):
            rows.append({"condition": label, "metric": f"system_{metric}", **describe(exec_subset[metric])})
        rows.append({
            "condition": label,
            "metric": "tardy_rows",
            "count": len(tardy_df[tardy_df["condition"] == label]),
            "mean": "",
            "median": "",
            "std": "",
            "p95": "",
            "max": "",
        })
    write_summary(args.out_dir / "gpu_inference_latency_summary.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    boxplot_by_label(
        axes[0],
        {label: llm_df.loc[llm_df["condition"] == label, "inference_ms"] for label in traces},
        "Inference latency (ms)",
    )
    axes[0].set_title("LLM inference latency")

    for label in traces:
        subset = llm_df[llm_df["condition"] == label].sort_values("logical_time_ms")
        axes[1].plot(
            subset["logical_time_ms"] / 1000.0,
            subset["inference_ms"],
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            label=label,
        )
    axes[1].set_xlabel("Logical time (s)")
    axes[1].set_ylabel("Inference latency (ms)")
    axes[1].set_title("Inference latency over execution")
    axes[1].legend()

    save_figure(fig, args.out_dir, "gpu_inference_latency")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    boxplot_by_label(
        axes[0],
        {label: exec_df.loc[exec_df["condition"] == label, "lag_ms"] for label in traces},
        "Reaction lag: physical start - logical tag (ms)",
    )
    axes[0].set_title("System reaction lag")

    non_llm = exec_df[~exec_df["federate"].astype(str).str.contains("llm", case=False, na=False)]
    boxplot_by_label(
        axes[1],
        {label: non_llm.loc[non_llm["condition"] == label, "execution_time_ms"] for label in traces},
        "Non-LLM reaction runtime (ms)",
    )
    axes[1].set_title("System runtime outside LLM")

    tardy_counts = tardy_df.groupby("condition").size().reindex(traces.keys(), fill_value=0)
    axes[2].bar(tardy_counts.index, tardy_counts.values, color=[PALETTE.get(x, "#64748b") for x in tardy_counts.index])
    axes[2].set_ylabel("Tardy handler invocations")
    axes[2].set_title("System tardy-handler impact")
    axes[2].tick_params(axis="x", rotation=10)

    save_figure(fig, args.out_dir, "gpu_system_impact")

    print(f"Wrote {args.out_dir / 'gpu_inference_latency.png'}")
    print(f"Wrote {args.out_dir / 'gpu_system_impact.png'}")
    print(f"Wrote {args.out_dir / 'gpu_inference_latency_summary.csv'}")


if __name__ == "__main__":
    main()
