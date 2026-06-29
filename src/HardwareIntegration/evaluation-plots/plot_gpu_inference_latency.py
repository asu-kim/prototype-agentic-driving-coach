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
    apply_style,
    boxplot_by_label,
    describe,
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
        "Normal": args.data_root / args.normal_trace,
        "GPU loaded": args.data_root / args.gpu_trace,
    }
    llm_df = pd.concat(
        [load_llm_behavior(path, label) for label, path in traces.items()],
        ignore_index=True,
    )

    rows = []
    for label in traces:
        subset = llm_df[llm_df["condition"] == label]
        for metric in ("inference_ms", "lag_ms"):
            rows.append({"condition": label, "metric": metric, **describe(subset[metric])})
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
    print(f"Wrote {args.out_dir / 'gpu_inference_latency.png'}")
    print(f"Wrote {args.out_dir / 'gpu_inference_latency_summary.csv'}")


if __name__ == "__main__":
    main()
