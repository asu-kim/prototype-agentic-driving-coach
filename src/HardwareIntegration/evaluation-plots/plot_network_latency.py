#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_common import (
    DATA_ROOT,
    NETWORK_TRACE,
    NORMAL_TRACE,
    OUT_ROOT,
    PALETTE,
    apply_style,
    boxplot_by_label,
    collect_tardy_rows,
    describe,
    load_execution_times,
    save_figure,
    write_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the effect of injected Raspberry Pi network latency.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--normal-trace", default=NORMAL_TRACE)
    parser.add_argument("--network-trace", default=NETWORK_TRACE)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT / "network-latency")
    args = parser.parse_args()

    apply_style()
    traces = {
        "Normal": args.data_root / args.normal_trace,
        "Network +150 ms": args.data_root / args.network_trace,
    }

    exec_df = pd.concat(
        [load_execution_times(path, label) for label, path in traces.items()],
        ignore_index=True,
    )
    tardy_df = pd.concat(
        [collect_tardy_rows(path, label) for label, path in traces.items()],
        ignore_index=True,
    )

    summary_rows = []
    for label in traces:
        subset = exec_df[exec_df["condition"] == label]
        for metric in ("lag_ms", "execution_time_ms"):
            stats = describe(subset[metric]) if metric in subset else describe([])
            summary_rows.append({"condition": label, "metric": metric, **stats})
        summary_rows.append({
            "condition": label,
            "metric": "tardy_rows",
            "count": len(tardy_df[tardy_df["condition"] == label]),
            "mean": "",
            "median": "",
            "std": "",
            "p95": "",
            "max": "",
        })
    write_summary(args.out_dir / "network_latency_summary.csv", summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    boxplot_by_label(
        axes[0],
        {label: exec_df.loc[exec_df["condition"] == label, "lag_ms"] for label in traces},
        "Physical minus logical time (ms)",
    )
    axes[0].set_title("Federate timing lag")

    tardy_counts = tardy_df.groupby("condition").size().reindex(traces.keys(), fill_value=0)
    axes[1].bar(tardy_counts.index, tardy_counts.values, color=[PALETTE.get(x, "#64748b") for x in tardy_counts.index])
    axes[1].set_ylabel("Tardy rows")
    axes[1].set_title("Timing violations under network delay")
    axes[1].tick_params(axis="x", rotation=10)

    save_figure(fig, args.out_dir, "network_latency_effect")
    print(f"Wrote {args.out_dir / 'network_latency_effect.png'}")
    print(f"Wrote {args.out_dir / 'network_latency_summary.csv'}")


if __name__ == "__main__":
    main()
