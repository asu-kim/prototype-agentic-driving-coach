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


def _tardy_breakdown(tardy_df: pd.DataFrame, conditions) -> pd.DataFrame:
    if tardy_df.empty:
        return pd.DataFrame(columns=["condition", "category", "count"])
    df = tardy_df.copy()
    df["category"] = df["reactor"].fillna("").astype(str)
    reaction = df["reaction"].fillna("").astype(str)
    df.loc[reaction != "", "category"] = df.loc[reaction != "", "category"] + ":" + reaction[reaction != ""]
    grouped = df.groupby(["condition", "category"]).size().reset_index(name="count")
    top = grouped.groupby("category")["count"].sum().sort_values(ascending=False).head(8).index
    grouped = grouped[grouped["category"].isin(top)]
    full = pd.MultiIndex.from_product([conditions, top], names=["condition", "category"]).to_frame(index=False)
    return full.merge(grouped, on=["condition", "category"], how="left").fillna({"count": 0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the effect of injected Raspberry Pi network latency.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--normal-trace", default=NORMAL_TRACE)
    parser.add_argument("--network-trace", default=NETWORK_TRACE)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT / "network-latency")
    args = parser.parse_args()

    apply_style()
    traces = {
        "No netem": args.data_root / args.normal_trace,
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    boxplot_by_label(
        axes[0],
        {label: exec_df.loc[exec_df["condition"] == label, "lag_ms"] for label in traces},
        "Reaction lag: physical start - logical tag (ms)",
    )
    axes[0].set_title("Federate reaction lag")

    boxplot_by_label(
        axes[1],
        {label: exec_df.loc[exec_df["condition"] == label, "execution_time_ms"] for label in traces},
        "Reaction execution time (ms)",
    )
    axes[1].set_title("Reaction runtime")

    tardy_counts = tardy_df.groupby("condition").size().reindex(traces.keys(), fill_value=0)
    axes[2].bar(tardy_counts.index, tardy_counts.values, color=[PALETTE.get(x, "#64748b") for x in tardy_counts.index])
    axes[2].set_ylabel("Tardy handler invocations")
    axes[2].set_title("Tardy handler invocations")
    axes[2].tick_params(axis="x", rotation=10)

    save_figure(fig, args.out_dir, "network_latency_effect")

    breakdown = _tardy_breakdown(tardy_df, list(traces.keys()))
    if not breakdown.empty:
        write_summary(args.out_dir / "network_tardy_breakdown_summary.csv", breakdown.to_dict("records"))
        pivot = breakdown.pivot(index="condition", columns="category", values="count").reindex(traces.keys()).fillna(0)
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        bottom = pd.Series(0, index=pivot.index, dtype=float)
        colors = plt.cm.tab20.colors
        for i, category in enumerate(pivot.columns):
            ax.bar(pivot.index, pivot[category], bottom=bottom, label=category, color=colors[i % len(colors)])
            bottom = bottom + pivot[category]
        ax.set_ylabel("Tardy handler invocations")
        ax.set_title("Tardy-message breakdown under network delay")
        ax.tick_params(axis="x", rotation=10)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=7)
        save_figure(fig, args.out_dir, "network_tardy_breakdown")
        print(f"Wrote {args.out_dir / 'network_tardy_breakdown.png'}")
        print(f"Wrote {args.out_dir / 'network_tardy_breakdown_summary.csv'}")

    print(f"Wrote {args.out_dir / 'network_latency_effect.png'}")
    print(f"Wrote {args.out_dir / 'network_latency_summary.csv'}")


if __name__ == "__main__":
    main()
