#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import (
    DATA_ROOT,
    OUT_ROOT,
    MODEL_TRACES,
    PALETTE,
    apply_style,
    describe,
    load_execution_times,
    load_llm_behavior,
    save_figure,
    write_summary,
)


def p95(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(np.percentile(vals, 95))


def max_value(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(vals.max())


def tardy_count(exec_df: pd.DataFrame, condition: str, federate: str | None = None) -> int:
    subset = exec_df[exec_df["condition"] == condition]
    if federate is not None:
        subset = subset[subset["federate"].astype(str).str.contains(federate, case=False, na=False)]
    return int(subset["reaction"].astype(str).str.contains("tardy", case=False, na=False).sum())


def grouped_metric_by_federate(exec_df: pd.DataFrame, conditions: list[str], federates: list[str], metric: str) -> pd.DataFrame:
    rows = []
    for condition in conditions:
        cond = exec_df[exec_df["condition"] == condition]
        for fed in federates:
            subset = cond[cond["federate"].astype(str).eq(fed)]
            rows.append({"condition": condition, "federate": fed, f"p95_{metric}": p95(subset.get(metric, pd.Series(dtype=float)))})
    return pd.DataFrame(rows)


def plot_grouped_bars(ax, table: pd.DataFrame, x_col: str, group_col: str, y_col: str, ylabel: str, title: str) -> None:
    xs = list(table[x_col].drop_duplicates())
    groups = list(table[group_col].drop_duplicates())
    width = min(0.8 / max(len(groups), 1), 0.32)
    x = np.arange(len(xs))
    colors = plt.cm.Set2.colors
    for idx, group in enumerate(groups):
        vals = []
        for item in xs:
            row = table[(table[x_col] == item) & (table[group_col] == group)]
            vals.append(float(row[y_col].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + (idx - (len(groups) - 1) / 2) * width, vals, width=width, label=group, color=colors[idx % len(colors)])
    ax.set_xticks(x, xs)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)


def network_compensation(data_root: Path, out_dir: Path) -> None:
    traces = {
        "No netem": data_root / "logs-trace-removed-netem",
        "+150 ms netem": data_root / "logs-trace-netem-150ms-rasp",
    }
    exec_df = pd.concat([load_execution_times(path, label) for label, path in traces.items()], ignore_index=True)
    conditions = list(traces)
    federates = ["d_control", "d_monitor", "adc", "llm", "planner", "c", "sim"]
    lag_table = grouped_metric_by_federate(exec_df, conditions, federates, "lag_ms")
    lag_table = lag_table[lag_table["p95_lag_ms"] > 0]

    tardy_rows = []
    for condition in conditions:
        for federate in federates:
            count = tardy_count(exec_df, condition, federate)
            if count:
                tardy_rows.append({"condition": condition, "federate": federate, "tardy_count": count})
    tardy_table = pd.DataFrame(tardy_rows)
    if tardy_table.empty:
        tardy_table = pd.DataFrame({"condition": conditions, "federate": ["none"] * len(conditions), "tardy_count": [0] * len(conditions)})

    summary_rows = []
    for condition in conditions:
        subset = exec_df[exec_df["condition"] == condition]
        summary_rows.append({"condition": condition, "metric": "all_reactions_lag_ms", **describe(subset["lag_ms"])})
        summary_rows.append({"condition": condition, "metric": "all_reactions_execution_time_ms", **describe(subset["execution_time_ms"])})
        summary_rows.append({"condition": condition, "metric": "tardy_handler_invocations", "count": tardy_count(exec_df, condition), "mean": "", "median": "", "std": "", "p95": "", "max": ""})
    write_summary(out_dir / "network_latency_compensation_summary.csv", summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))
    plot_grouped_bars(
        axes[0], lag_table, "federate", "condition", "p95_lag_ms",
        "p95 reaction lag (physical - logical, ms)", "Network delay increases LF reaction lag",
    )
    plot_grouped_bars(
        axes[1], tardy_table, "federate", "condition", "tardy_count",
        "Tardy handler invocations", "LF tardy handlers invoked by late messages",
    )
    save_figure(fig, out_dir, "network_latency_compensation")


def gpu_compensation(data_root: Path, out_dir: Path) -> None:
    traces = {
        "Reference GPU": data_root / "logs-trace-dongha",
        "GPU workload": data_root / "logs-trace-added-another-process-gpu",
    }
    llm_df = pd.concat([load_llm_behavior(path, label) for label, path in traces.items()], ignore_index=True)
    exec_df = pd.concat([load_execution_times(path, label) for label, path in traces.items()], ignore_index=True)
    conditions = list(traces)

    llm_summary = []
    for condition in conditions:
        subset = llm_df[llm_df["condition"] == condition]
        llm_summary.append({
            "condition": condition,
            "mean_inference_ms": float(pd.to_numeric(subset["inference_ms"], errors="coerce").mean()),
            "p95_inference_ms": p95(subset["inference_ms"]),
            "worst_inference_ms": max_value(subset["inference_ms"]),
        })
    llm_summary_df = pd.DataFrame(llm_summary)

    system_rows = []
    for condition in conditions:
        system_rows.append({"condition": condition, "metric": "system_p95_reaction_lag_ms", "value": p95(exec_df.loc[exec_df["condition"] == condition, "lag_ms"])})
        system_rows.append({"condition": condition, "metric": "planner_tardy_count", "value": tardy_count(exec_df, condition, "planner")})
        system_rows.append({"condition": condition, "metric": "all_tardy_count", "value": tardy_count(exec_df, condition)})
    normalized_rows = []
    for row in llm_summary:
        condition = row["condition"]
        for key, value in row.items():
            if key != "condition":
                normalized_rows.append({"condition": condition, "metric": key, "value": value})
    normalized_rows.extend(system_rows)
    write_summary(out_dir / "gpu_latency_compensation_summary.csv", normalized_rows)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    x = np.arange(len(conditions))
    width = 0.25
    axes[0].bar(x - width, llm_summary_df["mean_inference_ms"], width, label="Mean", color="#93c5fd")
    axes[0].bar(x, llm_summary_df["p95_inference_ms"], width, label="p95", color="#2563eb")
    axes[0].bar(x + width, llm_summary_df["worst_inference_ms"], width, label="Worst", color="#1e3a8a")
    axes[0].set_xticks(x, conditions)
    axes[0].set_ylabel("LLM inference latency (ms)")
    axes[0].set_title("GPU workload increases inference latency")
    axes[0].legend()

    system_lag = [p95(exec_df.loc[exec_df["condition"] == c, "lag_ms"]) for c in conditions]
    axes[1].bar(conditions, system_lag, color=[PALETTE.get(c, "#64748b") for c in conditions])
    axes[1].set_ylabel("p95 reaction lag (physical - logical, ms)")
    axes[1].set_title("Inference latency propagates to LF reaction lag")
    axes[1].tick_params(axis="x", rotation=8)

    planner_tardy = [tardy_count(exec_df, c, "planner") for c in conditions]
    all_tardy = [tardy_count(exec_df, c) for c in conditions]
    axes[2].bar(x - width / 2, planner_tardy, width, label="Planner", color="#f97316")
    axes[2].bar(x + width / 2, all_tardy, width, label="All federates", color="#111827")
    axes[2].set_xticks(x, conditions)
    axes[2].set_ylabel("Tardy handler invocations")
    axes[2].set_title("System-level timing impact")
    axes[2].legend()

    save_figure(fig, out_dir, "gpu_latency_compensation")


def model_maxwait_variation(data_root: Path, out_dir: Path) -> None:
    trace_paths = {label: data_root / dirname for label, dirname in MODEL_TRACES.items()}
    llm_df = pd.concat([load_llm_behavior(path, label) for label, path in trace_paths.items()], ignore_index=True)
    exec_df = pd.concat([load_execution_times(path, label) for label, path in trace_paths.items()], ignore_index=True)
    labels = list(MODEL_TRACES)

    model_rows = []
    for label in labels:
        llm_subset = llm_df[llm_df["condition"] == label]
        planner_subset = exec_df[(exec_df["condition"] == label) & exec_df["federate"].astype(str).str.contains("planner", case=False, na=False)]
        model_rows.append({
            "model": label,
            "mean_inference_ms": float(pd.to_numeric(llm_subset["inference_ms"], errors="coerce").mean()),
            "p95_inference_ms": p95(llm_subset["inference_ms"]),
            "worst_inference_ms": max_value(llm_subset["inference_ms"]),
            "planner_p95_reaction_lag_ms": p95(planner_subset.get("lag_ms", pd.Series(dtype=float))),
            "planner_tardy_count": tardy_count(exec_df, label, "planner"),
        })
    table = pd.DataFrame(model_rows)
    configured_maxwait = float(table["worst_inference_ms"].max())
    table["configured_maxwait_ms"] = configured_maxwait
    write_summary(out_dir / "model_maxwait_variation_summary.csv", table.to_dict("records"))

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    x = np.arange(len(labels))
    width = 0.25
    axes[0].bar(x - width, table["mean_inference_ms"], width, label="Mean", color="#bbf7d0")
    axes[0].bar(x, table["p95_inference_ms"], width, label="p95", color="#16a34a")
    axes[0].bar(x + width, table["worst_inference_ms"], width, label="Worst", color="#14532d")
    axes[0].axhline(configured_maxwait, color="#dc2626", linestyle="--", linewidth=1.1, label=f"LF maxwait={configured_maxwait:.0f} ms")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Inference latency (ms)")
    axes[0].set_title("Model-dependent inference timing")
    axes[0].legend(fontsize=7)

    axes[1].bar(labels, table["planner_p95_reaction_lag_ms"], color=[PALETTE.get(label, "#64748b") for label in labels])
    axes[1].set_ylabel("Planner p95 reaction lag (physical - logical, ms)")
    axes[1].set_title("Downstream LF reaction-lag variation")

    axes[2].bar(labels, table["planner_tardy_count"], color=[PALETTE.get(label, "#64748b") for label in labels])
    axes[2].set_ylabel("Planner tardy handler invocations")
    axes[2].set_title("Planner tardy-handler demand")

    save_figure(fig, out_dir, "model_maxwait_variation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures aligned with the latency compensation evaluation section.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT / "latency-compensation-section")
    args = parser.parse_args()

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    network_compensation(args.data_root, args.out_dir)
    gpu_compensation(args.data_root, args.out_dir)
    model_maxwait_variation(args.data_root, args.out_dir)

    print(f"Wrote {args.out_dir / 'network_latency_compensation.png'}")
    print(f"Wrote {args.out_dir / 'gpu_latency_compensation.png'}")
    print(f"Wrote {args.out_dir / 'model_maxwait_variation.png'}")


if __name__ == "__main__":
    main()
