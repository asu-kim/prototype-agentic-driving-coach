#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_common import (
    PAPER_STYLE, COLORS, trace_label, trace_color, read_pair, load_execution,
    load_federate_lag, save_figure, parse_common_args
)

def plot_tardy_timeline(car: pd.DataFrame, out_dir):
    df = car[car["event"].isin(["reaction", "tardy"])].copy()
    traces = list(df["trace"].dropna().drop_duplicates())
    fig, axes = plt.subplots(len(traces), 1, figsize=(8.4, 2.4 * len(traces)), sharex=True)
    axes = np.asarray(axes).reshape(len(traces))
    ymap = {"reaction": 0, "tardy": 1}
    for ax, trace in zip(axes, traces):
        g = df[df["trace"].eq(trace)]
        colors = g["event"].map({"reaction": COLORS["Reaction"], "tardy": COLORS["Tardy"]})
        ax.scatter(g["logical_time_ms"] / 1000, g["event"].map(ymap), s=14, c=colors, alpha=0.8, edgecolors="none")
        ax.set_title(f"{trace_label(trace)}: reaction and tardy timeline")
        ax.set_yticks([0, 1], ["reaction", "tardy"])
        ax.set_ylabel("event")
    axes[-1].set_xlabel("logical time (s)")
    save_figure(fig, out_dir, "01_tardy_timeline")

def plot_tardy_counts(car: pd.DataFrame, llm: pd.DataFrame | None, planner: pd.DataFrame | None, out_dir):
    rows = []
    for name, df in [("Car", car), ("LLM input", llm), ("Planner", planner)]:
        if df is None or df.empty or "event" not in df.columns:
            continue
        for trace, g in df.groupby("trace", sort=False):
            rows.append({
                "component": name,
                "trace": trace,
                "tardy_count": int(g["event"].astype(str).str.contains("tardy", case=False, na=False).sum())
            })
    counts = pd.DataFrame(rows)
    counts.to_csv(out_dir / "tardy_counts_by_component.csv", index=False)
    if counts.empty:
        return
    pivot = counts.pivot(index="component", columns="trace", values="tardy_count").fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    pivot.plot(kind="bar", ax=ax, color=[trace_color(c) for c in pivot.columns])
    ax.set_xlabel("")
    ax.set_ylabel("tardy rows")
    ax.tick_params(axis="x", rotation=0)
    save_figure(fig, out_dir, "02_tardy_counts_by_component")

def plot_physical_logical_lag(car: pd.DataFrame, out_dir):
    traces = list(car["trace"].dropna().drop_duplicates())
    fig, axes = plt.subplots(len(traces), 1, figsize=(8.4, 2.4 * len(traces)), sharex=True)
    axes = np.asarray(axes).reshape(len(traces))
    for ax, trace in zip(axes, traces):
        g = car[car["trace"].eq(trace)].copy()
        lag = g["physical_time_ms"] - g["logical_time_ms"]
        ax.plot(g["logical_time_ms"] / 1000, lag, linewidth=1.1, color=trace_color(trace))
        ax.set_title(f"{trace_label(trace)}: physical-logical lag")
        ax.set_ylabel("lag (ms)")
    axes[-1].set_xlabel("logical time (s)")
    save_figure(fig, out_dir, "03_physical_logical_lag")

def plot_execution_p95(exec_df: pd.DataFrame | None, out_dir):
    if exec_df is None or exec_df.empty or "execution_time_ms" not in exec_df.columns:
        return
    df = exec_df.dropna(subset=["execution_time_ms"]).copy()
    summary = (
        df.groupby(["trace", "federate", "reactor", "reaction"], dropna=False)["execution_time_ms"]
        .agg(count="count", mean_ms="mean", p95_ms=lambda s: s.quantile(0.95), max_ms="max")
        .reset_index()
    )
    summary.to_csv(out_dir / "execution_time_summary.csv", index=False)
    summary["label"] = summary["federate"].astype(str) + "\n" + summary["reaction"].astype(str)
    order = summary.groupby("label")["p95_ms"].max().sort_values(ascending=False).head(12).index.tolist()[::-1]
    traces = list(summary["trace"].dropna().drop_duplicates())
    y = np.arange(len(order))
    width = min(0.8 / max(len(traces), 1), 0.32)
    fig, ax = plt.subplots(figsize=(8.6, max(4.2, 0.4 * len(order))))
    for i, trace in enumerate(traces):
        vals = []
        for label in order:
            m = summary[(summary["trace"].eq(trace)) & (summary["label"].eq(label))]
            vals.append(float(m["p95_ms"].iloc[0]) if not m.empty else 0)
        ax.barh(y + (i - (len(traces)-1)/2) * width, vals, height=width, label=trace_label(trace), color=trace_color(trace))
    ax.set_yticks(y, order)
    ax.set_xlabel("p95 execution time (ms)")
    ax.legend()
    save_figure(fig, out_dir, "04_execution_time_p95")

def plot_federate_lag(federate_lag: pd.DataFrame | None, out_dir):
    if federate_lag is None or federate_lag.empty:
        return
    federate_lag.to_csv(out_dir / "all_federate_lag_samples.csv", index=False)
    summary = (
        federate_lag.groupby(["trace", "federate"])["lag_ms"]
        .agg(count="count", mean_ms="mean", p95_ms=lambda s: s.quantile(0.95), max_ms="max")
        .reset_index()
    )
    summary.to_csv(out_dir / "all_federate_lag_summary.csv", index=False)
    p95 = summary.pivot(index="federate", columns="trace", values="p95_ms").fillna(0)
    fig, ax = plt.subplots(figsize=(8.0, max(3.8, 0.42 * len(p95))))
    p95.plot(kind="barh", ax=ax, color=[trace_color(c) for c in p95.columns])
    ax.set_xlabel("p95 lag (ms)")
    ax.set_ylabel("federate")
    save_figure(fig, out_dir, "05_federate_lag_p95")

def plot_clock_period(car: pd.DataFrame, out_dir):
    rows = []
    for trace, g in car[car["event"].eq("reaction")].groupby("trace", sort=False):
        times = g["logical_time_ms"].dropna().drop_duplicates().sort_values()
        for p in times.diff().dropna():
            rows.append({"trace": trace, "period_ms": p, "frequency_hz": 1000.0 / p if p > 0 else np.nan})
    periods = pd.DataFrame(rows)
    if periods.empty:
        return
    periods.to_csv(out_dir / "clock_period_samples.csv", index=False)
    traces = list(periods["trace"].dropna().drop_duplicates())
    fig, axes = plt.subplots(1, len(traces), figsize=(4.6 * len(traces), 3.6), sharey=True)
    axes = np.asarray(axes).reshape(len(traces))
    for ax, trace in zip(axes, traces):
        g = periods[periods["trace"].eq(trace)]
        ax.hist(g["period_ms"], bins=25, alpha=0.8, color=trace_color(trace))
        ax.set_title(trace_label(trace))
        ax.set_xlabel("period (ms)")
    axes[0].set_ylabel("count")
    save_figure(fig, out_dir, "06_clock_period_histogram")

def main():
    args = parse_common_args("Generate latency and tardiness plots.")
    plt.rcParams.update(PAPER_STYLE)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    car = read_pair(args.original_dir, args.replay_dir, "car_inputs.csv", required=True)
    llm = read_pair(args.original_dir, args.replay_dir, "llm_inputs.csv")
    planner = read_pair(args.original_dir, args.replay_dir, "planner_events.csv")
    exec_df = load_execution(args.original_dir, args.replay_dir)
    federate_lag = load_federate_lag(args.original_dir, args.replay_dir)

    plot_tardy_timeline(car, args.out_dir)
    plot_tardy_counts(car, llm, planner, args.out_dir)
    plot_physical_logical_lag(car, args.out_dir)
    plot_execution_p95(exec_df, args.out_dir)
    plot_federate_lag(federate_lag, args.out_dir)
    plot_clock_period(car, args.out_dir)

    print(f"Wrote latency plots to: {args.out_dir}")

if __name__ == "__main__":
    main()
