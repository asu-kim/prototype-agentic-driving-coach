#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from plot_common import (
    PAPER_STYLE, trace_label, trace_color, read_pair,
    save_figure, parse_common_args
)

def plot_event_count_comparison(car, llm, planner, out_dir):
    rows = []
    for component, df in [("Car", car), ("LLM", llm), ("Planner", planner)]:
        if df is None or df.empty or "event" not in df.columns:
            continue
        for trace, g in df.groupby("trace", sort=False):
            for event, count in g["event"].value_counts().items():
                rows.append({"component": component, "trace": trace, "event": event, "count": int(count)})
    counts = pd.DataFrame(rows)
    counts.to_csv(out_dir / "event_count_comparison.csv", index=False)
    focus = counts[counts["event"].isin(["reaction", "tardy", "input_arrival", "llm_output_tick", "mode_transition", "act_command_set", "act_command_set_tardy"])]
    if focus.empty:
        return
    focus = focus.copy()
    focus["label"] = focus["component"] + "\n" + focus["event"].astype(str)
    pivot = focus.pivot_table(index="label", columns="trace", values="count", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.42 * len(pivot))))
    pivot.plot(kind="barh", ax=ax, color=[trace_color(c) for c in pivot.columns])
    ax.set_xlabel("rows")
    ax.set_ylabel("")
    save_figure(fig, out_dir, "01_event_count_comparison")

def plot_car_input_overlay(car: pd.DataFrame, out_dir):
    df = car[car["event"].eq("reaction")].copy()
    signals = [("accel_in", "accelerator"), ("brake_in", "brake"), ("steer_in", "steer"), ("car_velocity", "car velocity")]
    signals = [(c, label) for c, label in signals if c in df.columns]
    traces = list(df["trace"].dropna().drop_duplicates())
    fig, axes = plt.subplots(len(signals), 1, figsize=(8.4, 2.2 * len(signals)), sharex=True)
    axes = np.asarray(axes).reshape(len(signals))
    for ax, (signal, ylabel) in zip(axes, signals):
        for trace in traces:
            g = df[df["trace"].eq(trace)]
            ax.plot(g["logical_time_ms"] / 1000, pd.to_numeric(g[signal], errors="coerce"),
                    linewidth=1.1, label=trace_label(trace), color=trace_color(trace), alpha=0.9)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("logical time (s)")
    axes[0].legend()
    save_figure(fig, out_dir, "02_car_input_overlay")

def plot_car_input_mismatch_heatmap(car: pd.DataFrame, out_dir):
    c = car[(car["trace"] == "Collected") & (car["event"] == "reaction")].copy()
    r = car[(car["trace"] == "Replay") & (car["event"] == "reaction")].copy()
    cols = ["steer_in_present", "accel_in_present", "brake_in_present", "actuate_in_present", "steer_in", "accel_in", "brake_in", "actuate_in", "car_velocity", "car_steer"]
    cols = [col for col in cols if col in c.columns and col in r.columns]
    merged = c[["logical_time_ns", "logical_time_ms"] + cols].merge(
        r[["logical_time_ns"] + cols], on="logical_time_ns", how="inner", suffixes=("_collected", "_replay")
    )
    if merged.empty:
        print("No common logical times for car mismatch heatmap.")
        return
    heat_rows = []
    for col in cols:
        a = pd.to_numeric(merged[f"{col}_collected"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_replay"], errors="coerce")
        if col.endswith("_present"):
            mismatch = (a.fillna(-999) != b.fillna(-999)).astype(int)
        else:
            mismatch = (~np.isclose(a.fillna(-999), b.fillna(-999), atol=1e-6)).astype(int)
        heat_rows.append(mismatch.to_numpy())
    heat = np.vstack(heat_rows)
    pd.DataFrame({
        "signal": cols,
        "mismatch_count": heat.sum(axis=1).astype(int),
        "total_compared": heat.shape[1],
        "mismatch_percent": 100 * heat.sum(axis=1) / heat.shape[1],
    }).to_csv(out_dir / "car_input_mismatch_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    cmap = ListedColormap(["#F2F2F2", "#C33149"])
    im = ax.imshow(heat, aspect="auto", interpolation="nearest", cmap=cmap, origin="lower", vmin=0, vmax=1)
    xticks = np.linspace(0, len(merged) - 1, min(8, len(merged))).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels((merged.iloc[xticks]["logical_time_ms"] / 1000).round(1))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xlabel("logical time (s)")
    ax.set_ylabel("signal")
    ax.set_title("Collection vs replay input mismatch")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["match", "mismatch"])
    save_figure(fig, out_dir, "03_car_input_mismatch_heatmap")

def plot_planner_decision_timeline(planner: pd.DataFrame | None, out_dir):
    if planner is None or planner.empty:
        return
    df = planner.copy()
    token_order = ["NONE", "WARNING", "ACTUATE"]
    mode_order = ["MONITORING", "WARNING", "ACTUATE"]
    act_order = ["STOP", "BRAKE", "HOLD", "ACCEL"]
    token_to_y = {x: i for i, x in enumerate(token_order)}
    mode_to_y = {x: i for i, x in enumerate(mode_order)}
    act_to_y = {x: i for i, x in enumerate(act_order)}
    traces = list(df["trace"].dropna().drop_duplicates())
    fig, axes = plt.subplots(3, len(traces), figsize=(4.8 * len(traces), 6.4), sharex="col", sharey="row")
    axes = np.asarray(axes).reshape(3, len(traces))
    for col, trace in enumerate(traces):
        g = df[df["trace"].eq(trace)]
        if "control_token" in g.columns:
            axes[0, col].scatter(g["logical_time_ms"] / 1000, g["control_token"].astype(str).map(token_to_y), s=18, alpha=0.75, color=trace_color(trace), edgecolors="none")
        transitions = g[g["event"].eq("mode_transition")] if "event" in g.columns else g.iloc[0:0]
        if "to_mode" in transitions.columns:
            axes[1, col].scatter(transitions["logical_time_ms"] / 1000, transitions["to_mode"].astype(str).map(mode_to_y), s=24, alpha=0.8, color=trace_color(trace), edgecolors="none")
        act_rows = g[g.get("act_set", pd.Series(dtype=str)).astype(str).eq("1")]
        if "act_command" in act_rows.columns:
            axes[2, col].scatter(act_rows["logical_time_ms"] / 1000, act_rows["act_command"].astype(str).map(act_to_y), s=28, alpha=0.9, color=trace_color(trace), edgecolors="none")
        axes[0, col].set_title(trace_label(trace))
        axes[2, col].set_xlabel("logical time (s)")
    axes[0, 0].set_ylabel("LLM token")
    axes[0, 0].set_yticks(list(token_to_y.values()), token_order)
    axes[1, 0].set_ylabel("planner mode")
    axes[1, 0].set_yticks(list(mode_to_y.values()), mode_order)
    axes[2, 0].set_ylabel("act command")
    axes[2, 0].set_yticks(list(act_to_y.values()), act_order)
    save_figure(fig, out_dir, "04_planner_decision_timeline")

def write_first_divergence(car: pd.DataFrame, out_dir):
    c = car[(car["trace"] == "Collected") & (car["event"] == "reaction")].copy()
    r = car[(car["trace"] == "Replay") & (car["event"] == "reaction")].copy()
    cols = ["steer_in", "accel_in", "brake_in", "actuate_in", "car_velocity", "car_steer"]
    cols = [col for col in cols if col in c.columns and col in r.columns]
    merged = c[["logical_time_ns", "logical_time_ms"] + cols].merge(r[["logical_time_ns"] + cols], on="logical_time_ns", how="inner", suffixes=("_collected", "_replay"))
    rows = []
    for col in cols:
        a = pd.to_numeric(merged[f"{col}_collected"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_replay"], errors="coerce")
        mismatch = ~np.isclose(a.fillna(-999), b.fillna(-999), atol=1e-6)
        if mismatch.any():
            idx = mismatch.to_numpy().argmax()
            rows.append({
                "signal": col,
                "first_divergence_logical_time_ms": merged.iloc[idx]["logical_time_ms"],
                "first_divergence_logical_time_s": merged.iloc[idx]["logical_time_ms"] / 1000,
                "collected_value": merged.iloc[idx][f"{col}_collected"],
                "replay_value": merged.iloc[idx][f"{col}_replay"],
            })
    pd.DataFrame(rows).to_csv(out_dir / "first_divergence_points.csv", index=False)

def main():
    args = parse_common_args("Generate deterministic replay comparison plots.")
    plt.rcParams.update(PAPER_STYLE)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    car = read_pair(args.original_dir, args.replay_dir, "car_inputs.csv", required=True)
    llm = read_pair(args.original_dir, args.replay_dir, "llm_inputs.csv")
    planner = read_pair(args.original_dir, args.replay_dir, "planner_events.csv")
    plot_event_count_comparison(car, llm, planner, args.out_dir)
    plot_car_input_overlay(car, args.out_dir)
    plot_car_input_mismatch_heatmap(car, args.out_dir)
    plot_planner_decision_timeline(planner, args.out_dir)
    write_first_divergence(car, args.out_dir)
    print(f"Wrote determinism plots to: {args.out_dir}")

if __name__ == "__main__":
    main()
