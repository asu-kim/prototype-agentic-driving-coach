#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import REPO_ROOT, apply_style, save_figure, write_summary

DATA_ROOT = REPO_ROOT / "src/HardwareIntegration/data-maxwait"
OUT_ROOT = REPO_ROOT / "src/HardwareIntegration/evaluation-plots/out/data-maxwait-clean"

FED_LABELS = {
    "d_control": "DriverControl",
    "d_monitor": "DriverMonitor",
    "adc": "InputForCoach",
    "llm": "AgenticCoach",
    "planner": "ActionPlanner",
    "c": "Car",
    "sim": "Simulation",
}
FED_ORDER = ["d_control", "d_monitor", "adc", "llm", "planner", "c", "sim"]
PROP_ORDER = ["llm", "planner", "c"]
MODEL_TRACES = {
    "llama3.1:8b": "logs-trace-750-8b",
    "phi4:14b": "logs-trace-750-14b",
    "llama3.1:70b": "logs-trace-750-70b",
}


def write_rows(path: Path, rows: list[dict]):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, engine="python", on_bad_lines="skip")
    for col in ["logical_time_ms", "logical_time_ns", "lag_ms", "execution_time_ms", "inference_ms"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_exec(trace_dir: Path, condition: str) -> pd.DataFrame:
    frames = []
    for path in sorted(trace_dir.glob("federate_execution_times*.csv")):
        df = read_csv(path)
        if "federate" not in df or "reaction" not in df:
            continue
        df["condition"] = condition
        df["source_file"] = path.name
        df = df[df["federate"].astype(str).isin(FED_LABELS)]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_llm(trace_dir: Path, condition: str) -> pd.DataFrame:
    df = read_csv(trace_dir / "llm_behavior.csv")
    df["condition"] = condition
    return df


def p95(values) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.percentile(vals, 95)) if len(vals) else 0.0


def meanv(values) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else 0.0


def maxv(values) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.max()) if len(vals) else 0.0


def tardy_count(df: pd.DataFrame, condition: str, fed: str | None = None) -> int:
    sub = df[df["condition"] == condition]
    if fed:
        sub = sub[sub["federate"].astype(str).eq(fed)]
    return int(sub["reaction"].astype(str).str.contains("tardy", case=False, na=False).sum())


def metric_by_fed(df: pd.DataFrame, condition: str, feds: list[str], metric: str, fn=p95) -> list[float]:
    out = []
    for fed in feds:
        sub = df[(df["condition"] == condition) & (df["federate"].astype(str).eq(fed))]
        out.append(fn(sub.get(metric, pd.Series(dtype=float))))
    return out


def barh_compare(labels, a, b, a_label, b_label, xlabel, title, out_dir, stem):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y - h / 2, a, height=h, label=a_label, color="#2563eb")
    ax.barh(y + h / 2, b, height=h, label=b_label, color="#dc2626")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False, prop={"weight": "bold", "size": 9})
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, out_dir, stem)


def grouped_bars(labels, series, ylabel, title, out_dir, stem, note: str | None = None, hline: float | None = None):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(labels))
    n = len(series)
    w = min(0.8 / max(n, 1), 0.28)
    colors = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed"]
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, width=w, label=name, color=colors[i % len(colors)])
    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.0, color="#111827", label=f"LF maxwait={hline:.0f} ms")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if note:
        ax.text(0.0, 1.02, note, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.legend(fontsize=7, frameon=False, prop={"weight": "bold", "size": 9})
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, stem)


def plot_network(root: Path, out: Path):
    traces = {
        "Nominal communication latency": root / "logs-trace-coach-120",
        "Netem added network delay": root / "logs-trace-coach-120-network-delay",
    }
    df = pd.concat([load_exec(path, label) for label, path in traces.items()], ignore_index=True)
    feds = [fed for fed in FED_ORDER if fed in set(df["federate"].astype(str))]
    labels = [FED_LABELS[f] for f in feds]
    a, b = list(traces)

    barh_compare(
        labels,
        metric_by_fed(df, a, feds, "lag_ms", p95),
        metric_by_fed(df, b, feds, "lag_ms", p95),
        a,
        b,
        "p95 Lag (ms)",
        "",
        out,
        "network_federate_reaction_lag_p95",
    )
    barh_compare(
        labels,
        [tardy_count(df, a, fed) for fed in feds],
        [tardy_count(df, b, fed) for fed in feds],
        a,
        b,
        "Tardy handler invocations",
        "",
        out,
        "network_tardy_handlers_by_federate",
    )

    focus = [fed for fed in ["adc", "c", "llm", "planner"] if fed in feds]
    grouped_bars(
        [FED_LABELS[f] for f in focus],
        {
            a: metric_by_fed(df, a, focus, "lag_ms", p95),
            b: metric_by_fed(df, b, focus, "lag_ms", p95),
        },
        "p95 Lag (ms)",
        "",
        out,
        "network_adc_car_path_lag",
    )

    rows = []
    for cond in traces:
        for fed in feds:
            rows.append({
                "condition": cond,
                "federate": FED_LABELS[fed],
                "p95_reaction_lag_ms": p95(df.loc[(df["condition"] == cond) & (df["federate"] == fed), "lag_ms"]),
                "tardy_handler_invocations": tardy_count(df, cond, fed),
            })
    write_rows(out / "network_lf_federate_summary.csv", rows)


def plot_gpu(root: Path, out: Path):
    traces = {
        "No GPU overload": root / "logs-trace-nogpuoverload",
        "GPU overload": root / "logs-trace-gpuoverload",
    }
    llm = pd.concat([load_llm(path, label) for label, path in traces.items()], ignore_index=True)
    df = pd.concat([load_exec(path, label) for label, path in traces.items()], ignore_index=True)
    a, b = list(traces)

    grouped_bars(
        list(traces),
        {
            "Mean": [meanv(llm.loc[llm["condition"] == c, "inference_ms"]) for c in traces],
            "p95": [p95(llm.loc[llm["condition"] == c, "inference_ms"]) for c in traces],
            "Worst": [maxv(llm.loc[llm["condition"] == c, "inference_ms"]) for c in traces],
        },
        "LLM inference latency (ms)",
        "",
        out,
        "gpu_llm_inference_latency",
    )

    path_feds = [fed for fed in PROP_ORDER if fed in set(df["federate"].astype(str))]
    labels = [FED_LABELS[f] for f in path_feds]
    grouped_bars(
        labels,
        {
            a: metric_by_fed(df, a, path_feds, "lag_ms", p95),
            b: metric_by_fed(df, b, path_feds, "lag_ms", p95),
        },
        "p95 Lag (ms)",
        "",
        out,
        "gpu_lag_propagation_llm_planner_car",
    )
    grouped_bars(
        labels,
        {
            a: [tardy_count(df, a, fed) for fed in path_feds],
            b: [tardy_count(df, b, fed) for fed in path_feds],
        },
        "Tardy handler invocations",
        "",
        out,
        "gpu_tardy_propagation_llm_planner_car",
    )

    rows = []
    for cond in traces:
        rows.append({
            "condition": cond,
            "agenticcoach_mean_inference_ms": meanv(llm.loc[llm["condition"] == cond, "inference_ms"]),
            "agenticcoach_p95_inference_ms": p95(llm.loc[llm["condition"] == cond, "inference_ms"]),
            "agenticcoach_worst_inference_ms": maxv(llm.loc[llm["condition"] == cond, "inference_ms"]),
        })
        for fed in path_feds:
            rows.append({
                "condition": cond,
                "federate": FED_LABELS[fed],
                "p95_reaction_lag_ms": p95(df.loc[(df["condition"] == cond) & (df["federate"] == fed), "lag_ms"]),
                "tardy_handler_invocations": tardy_count(df, cond, fed),
            })
    write_rows(out / "gpu_lag_propagation_summary.csv", rows)


def plot_models(root: Path, out: Path):
    llm = pd.concat([load_llm(root / dirname, label) for label, dirname in MODEL_TRACES.items()], ignore_index=True)
    df = pd.concat([load_exec(root / dirname, label) for label, dirname in MODEL_TRACES.items()], ignore_index=True)
    labels = list(MODEL_TRACES)
    maxwait = 750.0

    grouped_bars(
        labels,
        {
            "Mean": [meanv(llm.loc[llm["condition"] == c, "inference_ms"]) for c in labels],
            "p95": [p95(llm.loc[llm["condition"] == c, "inference_ms"]) for c in labels],
            "Worst": [maxv(llm.loc[llm["condition"] == c, "inference_ms"]) for c in labels],
        },
        "AgenticCoach inference latency (ms)",
        "",
        out,
        "model_llm_clock_750_inference",
        note="LLM triggered = 750 ms; ActionPlanner LF maxwait = 750 ms",
        hline=maxwait,
    )

    feds = [fed for fed in ["llm", "planner", "c", "d_monitor", "d_control"] if fed in set(df["federate"].astype(str))]
    for metric_name, stem, ylabel in [
        ("lag", "model_reaction_lag_by_federate", "p95 Lag (ms)"),
        ("tardy", "model_tardy_handlers_by_federate", "Tardy handler invocations"),
    ]:
        series = {}
        for model in labels:
            if metric_name == "lag":
                series[model] = metric_by_fed(df, model, feds, "lag_ms", p95)
            else:
                series[model] = [tardy_count(df, model, fed) for fed in feds]
        grouped_bars(
            [FED_LABELS[f] for f in feds],
            series,
            ylabel,
            "",
            out,
            stem,
            note="LLM triggered = 750 ms; ActionPlanner LF maxwait = 750 ms",
        )

    rows = []
    for model in labels:
        rows.append({
            "model": model,
            "llm_clock_ms": 750,
            "actionplanner_maxwait_ms": 750,
            "mean_inference_ms": meanv(llm.loc[llm["condition"] == model, "inference_ms"]),
            "p95_inference_ms": p95(llm.loc[llm["condition"] == model, "inference_ms"]),
            "worst_inference_ms": maxv(llm.loc[llm["condition"] == model, "inference_ms"]),
        })
        for fed in feds:
            rows.append({
                "model": model,
                "federate": FED_LABELS[fed],
                "p95_reaction_lag_ms": p95(df.loc[(df["condition"] == model) & (df["federate"] == fed), "lag_ms"]),
                "tardy_handler_invocations": tardy_count(df, model, fed),
            })
    write_rows(out / "model_750ms_maxwait_summary.csv", rows)


def main():
    parser = argparse.ArgumentParser(description="Plot data-maxwait LF timing experiments.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_network(args.data_root, args.out_dir)
    plot_gpu(args.data_root, args.out_dir)
    plot_models(args.data_root, args.out_dir)
    print(f"Wrote data-maxwait LF plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
