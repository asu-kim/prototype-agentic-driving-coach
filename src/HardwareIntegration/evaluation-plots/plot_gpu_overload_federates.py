#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import REPO_ROOT


NO_GPU_OVERLOAD_TRACE_DIR = REPO_ROOT / "/home/asurite.ad.asu.edu/dprahlad/deterministic-prototyping-agenticCPS/src/HardwareIntegration/evaluation/logs-trace-normal-assumption-gpu"
GPU_OVERLOAD_TRACE_DIR = REPO_ROOT / "/home/asurite.ad.asu.edu/dprahlad/deterministic-prototyping-agenticCPS/src/HardwareIntegration/evaluation/logs-trace-gpu-overload"
OUT_DIR = REPO_ROOT / "src/HardwareIntegration/evaluation-plots/out/gpu-overload-federates"

FED_LABELS = {
    "d_control": "Driver\nControl",
    "d_monitor": "Driver\nMonitor",
    "c": "Car",
    "adc": "Input\nForCoach",
    "llm": "Agentic\nCoach",
    "planner": "Action\nPlanner",
    "sim": "Envir-\nonment",
}
FED_ORDER = ["d_control", "d_monitor", "c", "adc", "llm", "planner", "sim"]

PLOT_FONT_SIZE = 30
NO_GPU_OVERLOAD_COLOR = "#93c5fd"
GPU_OVERLOAD_COLOR = "#fdba74"
TRACE_LABELS = {
    "no_gpu_overload": "No GPU overload",
    "gpu_overload": "GPU overload",
}


def read_execution_logs(trace_dir: Path) -> pd.DataFrame:
    frames = []
    paths = sorted(trace_dir.glob("federate_execution_times*.csv"))
    if not paths:
        raise FileNotFoundError(f"No federate_execution_times*.csv files found in {trace_dir}")

    for path in paths:
        df = pd.read_csv(path, skipinitialspace=True, engine="python", on_bad_lines="skip")
        if "federate" not in df.columns:
            continue
        df = df[df["federate"].astype(str).isin(FED_LABELS)].copy()
        if df.empty:
            continue
        for col in ["lag_ms", "tardy_invocation_count", "logical_time_ms", "execution_time_ms"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise ValueError(f"No usable federate execution rows found in {trace_dir}")
    return pd.concat(frames, ignore_index=True)


def tardy_count_for(df: pd.DataFrame, federate: str) -> int:
    sub = df[df["federate"].astype(str).eq(federate)]
    if sub.empty or "reaction" not in sub.columns:
        return 0

    if federate == "sim" and "reactor" in sub.columns:
        sub = sub[sub["reactor"].astype(str).eq("Environment")]

    reactions = sub["reaction"].astype(str)
    if federate == "sim":
        return int(reactions.eq("simulation_input_tardy").sum())

    if "tardy_invocation_count" in sub.columns:
        vals = pd.to_numeric(sub["tardy_invocation_count"], errors="coerce").dropna()
        if len(vals):
            return int(vals.max())
    return int(reactions.str.contains("tardy", case=False, na=False).sum())


def p95_lag_for(df: pd.DataFrame, federate: str) -> float:
    sub = df[df["federate"].astype(str).eq(federate)]
    if sub.empty or "lag_ms" not in sub.columns:
        return 0.0
    if federate == "sim" and "reactor" in sub.columns:
        sub = sub[sub["reactor"].astype(str).eq("Environment")]
    vals = pd.to_numeric(sub["lag_ms"], errors="coerce").dropna()
    return float(np.percentile(vals, 95)) if len(vals) else 0.0


def apply_large_bold_style() -> None:
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE,
            "font.weight": "bold",
            "axes.labelsize": PLOT_FONT_SIZE,
            "axes.labelweight": "bold",
            "axes.titlesize": PLOT_FONT_SIZE,
            "axes.titleweight": "bold",
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_FONT_SIZE,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.grid": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(PLOT_FONT_SIZE)
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.margins(x=0.04, y=0.15)
    ax.grid(False)


def save_plot(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_bar(
    labels: list[str],
    series: dict[str, list[float]],
    ylabel: str,
    out_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(22, 11))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.86)
    x = np.arange(len(labels))
    width = 0.24
    group_spacing = width * 1.35
    colors = [NO_GPU_OVERLOAD_COLOR, GPU_OVERLOAD_COLOR]
    all_values = []
    for i, (legend_label, values) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2) * group_spacing
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=colors[i % len(colors)],
            label=legend_label,
        )
        all_values.extend(values)
        ax.bar_label(
            bars,
            labels=[f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}" for v in values],
            padding=8,
            fontsize=PLOT_FONT_SIZE,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel("Federates", fontweight="bold", fontsize=PLOT_FONT_SIZE, labelpad=12)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=PLOT_FONT_SIZE, labelpad=12)
    ax.legend(
        prop={"weight": "bold", "size": PLOT_FONT_SIZE},
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    ymax = max(all_values) if all_values else 0
    ax.set_ylim(0, ymax * 1.4 if ymax > 0 else 1)
    style_axes(ax)
    save_plot(fig, out_dir, stem)


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gpu_overload_federate_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "federate",
                "label",
                "tardy_handler_invocations",
                "lag_p95_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot tardy handler invocation counts and p95 lag by GPU-overload condition."
    )
    parser.add_argument("--no-gpu-overload-trace-dir", type=Path, default=NO_GPU_OVERLOAD_TRACE_DIR)
    parser.add_argument("--gpu-overload-trace-dir", type=Path, default=GPU_OVERLOAD_TRACE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    apply_large_bold_style()
    no_gpu_df = read_execution_logs(args.no_gpu_overload_trace_dir)
    gpu_df = read_execution_logs(args.gpu_overload_trace_dir)
    available_feds = set(no_gpu_df["federate"].astype(str)) | set(gpu_df["federate"].astype(str))
    feds = [fed for fed in FED_ORDER if fed in available_feds]
    labels = [FED_LABELS[fed] for fed in feds]
    no_gpu_tardy_counts = [tardy_count_for(no_gpu_df, fed) for fed in feds]
    gpu_tardy_counts = [tardy_count_for(gpu_df, fed) for fed in feds]
    no_gpu_lag_p95 = [p95_lag_for(no_gpu_df, fed) for fed in feds]
    gpu_lag_p95 = [p95_lag_for(gpu_df, fed) for fed in feds]

    rows = [
        {
            "condition": TRACE_LABELS["no_gpu_overload"],
            "federate": fed,
            "label": FED_LABELS[fed],
            "tardy_handler_invocations": tardy,
            "lag_p95_ms": lag,
        }
        for fed, tardy, lag in zip(feds, no_gpu_tardy_counts, no_gpu_lag_p95)
    ]
    rows.extend(
        {
            "condition": TRACE_LABELS["gpu_overload"],
            "federate": fed,
            "label": FED_LABELS[fed],
            "tardy_handler_invocations": tardy,
            "lag_p95_ms": lag,
        }
        for fed, tardy, lag in zip(feds, gpu_tardy_counts, gpu_lag_p95)
    )
    write_summary(args.out_dir, rows)

    plot_bar(
        labels,
        {
            TRACE_LABELS["no_gpu_overload"]: no_gpu_tardy_counts,
            TRACE_LABELS["gpu_overload"]: gpu_tardy_counts,
        },
        "Number of tardy handler invocations",
        args.out_dir,
        "gpu_overload_tardy_handler_invocations_by_federate",
    )
    plot_bar(
        labels,
        {
            TRACE_LABELS["no_gpu_overload"]: no_gpu_lag_p95,
            TRACE_LABELS["gpu_overload"]: gpu_lag_p95,
        },
        "p95 lag (ms)",
        args.out_dir,
        "gpu_overload_lag_p95_by_federate",
    )

    print(f"Wrote plots and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
