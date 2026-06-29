#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "src/HardwareIntegration/data-collection"
OUT_ROOT = REPO_ROOT / "src/HardwareIntegration/evaluation-plots/out"

NORMAL_TRACE = "logs-trace-deeksha-normal"
NETWORK_TRACE = "logs-trace-netem-150ms-rasp"
GPU_TRACE = "logs-trace-added-another-process-gpu"

MODEL_TRACES = {
    "8B": "logs-trace-8b-700",
    "70B": "logs-trace-70b-700",
    "Phi-4 14B": "logs-trace-phi414b-700",
}

PAPER_STYLE = {
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
}

PALETTE = {
    "Normal": "#2563eb",
    "Network +150 ms": "#dc2626",
    "GPU loaded": "#ea580c",
    "8B": "#16a34a",
    "70B": "#7c3aed",
    "Phi-4 14B": "#0891b2",
    "Reference": "#2563eb",
    "Replay": "#dc2626",
}


def apply_style() -> None:
    plt.rcParams.update(PAPER_STYLE)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    df = pd.read_csv(path, skipinitialspace=True, engine="python", on_bad_lines="skip")
    for column in df.columns:
        if column.endswith("_ms") or column.endswith("_ns") or column in {"microstep", "inference_ms", "execution_time_ms", "lag_ms"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "logical_time_ms" not in df.columns and "logical_time_ns" in df.columns:
        df["logical_time_ms"] = df["logical_time_ns"] / 1e6
    return df


def load_llm_behavior(trace_dir: Path, label: str) -> pd.DataFrame:
    df = read_csv(trace_dir / "llm_behavior.csv")
    df["condition"] = label
    return df


def load_execution_times(trace_dir: Path, label: str) -> pd.DataFrame:
    paths = []
    main = trace_dir / "federate_execution_times.csv"
    if main.exists():
        paths.append(main)
    paths.extend(sorted(trace_dir.glob("federate_execution_times_*.csv")))
    frames = []
    for path in paths:
        df = read_csv(path, required=False)
        if df.empty:
            continue
        df["condition"] = label
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_tardy_rows(trace_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(trace_dir.glob("*.csv")):
        df = read_csv(path, required=False)
        if df.empty:
            continue
        text = df.astype(str).apply(lambda col: col.str.contains("tardy", case=False, na=False))
        if not text.any(axis=None):
            continue
        for _, row in df[text.any(axis=1)].iterrows():
            rows.append({
                "condition": label,
                "file": path.name,
                "reactor": row.get("reactor", path.stem),
                "event": row.get("event", ""),
                "reaction": row.get("reaction", ""),
                "reason": row.get("reason", ""),
                "logical_time_ms": row.get("logical_time_ms", np.nan),
                "lag_ms": row.get("lag_ms", np.nan),
            })
    return pd.DataFrame(rows)


def describe(values: Iterable[float]) -> dict[str, float | int]:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return {"count": 0, "mean": math.nan, "median": math.nan, "std": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
        "p95": float(np.percentile(clean, 95)),
        "max": max(clean),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def boxplot_by_label(ax: plt.Axes, series: dict[str, pd.Series], ylabel: str) -> None:
    labels = list(series)
    values = [pd.to_numeric(series[label], errors="coerce").dropna().to_numpy() for label in labels]
    bp = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(PALETTE.get(label, "#64748b"))
        patch.set_alpha(0.65)
    for i, vals in enumerate(values, start=1):
        if len(vals) == 0:
            continue
        x = np.random.default_rng(7).normal(i, 0.035, len(vals))
        ax.scatter(x, vals, s=12, alpha=0.35, color="#111827", linewidths=0)
    ax.set_ylabel(ylabel)
