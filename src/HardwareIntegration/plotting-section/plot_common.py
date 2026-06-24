#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 4 else Path.cwd()
DEFAULT_ORIGINAL_DIR = REPO_ROOT / "src/HardwareIntegration/data-collection/logs-trace-2"
DEFAULT_REPLAY_DIR = REPO_ROOT / "fed-gen/HardwareIntegratedADCReplay/logs-trace"

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

COLORS = {
    "Collected": "#255C99",
    "Replay": "#D95F02",
    "Current": "#2A9D8F",
    "Reaction": "#2A9D8F",
    "Tardy": "#C33149",
}

DISPLAY_LABELS = {
    "Collected": "Data Collection Mode",
    "Replay": "Data Replay Mode",
    "Current": "Current Mode",
}

def trace_label(trace: str) -> str:
    return DISPLAY_LABELS.get(trace, trace)

def trace_color(trace: str) -> str:
    return COLORS.get(trace, "#555555")

def read_csv(path: Path, label: str, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {label} CSV: {path}")
        print(f"Skipping missing {label} CSV: {path}")
        return None
    df = pd.read_csv(path, skipinitialspace=True, engine="python", on_bad_lines="skip")
    df["trace"] = label
    for col in ["logical_time_ns", "logical_time_ms", "physical_time_ms", "microstep", "execution_time_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "logical_time_ms" not in df.columns and "logical_time_ns" in df.columns:
        df["logical_time_ms"] = df["logical_time_ns"] / 1e6
    return df

def read_pair(original_dir: Path, replay_dir: Path, filename: str, required: bool = False) -> pd.DataFrame | None:
    frames = []
    for d, label in [(original_dir, "Collected"), (replay_dir, "Replay")]:
        df = read_csv(d / filename, label, required=required)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()

def load_execution(original_dir: Path, replay_dir: Path) -> pd.DataFrame | None:
    frames = []
    for d, label in [(original_dir, "Collected"), (replay_dir, "Replay")]:
        main = d / "federate_execution_times.csv"
        paths = [main] if main.exists() else sorted(d.glob("federate_execution_times_*.csv"))
        for p in paths:
            df = read_csv(p, label, required=False)
            if df is not None:
                frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "execution_time_ms" in df.columns:
        df["execution_time_ms"] = pd.to_numeric(df["execution_time_ms"], errors="coerce")
    return df

def load_federate_lag(original_dir: Path, replay_dir: Path) -> pd.DataFrame | None:
    frames = []
    for d, label in [(original_dir, "Collected"), (replay_dir, "Replay")]:
        for p in sorted(d.glob("federate__*_main_*.csv")):
            df = pd.read_csv(p, skipinitialspace=True, engine="python", on_bad_lines="skip")
            if "Elapsed Logical Time" not in df.columns or "Elapsed Physical Time" not in df.columns:
                continue
            df["trace"] = label
            df["federate"] = p.name.replace("federate__", "").replace(".csv", "")
            df["logical_raw_ms"] = pd.to_numeric(df["Elapsed Logical Time"], errors="coerce") / 1e6
            df["physical_raw_ms"] = pd.to_numeric(df["Elapsed Physical Time"], errors="coerce") / 1e6
            df = df.dropna(subset=["logical_raw_ms", "physical_raw_ms"])
            if df.empty:
                continue
            df["logical_time_ms"] = df["logical_raw_ms"] - df["logical_raw_ms"].iloc[0]
            df["physical_time_ms"] = df["physical_raw_ms"] - df["physical_raw_ms"].iloc[0]
            df["lag_ms"] = df["physical_time_ms"] - df["logical_time_ms"]
            frames.append(df[["trace", "federate", "logical_time_ms", "physical_time_ms", "lag_ms"]])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)
