#!/usr/bin/env python3
"""Generate Section 3 latency/behavior plots from Lingua Franca trace CSVs.

The defaults point at the current collected trace and replay trace in this repo.
Override any path with CLI flags after regenerating aligned experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing Python dependency: {missing}. Install dependencies with:\n"
        f"  python3 -m pip install -r src/HardwareIntegration/plotting-section/requirements.txt"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ORIGINAL_CAR = REPO_ROOT / "src/HardwareIntegration/data-collection/logs-trace-2/car_inputs.csv"
DEFAULT_REPLAY_CAR = REPO_ROOT / "fed-gen/HardwareIntegratedADCReplay/logs-trace/car_inputs.csv"
DEFAULT_ORIGINAL_LLM = REPO_ROOT / "src/HardwareIntegration/data-collection/logs-trace-2/llm_inputs.csv"
DEFAULT_REPLAY_LLM = REPO_ROOT / "fed-gen/HardwareIntegratedADCReplay/logs-trace/llm_inputs.csv"
DEFAULT_ORIGINAL_PLANNER = REPO_ROOT / "src/HardwareIntegration/data-collection/logs-trace-2/planner_events.csv"
DEFAULT_REPLAY_PLANNER = REPO_ROOT / "fed-gen/HardwareIntegratedADCReplay/logs-trace/planner_events.csv"
DEFAULT_ORIGINAL_ENV = REPO_ROOT / "src/HardwareIntegration/data-collection/logs-trace-2/sim_environment_inputs.csv"
DEFAULT_REPLAY_ENV = REPO_ROOT / "fed-gen/HardwareIntegratedADCReplay/logs-trace/sim_environment_inputs.csv"
DEFAULT_ORIGINAL_EXEC = REPO_ROOT / "src/HardwareIntegration/data-collection/logs-trace-2/federate_execution_times.csv"
DEFAULT_REPLAY_EXEC = REPO_ROOT / "fed-gen/HardwareIntegratedADCReplay/logs-trace/federate_execution_times.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "figures"


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
    "Tardy": "#C33149",
    "Reaction": "#2A9D8F",
    "Actuation": "#6A4C93",
    "Warning": "#E9C46A",
    "None": "#8D99AE",
    "Accel": "#2A9D8F",
    "Brake": "#C33149",
    "Hold": "#457B9D",
}


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} CSV: {path}")
    df = pd.read_csv(path)
    df["trace"] = label
    for col in ["logical_time_ns", "logical_time_ms", "physical_time_ms", "microstep"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "logical_time_ms" not in df.columns and "logical_time_ns" in df.columns:
        df["logical_time_ms"] = df["logical_time_ns"] / 1e6
    return df


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def present_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["steer_in_present", "accel_in_present", "brake_in_present", "actuate_in_present"] if c in df.columns]


def value_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["steer_in", "accel_in", "brake_in", "actuate_in", "car_velocity", "car_steer"] if c in df.columns]


def write_summary_tables(car: pd.DataFrame, exec_df: pd.DataFrame | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, group in car.groupby("trace", sort=False):
        logical = group["logical_time_ms"].dropna()
        rows.append({
            "trace": label,
            "rows": len(group),
            "unique_logical_tags": group["logical_time_ns"].nunique() if "logical_time_ns" in group.columns else np.nan,
            "start_logical_ms": logical.min() if not logical.empty else np.nan,
            "end_logical_ms": logical.max() if not logical.empty else np.nan,
            "duration_s": (logical.max() - logical.min()) / 1000 if len(logical) else np.nan,
            "reaction_rows": int((group.get("event", pd.Series(dtype=str)) == "reaction").sum()),
            "tardy_rows": int((group.get("event", pd.Series(dtype=str)) == "tardy").sum()),
            "max_microstep": group["microstep"].max() if "microstep" in group.columns else np.nan,
            "mean_physical_lag_ms": (group["physical_time_ms"] - group["logical_time_ms"]).mean() if "physical_time_ms" in group.columns else np.nan,
            "p95_physical_lag_ms": (group["physical_time_ms"] - group["logical_time_ms"]).quantile(0.95) if "physical_time_ms" in group.columns else np.nan,
        })
    pd.DataFrame(rows).to_csv(out_dir / "summary_metrics.csv", index=False)

    trigger_counts = (
        car.groupby(["trace", "event"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["trace", "event"])
    )
    trigger_counts.to_csv(out_dir / "reaction_trigger_counts.csv", index=False)

    tardy = car[car.get("event", pd.Series(dtype=str)).eq("tardy")].copy()
    if not tardy.empty:
        cols = ["trace", "logical_time_ms", "microstep", "event"] + present_cols(tardy) + value_cols(tardy)
        tardy[cols].to_csv(out_dir / "tardy_events.csv", index=False)

    if exec_df is not None and not exec_df.empty:
        summary = (
            exec_df.groupby(["trace", "federate", "reactor", "reaction"], dropna=False)["execution_time_ms"]
            .agg(count="count", mean_ms="mean", median_ms="median", p95_ms=lambda s: s.quantile(0.95), max_ms="max")
            .reset_index()
            .sort_values(["trace", "federate", "reactor", "reaction"])
        )
        summary.to_csv(out_dir / "execution_time_summary.csv", index=False)


def plot_reaction_triggers(car: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)
    event_to_y = {"reaction": 0, "tardy": 1}
    for ax, (label, group) in zip(axes, car.groupby("trace", sort=False)):
        y = group["event"].map(event_to_y).fillna(0)
        colors = group["event"].map({"reaction": COLORS["Reaction"], "tardy": COLORS["Tardy"]}).fillna("#777777")
        ax.scatter(group["logical_time_ms"] / 1000, y, s=16, c=colors, alpha=0.8, edgecolors="none")
        ax.set_title(f"{label}: Car reaction trigger timeline")
        ax.set_yticks([0, 1], ["reaction", "tardy"])
        ax.set_ylim(-0.5, 1.5)
        ax.set_ylabel("event")
    axes[-1].set_xlabel("logical time (s)")
    save_figure(fig, out_dir, "01_reaction_trigger_timeline")

    counts = car.groupby(["trace", "event"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    counts.plot(kind="bar", stacked=True, color=[COLORS.get("Reaction"), COLORS.get("Tardy")], ax=ax)
    ax.set_title("Car reaction and tardy event counts")
    ax.set_xlabel("")
    ax.set_ylabel("rows")
    ax.tick_params(axis="x", rotation=0)
    save_figure(fig, out_dir, "02_reaction_trigger_counts")


def plot_car_behavior(car: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 8.0), sharex=True)
    for label, group in car.groupby("trace", sort=False):
        t = group["logical_time_ms"] / 1000
        axes[0].plot(t, pd.to_numeric(group.get("car_velocity"), errors="coerce"), label=label, linewidth=1.6, color=COLORS[label])
        axes[1].plot(t, pd.to_numeric(group.get("accel_in"), errors="coerce"), label=label, linewidth=1.3, color=COLORS[label])
        axes[2].plot(t, pd.to_numeric(group.get("brake_in"), errors="coerce"), label=label, linewidth=1.3, color=COLORS[label])
        axes[3].step(t, pd.to_numeric(group.get("car_steer"), errors="coerce"), where="post", label=label, linewidth=1.3, color=COLORS[label])
    axes[0].set_ylabel("velocity\n(km/h)")
    axes[1].set_ylabel("accelerator")
    axes[2].set_ylabel("brake")
    axes[3].set_ylabel("steer")
    axes[3].set_xlabel("logical time (s)")
    axes[0].set_title("Car behavior under collected and replayed inputs")
    for ax in axes:
        ax.legend(loc="best")
    save_figure(fig, out_dir, "03_car_behavior_comparison")


def plot_logical_time_alignment(car: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for label, group in car.groupby("trace", sort=False):
        row_index = np.arange(len(group))
        ax.plot(row_index, group["logical_time_ms"] / 1000, label=label, linewidth=1.5, color=COLORS[label])
    ax.set_title("Logical time coverage by CSV row")
    ax.set_xlabel("CSV data row index")
    ax.set_ylabel("logical time (s)")
    ax.legend()
    save_figure(fig, out_dir, "04_logical_time_by_row")

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    for label, group in car.groupby("trace", sort=False):
        lag = group["physical_time_ms"] - group["logical_time_ms"]
        ax.plot(group["logical_time_ms"] / 1000, lag, label=label, linewidth=1.2, color=COLORS[label])
    ax.set_title("Physical minus logical time lag at Car")
    ax.set_xlabel("logical time (s)")
    ax.set_ylabel("lag (ms)")
    ax.legend()
    save_figure(fig, out_dir, "05_physical_logical_lag")


def plot_execution_times(exec_df: pd.DataFrame | None, out_dir: Path) -> None:
    if exec_df is None or exec_df.empty:
        return
    exec_df = exec_df.dropna(subset=["execution_time_ms"]).copy()
    if exec_df.empty:
        return

    top = (
        exec_df.groupby(["trace", "reactor", "reaction"], dropna=False)["execution_time_ms"]
        .agg(mean_ms="mean", p95_ms=lambda s: s.quantile(0.95), count="count")
        .reset_index()
    )
    top["label"] = top["reactor"].astype(str) + "\n" + top["reaction"].astype(str)

    order = (
        top.groupby("label")["p95_ms"]
        .max()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    plot_df = top[top["label"].isin(order)].copy()
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=order[::-1], ordered=True)

    fig, ax = plt.subplots(figsize=(8.6, max(4.2, 0.38 * len(order))))
    y = np.arange(len(order))
    width = 0.36
    labels = order[::-1]
    for offset, trace in [(-width / 2, "Collected"), (width / 2, "Replay")]:
        vals = []
        for item in labels:
            match = plot_df[(plot_df["trace"] == trace) & (plot_df["label"].astype(str) == item)]
            vals.append(float(match["p95_ms"].iloc[0]) if not match.empty else 0.0)
        ax.barh(y + offset, vals, height=width, label=trace, color=COLORS[trace])
    ax.set_yticks(y, labels)
    ax.set_xlabel("p95 execution time (ms)")
    ax.set_title("Reaction execution-time comparison")
    ax.legend()
    save_figure(fig, out_dir, "06_execution_time_p95_bars")

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for trace, group in exec_df.groupby("trace", sort=False):
        ax.hist(group["execution_time_ms"], bins=40, alpha=0.55, label=trace, color=COLORS[trace])
    ax.set_title("Execution-time distribution across traced reactions")
    ax.set_xlabel("execution time (ms)")
    ax.set_ylabel("reaction count")
    ax.legend()
    save_figure(fig, out_dir, "07_execution_time_distribution")


def plot_clock_period(car: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for label, group in car[car["event"].eq("reaction")].groupby("trace", sort=False):
        times = group["logical_time_ms"].dropna().drop_duplicates().sort_values()
        periods = times.diff().dropna()
        for p in periods:
            rows.append({"trace": label, "period_ms": p, "frequency_hz": 1000.0 / p if p > 0 else np.nan})
    periods_df = pd.DataFrame(rows)
    if periods_df.empty:
        return
    periods_df.to_csv(out_dir / "clock_period_summary_samples.csv", index=False)
    summary = periods_df.groupby("trace").agg(
        mean_period_ms=("period_ms", "mean"),
        median_period_ms=("period_ms", "median"),
        p95_period_ms=("period_ms", lambda s: s.quantile(0.95)),
        mean_frequency_hz=("frequency_hz", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "clock_period_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for label, group in periods_df.groupby("trace", sort=False):
        ax.hist(group["period_ms"], bins=25, alpha=0.6, label=label, color=COLORS[label])
    ax.set_title("Observed logical clock period at Car reactions")
    ax.set_xlabel("period between reaction logical tags (ms)")
    ax.set_ylabel("count")
    ax.legend()
    save_figure(fig, out_dir, "08_clock_period_histogram")


def load_execution(original_path: Path, replay_path: Path) -> pd.DataFrame | None:
    frames = []
    for path, label in [(original_path, "Collected"), (replay_path, "Replay")]:
        if path.exists():
            df = read_csv(path, label)
            if "execution_time_ms" in df.columns:
                df["execution_time_ms"] = pd.to_numeric(df["execution_time_ms"], errors="coerce")
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_optional_pair(original_path: Path, replay_path: Path, name: str) -> pd.DataFrame | None:
    frames = []
    for path, label in [(original_path, "Collected"), (replay_path, "Replay")]:
        if path.exists():
            frames.append(read_csv(path, label))
        else:
            print(f"Skipping missing {label} {name} CSV: {path}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def write_decision_summary_tables(llm: pd.DataFrame | None, planner: pd.DataFrame | None, env: pd.DataFrame | None, out_dir: Path) -> None:
    if llm is not None and not llm.empty:
        rows = []
        for label, group in llm.groupby("trace", sort=False):
            output_ticks = group[group.get("event", pd.Series(dtype=str)).eq("llm_output_tick")]
            rows.append({
                "trace": label,
                "rows": len(group),
                "llm_output_ticks": len(output_ticks),
                "mean_input_lag_ms": (group["physical_time_ms"] - group["logical_time_ms"]).mean(),
                "p95_input_lag_ms": (group["physical_time_ms"] - group["logical_time_ms"]).quantile(0.95),
            })
        pd.DataFrame(rows).to_csv(out_dir / "llm_input_summary.csv", index=False)

    if planner is not None and not planner.empty:
        planner.groupby(["trace", "event"], dropna=False).size().reset_index(name="count").to_csv(out_dir / "planner_event_counts.csv", index=False)
        planner.groupby(["trace", "from_mode", "to_mode"], dropna=False).size().reset_index(name="count").to_csv(out_dir / "planner_mode_transition_counts.csv", index=False)
        cols = ["trace", "logical_time_ms", "event", "mode", "from_mode", "to_mode", "control_token", "instruction", "act_command", "lane_command", "act_set"]
        available = [c for c in cols if c in planner.columns]
        planner[available].to_csv(out_dir / "planner_decision_timeline.csv", index=False)

    if env is not None and not env.empty and "road_phase" in env.columns:
        env.groupby(["trace", "road_phase"], dropna=False).size().reset_index(name="count").to_csv(out_dir / "environment_phase_counts.csv", index=False)


def plot_llm_inputs(llm: pd.DataFrame | None, out_dir: Path) -> None:
    if llm is None or llm.empty:
        return
    llm = llm.copy()
    for col in ["environment_distance", "car_velocity", "head", "eye"]:
        if col in llm.columns:
            llm[col] = pd.to_numeric(llm[col], errors="coerce")

    output = llm[llm.get("event", pd.Series(dtype=str)).eq("llm_output_tick")]
    if output.empty:
        output = llm

    fig, axes = plt.subplots(4, 1, figsize=(8.4, 8.0), sharex=True)
    for label, group in output.groupby("trace", sort=False):
        t = group["logical_time_ms"] / 1000
        color = COLORS[label]
        axes[0].plot(t, group.get("environment_distance"), label=label, linewidth=1.5, color=color)
        axes[1].plot(t, group.get("car_velocity"), label=label, linewidth=1.5, color=color)
        axes[2].step(t, group.get("head"), where="post", label=label, linewidth=1.3, color=color)
        axes[3].step(t, group.get("eye"), where="post", label=label, linewidth=1.3, color=color)
    axes[0].set_title("Inputs presented to the LLM")
    axes[0].set_ylabel("target\ndistance (m)")
    axes[1].set_ylabel("velocity\n(km/h)")
    axes[2].set_ylabel("head")
    axes[3].set_ylabel("eye")
    axes[3].set_xlabel("logical time (s)")
    axes[2].set_yticks([0, 1, 2], ["left", "center", "right"])
    axes[3].set_yticks([0, 1, 2], ["left", "center", "right"])
    for ax in axes:
        ax.legend(loc="best")
    save_figure(fig, out_dir, "09_llm_input_snapshot")

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    for label, group in llm.groupby("trace", sort=False):
        lag = group["physical_time_ms"] - group["logical_time_ms"]
        ax.plot(group["logical_time_ms"] / 1000, lag, label=label, linewidth=1.2, color=COLORS[label])
    ax.set_title("Physical minus logical time lag at InputForCoach")
    ax.set_xlabel("logical time (s)")
    ax.set_ylabel("lag (ms)")
    ax.legend()
    save_figure(fig, out_dir, "10_llm_input_lag")


def plot_environment_context(env: pd.DataFrame | None, out_dir: Path) -> None:
    if env is None or env.empty:
        return
    env = env.copy()
    for col in ["environment_distance", "velocity_kmh", "other_car_present"]:
        if col in env.columns:
            env[col] = pd.to_numeric(env[col], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.2), sharex=True)
    phase_names = sorted(str(x) for x in env["road_phase"].dropna().unique()) if "road_phase" in env.columns else []
    phase_to_y = {name: i for i, name in enumerate(phase_names)}
    for label, group in env.groupby("trace", sort=False):
        t = group["logical_time_ms"] / 1000
        color = COLORS[label]
        axes[0].plot(t, group.get("environment_distance"), label=label, linewidth=1.4, color=color)
        if phase_to_y:
            axes[1].step(t, group["road_phase"].astype(str).map(phase_to_y), where="post", label=label, linewidth=1.2, color=color)
        axes[2].step(t, group.get("other_car_present"), where="post", label=label, linewidth=1.2, color=color)
    axes[0].set_title("Environment context sent toward the coach")
    axes[0].set_ylabel("target\ndistance (m)")
    axes[1].set_ylabel("road phase")
    if phase_to_y:
        axes[1].set_yticks(list(phase_to_y.values()), list(phase_to_y.keys()))
    axes[2].set_ylabel("other car")
    axes[2].set_xlabel("logical time (s)")
    for ax in axes:
        ax.legend(loc="best")
    save_figure(fig, out_dir, "11_environment_context")


def plot_planner_decisions(planner: pd.DataFrame | None, out_dir: Path) -> None:
    if planner is None or planner.empty:
        return
    planner = planner.copy()
    token_order = ["NONE", "WARNING", "ACTUATE"]
    token_to_y = {name: i for i, name in enumerate(token_order)}
    mode_order = ["MONITORING", "WARNING", "ACTUATE"]
    mode_to_y = {name: i for i, name in enumerate(mode_order)}
    act_order = ["STOP", "BRAKE", "HOLD", "ACCEL"]
    act_to_y = {name: i for i, name in enumerate(act_order)}

    transitions = planner[planner.get("event", pd.Series(dtype=str)).eq("mode_transition")]
    act_rows = planner[planner.get("act_set", pd.Series(dtype=str)).astype(str).eq("1")]

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.8), sharex=True)
    for label, group in planner.groupby("trace", sort=False):
        token_group = group[group["control_token"].notna()] if "control_token" in group.columns else group
        axes[0].scatter(token_group["logical_time_ms"] / 1000, token_group["control_token"].astype(str).map(token_to_y), s=18, alpha=0.75, label=label, color=COLORS[label], edgecolors="none")
    for label, group in transitions.groupby("trace", sort=False):
        axes[1].scatter(group["logical_time_ms"] / 1000, group["to_mode"].astype(str).map(mode_to_y), s=24, alpha=0.8, label=label, color=COLORS[label], edgecolors="none")
    for label, group in act_rows.groupby("trace", sort=False):
        axes[2].scatter(group["logical_time_ms"] / 1000, group["act_command"].astype(str).map(act_to_y), s=32, alpha=0.9, label=label, color=COLORS[label], edgecolors="none")
    axes[0].set_title("LLM response tokens and planner decisions")
    axes[0].set_ylabel("LLM token")
    axes[0].set_yticks(list(token_to_y.values()), list(token_to_y.keys()))
    axes[1].set_ylabel("planner mode")
    axes[1].set_yticks(list(mode_to_y.values()), list(mode_to_y.keys()))
    axes[2].set_ylabel("act command")
    axes[2].set_yticks(list(act_to_y.values()), list(act_to_y.keys()))
    axes[2].set_xlabel("logical time (s)")
    for ax in axes:
        ax.legend(loc="best")
    save_figure(fig, out_dir, "12_llm_response_planner_timeline")

    counts = planner.groupby(["trace", "control_token"], dropna=False).size().unstack(fill_value=0)
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        counts.plot(kind="bar", ax=ax)
        ax.set_title("LLM control token counts observed by planner")
        ax.set_xlabel("")
        ax.set_ylabel("rows")
        ax.tick_params(axis="x", rotation=0)
        save_figure(fig, out_dir, "13_llm_control_token_counts")

    act_counts = act_rows.groupby(["trace", "act_command"], dropna=False).size().unstack(fill_value=0)
    if not act_counts.empty:
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        act_counts.plot(kind="bar", ax=ax)
        ax.set_title("Planner act command counts")
        ax.set_xlabel("")
        ax.set_ylabel("act command rows")
        ax.tick_params(axis="x", rotation=0)
        save_figure(fig, out_dir, "14_planner_act_command_counts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latency section plots and tables.")
    parser.add_argument("--original-car", type=Path, default=DEFAULT_ORIGINAL_CAR)
    parser.add_argument("--replay-car", type=Path, default=DEFAULT_REPLAY_CAR)
    parser.add_argument("--original-llm", type=Path, default=DEFAULT_ORIGINAL_LLM)
    parser.add_argument("--replay-llm", type=Path, default=DEFAULT_REPLAY_LLM)
    parser.add_argument("--original-planner", type=Path, default=DEFAULT_ORIGINAL_PLANNER)
    parser.add_argument("--replay-planner", type=Path, default=DEFAULT_REPLAY_PLANNER)
    parser.add_argument("--original-env", type=Path, default=DEFAULT_ORIGINAL_ENV)
    parser.add_argument("--replay-env", type=Path, default=DEFAULT_REPLAY_ENV)
    parser.add_argument("--original-exec", type=Path, default=DEFAULT_ORIGINAL_EXEC)
    parser.add_argument("--replay-exec", type=Path, default=DEFAULT_REPLAY_EXEC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--show", action="store_true", help="Show plots interactively after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update(PAPER_STYLE)

    original_car = read_csv(args.original_car, "Collected")
    replay_car = read_csv(args.replay_car, "Replay")
    car = pd.concat([original_car, replay_car], ignore_index=True)
    llm = load_optional_pair(args.original_llm, args.replay_llm, "LLM input")
    planner = load_optional_pair(args.original_planner, args.replay_planner, "planner events")
    env = load_optional_pair(args.original_env, args.replay_env, "environment")
    exec_df = load_execution(args.original_exec, args.replay_exec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_tables(car, exec_df, args.out_dir)
    write_decision_summary_tables(llm, planner, env, args.out_dir)
    plot_reaction_triggers(car, args.out_dir)
    plot_car_behavior(car, args.out_dir)
    plot_logical_time_alignment(car, args.out_dir)
    plot_execution_times(exec_df, args.out_dir)
    plot_clock_period(car, args.out_dir)
    plot_llm_inputs(llm, args.out_dir)
    plot_environment_context(env, args.out_dir)
    plot_planner_decisions(planner, args.out_dir)

    print(f"Wrote plots and tables to: {args.out_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
