#!/usr/bin/env python3
"""Render environment.csv as an animated top-down road scenario."""

import argparse
import csv
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def car(axis, x, y, color, label, horizontal=False):
    width, height = ((8, 3.6) if horizontal else (3.6, 8))
    body = patches.FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.25",
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )
    axis.add_patch(body)
    axis.text(x, y, label, ha="center", va="center", fontsize=7, color="white", weight="bold", zorder=6)


def draw_road(axis, phase):
    axis.set_xlim(-35, 35)
    axis.set_ylim(-30, 70)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_facecolor("#7fb069")

    axis.add_patch(patches.Rectangle((-12, -30), 24, 100, color="#4b5563", zorder=0))
    axis.plot([0, 0], [-30, 70], color="white", linestyle="--", linewidth=2, zorder=1)
    axis.plot([-12, -12], [-30, 70], color="#f8fafc", linewidth=2)
    axis.plot([12, 12], [-30, 70], color="#f8fafc", linewidth=2)

    if phase in {"STOP_SIGN", "YIELD", "MOVE"}:
        axis.add_patch(patches.Rectangle((-35, 34), 70, 24, color="#4b5563", zorder=0))
        axis.plot([-35, 35], [46, 46], color="white", linestyle="--", linewidth=2)
        axis.plot([-12, 12], [32, 32], color="white", linewidth=4)
        axis.text(15, 29, "STOP LINE", color="white", fontsize=8, weight="bold")

    if phase == "EXIT":
        axis.plot([0, 28], [20, 70], color="#4b5563", linewidth=18, solid_capstyle="butt", zorder=0)
        axis.plot([1, 29], [20, 70], color="white", linestyle="--", linewidth=2, zorder=1)
        axis.text(21, 58, "EXIT", fontsize=10, weight="bold", rotation=55)


def draw_frame(axis, row, index, total):
    axis.clear()
    phase = row["road_phase"]
    draw_road(axis, phase)
    car(axis, -6, 0, "#2563eb", "EGO")

    if int(row["other_car_present"]):
        distance = max(10, min(float(row["other_distance_m"]), 55))
        car(axis, -6, distance, "#dc2626", f"FRONT\n{float(row['other_velocity_kmh']):.0f}")

    if int(row["right_car_present"]):
        distance = float(row["right_car_distance_m"])
        y = max(-22, min(distance, 55))
        if row["other_lane"] == "RIGHT_REAR":
            y = -abs(y)
        elif row["other_lane"] == "RIGHT_SIDE":
            y = 0
        car(axis, 6, y, "#f59e0b", f"RIGHT\n{float(row['right_car_velocity_kmh']):.0f}")

    if int(row["cross_left_car_present"]):
        distance = max(14, min(float(row["cross_left_car_distance_m"]), 30))
        car(axis, -distance, 46, "#7c3aed", f"LEFT\n{float(row['cross_left_car_velocity_kmh']):.0f}", horizontal=True)

    if int(row["cross_right_car_present"]):
        distance = max(14, min(float(row["cross_right_car_distance_m"]), 30))
        car(axis, distance, 46, "#0f766e", f"CROSS\n{float(row['cross_right_car_velocity_kmh']):.0f}", horizontal=True)

    axis.set_title(f"Road Phase: {phase}   |   Step {index + 1}/{total}", fontsize=14, weight="bold")
    axis.text(-33, -27, "Labels show velocity (km/h)", fontsize=8, bbox={"facecolor": "white", "alpha": 0.8})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("src/HardwareIntegration/environment.csv"))
    parser.add_argument("--output", type=Path, default=Path("src/HardwareIntegration/Logs/environment_road.gif"))
    parser.add_argument("--step", type=int, default=5, help="Use every Nth CSV row")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.input)[:: max(1, args.step)]
    figure, axis = plt.subplots(figsize=(7, 10))
    movie = animation.FuncAnimation(
        figure,
        lambda frame: draw_frame(axis, rows[frame], frame, len(rows)),
        frames=len(rows),
        interval=args.interval_ms,
        repeat=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(args.output, writer="pillow", fps=max(1, round(1000 / args.interval_ms)), dpi=100)
    print(f"Saved road animation to {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
