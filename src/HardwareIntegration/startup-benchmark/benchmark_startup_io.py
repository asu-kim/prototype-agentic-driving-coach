#!/usr/bin/env python3
"""Benchmark camera startup/read and MuJoCo startup rendering latency.

This script is intentionally standalone. It does not run Lingua Franca or modify
reactor behavior; it repeats the same low-level startup operations used by the
hardware-integrated flow and writes CSV detail/summary files.
"""

import argparse
import csv
import os
import statistics
import time
from datetime import datetime
from pathlib import Path


def now_iso():
    return datetime.now().isoformat(timespec="microseconds")


def ms_since(start):
    return (time.perf_counter() - start) * 1000.0


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int((p / 100.0) * len(values)) - 1
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def summary_row(name, values):
    ok_values = [v for v in values if v >= 0.0]
    if not ok_values:
        return {
            "metric": name,
            "runs_ok": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "metric": name,
        "runs_ok": len(ok_values),
        "mean_ms": statistics.mean(ok_values),
        "median_ms": statistics.median(ok_values),
        "p95_ms": percentile(ok_values, 95),
        "p99_ms": percentile(ok_values, 99),
        "max_ms": max(ok_values),
    }


def backend_value(name):
    import cv2

    normalized = name.strip().upper()
    if normalized == "ANY":
        return 0
    if normalized == "V4L2":
        return cv2.CAP_V4L2
    if normalized == "DSHOW":
        return cv2.CAP_DSHOW
    if normalized == "AVFOUNDATION":
        return cv2.CAP_AVFOUNDATION
    raise ValueError(f"Unsupported camera backend: {name}")


def benchmark_camera(args, output_dir):
    import cv2

    detail_path = output_dir / "camera_startup_detail.csv"
    summary_path = output_dir / "camera_startup_summary.csv"
    backend = backend_value(args.camera_backend)
    metrics = {
        "open_ms": [],
        "set_properties_ms": [],
        "first_read_ms": [],
        "startup_total_ms": [],
        "release_ms": [],
    }

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "run_id",
            "camera",
            "backend",
            "status",
            "open_ms",
            "set_properties_ms",
            "first_read_ms",
            "startup_total_ms",
            "release_ms",
            "frame_width",
            "frame_height",
            "error",
        ])

        cv2.setNumThreads(1)
        for run_id in range(args.camera_runs):
            cap = None
            status = "ok"
            error = ""
            open_ms = set_properties_ms = first_read_ms = startup_total_ms = release_ms = -1.0
            frame_width = frame_height = 0
            total_start = time.perf_counter()

            try:
                t0 = time.perf_counter()
                cap = cv2.VideoCapture(args.camera, backend)
                open_ms = ms_since(t0)

                if not cap.isOpened():
                    status = "error"
                    error = "camera_not_opened"
                else:
                    t0 = time.perf_counter()
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
                    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
                    set_properties_ms = ms_since(t0)

                    t0 = time.perf_counter()
                    ok, frame = cap.read()
                    first_read_ms = ms_since(t0)
                    if not ok or frame is None:
                        status = "error"
                        error = "first_read_failed"
                    else:
                        frame_height, frame_width = frame.shape[:2]

                startup_total_ms = ms_since(total_start)
            except Exception as exc:
                status = "error"
                error = str(exc)
                startup_total_ms = ms_since(total_start)
            finally:
                if cap is not None:
                    t0 = time.perf_counter()
                    cap.release()
                    release_ms = ms_since(t0)

            row_values = {
                "open_ms": open_ms,
                "set_properties_ms": set_properties_ms,
                "first_read_ms": first_read_ms,
                "startup_total_ms": startup_total_ms,
                "release_ms": release_ms,
            }
            if status == "ok":
                for name, value in row_values.items():
                    metrics[name].append(value)

            writer.writerow([
                now_iso(),
                run_id,
                args.camera,
                args.camera_backend,
                status,
                open_ms,
                set_properties_ms,
                first_read_ms,
                startup_total_ms,
                release_ms,
                frame_width,
                frame_height,
                error,
            ])
            f.flush()
            print(
                f"[camera] {run_id + 1}/{args.camera_runs}: "
                f"startup={startup_total_ms:.2f} ms read={first_read_ms:.2f} ms {status}",
                flush=True,
            )
            if args.sleep_between_runs_ms > 0:
                time.sleep(args.sleep_between_runs_ms / 1000.0)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "runs_ok", "mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric, values in metrics.items():
            writer.writerow(summary_row(metric, values))

    return detail_path, summary_path


def benchmark_mujoco_offscreen(args, output_dir):
    import mujoco

    detail_path = output_dir / "mujoco_startup_render_detail.csv"
    summary_path = output_dir / "mujoco_startup_render_summary.csv"
    render_mode = args.mujoco_render_mode
    metrics = {
        "model_load_ms": [],
        "data_create_ms": [],
        "mj_forward_ms": [],
        "renderer_or_viewer_create_ms": [],
        "first_render_or_sync_ms": [],
        "startup_total_ms": [],
        "close_ms": [],
    }

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "run_id",
            "model_file",
            "mode",
            "status",
            "model_load_ms",
            "data_create_ms",
            "mj_forward_ms",
            "renderer_or_viewer_create_ms",
            "first_render_or_sync_ms",
            "startup_total_ms",
            "close_ms",
            "image_width",
            "image_height",
            "error",
        ])

        for run_id in range(args.mujoco_runs):
            renderer = None
            viewer = None
            status = "ok"
            error = ""
            model_load_ms = data_create_ms = mj_forward_ms = -1.0
            renderer_or_viewer_create_ms = first_render_or_sync_ms = startup_total_ms = close_ms = -1.0
            image_width = image_height = 0
            total_start = time.perf_counter()

            try:
                t0 = time.perf_counter()
                model = mujoco.MjModel.from_xml_path(str(args.mujoco_model))
                model_load_ms = ms_since(t0)

                t0 = time.perf_counter()
                data = mujoco.MjData(model)
                data_create_ms = ms_since(t0)

                t0 = time.perf_counter()
                mujoco.mj_forward(model, data)
                mj_forward_ms = ms_since(t0)

                if render_mode == "offscreen":
                    t0 = time.perf_counter()
                    renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
                    renderer_or_viewer_create_ms = ms_since(t0)

                    t0 = time.perf_counter()
                    renderer.update_scene(data, camera=args.mujoco_camera)
                    image = renderer.render()
                    first_render_or_sync_ms = ms_since(t0)
                    image_height, image_width = image.shape[:2]
                else:
                    import glfw
                    import mujoco.viewer

                    glfw.init()
                    glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
                    t0 = time.perf_counter()
                    viewer = mujoco.viewer.launch_passive(
                        model,
                        data,
                        show_left_ui=False,
                        show_right_ui=False,
                    )
                    renderer_or_viewer_create_ms = ms_since(t0)

                    t0 = time.perf_counter()
                    viewer.sync()
                    first_render_or_sync_ms = ms_since(t0)
                    if getattr(viewer, "viewport", None) is not None:
                        image_width = getattr(viewer.viewport, "width", 0)
                        image_height = getattr(viewer.viewport, "height", 0)

                startup_total_ms = ms_since(total_start)
            except Exception as exc:
                status = "error"
                error = str(exc)
                startup_total_ms = ms_since(total_start)
            finally:
                if renderer is not None:
                    t0 = time.perf_counter()
                    renderer.close()
                    close_ms = ms_since(t0)
                if viewer is not None:
                    t0 = time.perf_counter()
                    viewer.close()
                    close_ms = ms_since(t0)

            row_values = {
                "model_load_ms": model_load_ms,
                "data_create_ms": data_create_ms,
                "mj_forward_ms": mj_forward_ms,
                "renderer_or_viewer_create_ms": renderer_or_viewer_create_ms,
                "first_render_or_sync_ms": first_render_or_sync_ms,
                "startup_total_ms": startup_total_ms,
                "close_ms": close_ms,
            }
            if status == "ok":
                for name, value in row_values.items():
                    metrics[name].append(value)

            writer.writerow([
                now_iso(),
                run_id,
                str(args.mujoco_model),
                render_mode,
                status,
                model_load_ms,
                data_create_ms,
                mj_forward_ms,
                renderer_or_viewer_create_ms,
                first_render_or_sync_ms,
                startup_total_ms,
                close_ms,
                image_width,
                image_height,
                error,
            ])
            f.flush()
            print(
                f"[mujoco] {run_id + 1}/{args.mujoco_runs}: "
                f"startup={startup_total_ms:.2f} ms first_render_or_sync={first_render_or_sync_ms:.2f} ms {status}"
                f"{(' error=' + error) if error else ''}",
                flush=True,
            )
            if args.sleep_between_runs_ms > 0:
                time.sleep(args.sleep_between_runs_ms / 1000.0)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "runs_ok", "mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric, values in metrics.items():
            writer.writerow(summary_row(metric, values))

    return detail_path, summary_path


def main():
    repo_root = Path(__file__).resolve().parents[3]
    default_output_dir = Path(__file__).resolve().parent
    default_model = repo_root / "src/HardwareIntegration/models/car.xml"

    parser = argparse.ArgumentParser(
        description="Benchmark camera first-read startup and MuJoCo first-render startup."
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--only", choices=["all", "camera", "mujoco"], default="all")
    parser.add_argument("--sleep-between-runs-ms", type=float, default=100.0)

    parser.add_argument("--camera-runs", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", default="V4L2", choices=["ANY", "V4L2", "DSHOW", "AVFOUNDATION"])
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=10.0)

    parser.add_argument("--mujoco-runs", type=int, default=300)
    parser.add_argument("--mujoco-model", type=Path, default=default_model)
    parser.add_argument("--mujoco-render-mode", choices=["offscreen", "passive-viewer"], default="offscreen")
    parser.add_argument("--mujoco-camera", default=None)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.only in ("all", "camera"):
        outputs.extend(benchmark_camera(args, args.output_dir))
    if args.only in ("all", "mujoco"):
        outputs.extend(benchmark_mujoco_offscreen(args, args.output_dir))

    print("\nWrote:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
