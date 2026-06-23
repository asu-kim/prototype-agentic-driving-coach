#!/usr/bin/env python3
import argparse
import csv
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



def benchmark_mujoco(args):
    import mujoco

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "mujoco_startup_render_detail.csv"
    summary_path = args.output_dir / "mujoco_startup_render_summary.csv"
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

        for run_id in range(args.runs):
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
                model = mujoco.MjModel.from_xml_path(str(args.model))
                model_load_ms = ms_since(t0)

                t0 = time.perf_counter()
                data = mujoco.MjData(model)
                data_create_ms = ms_since(t0)

                t0 = time.perf_counter()
                mujoco.mj_forward(model, data)
                mj_forward_ms = ms_since(t0)

                if args.render_mode == "offscreen":
                    t0 = time.perf_counter()
                    renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
                    renderer_or_viewer_create_ms = ms_since(t0)

                    t0 = time.perf_counter()
                    renderer.update_scene(data, camera=args.camera)
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
                str(args.model),
                args.render_mode,
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
                f"[mujoco] {run_id + 1}/{args.runs}: "
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

    print("\nWrote:")
    print(detail_path)
    print(summary_path)


def main():
    repo_root = Path(__file__).resolve().parents[3]
    default_output_dir = Path(__file__).resolve().parent / "mujoco-results"
    default_model = repo_root / "src/HardwareIntegration/models/car.xml"

    parser = argparse.ArgumentParser(description="Benchmark MuJoCo startup and first render/sync.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--render-mode", choices=["offscreen", "passive-viewer"], default="passive-viewer")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--sleep-between-runs-ms", type=float, default=100.0)
    args = parser.parse_args()
    benchmark_mujoco(args)


if __name__ == "__main__":
    main()
