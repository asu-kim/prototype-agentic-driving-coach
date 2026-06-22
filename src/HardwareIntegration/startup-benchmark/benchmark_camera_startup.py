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


def benchmark_camera(args):
    import cv2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "camera_startup_detail.csv"
    summary_path = args.output_dir / "camera_startup_summary.csv"
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
        for run_id in range(args.runs):
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
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                    cap.set(cv2.CAP_PROP_FPS, args.fps)
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
                f"[camera] {run_id + 1}/{args.runs}: "
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

    print("\nWrote:")
    print(detail_path)
    print(summary_path)


def main():
    default_output_dir = Path(__file__).resolve().parent / "camera-results"
    parser = argparse.ArgumentParser(description="Benchmark camera startup and first frame read.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", default="V4L2", choices=["ANY", "V4L2", "DSHOW", "AVFOUNDATION"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--sleep-between-runs-ms", type=float, default=100.0)
    args = parser.parse_args()
    benchmark_camera(args)


if __name__ == "__main__":
    main()
