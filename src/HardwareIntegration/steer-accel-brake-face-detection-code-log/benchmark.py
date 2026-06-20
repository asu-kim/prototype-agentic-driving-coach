#!/usr/bin/env python3
import argparse
import csv
import os
import statistics
import threading
import time
from collections import defaultdict

import cv2
import numpy as np
from evdev import InputDevice, ecodes, list_devices

from face_detection import FaceDetector
def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def stats(values):
    if not values:
        return {}
    values = sorted(values)
    return {
        "runs": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
    }


def find_wheel():
    for p in list_devices():
        try:
            d = InputDevice(p)
            if "shanwan" in d.name.lower() and "gamepad" in d.name.lower():
                return p
        except Exception:
            pass
    return None


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm(dev, code, val):
    info = dev.absinfo(code)
    if info.max == info.min:
        return 0.0
    return (val - info.min) / (info.max - info.min)


def face_thread(args, done, face_latencies):
    detector = FaceDetector()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[FACE] Camera cannot be opened", flush=True)
        return

    cv2.setNumThreads(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    for _ in range(10):
        cap.read()
    ensure_parent_dir(args.face_csv)

    with open(args.face_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "camera_read_ms", "face_detection_ms", "total_ms", "face_found", "head", "eye"])

        i = 0
        while i < args.runs:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()

            if not ok or frame is None:
                continue

            frame = np.ascontiguousarray(frame)

            t2 = time.perf_counter()
            result = detector.process(frame)
            t3 = time.perf_counter()

            read_ms = (t1 - t0) * 1000
            detect_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000

            face_latencies.append(detect_ms)

            writer.writerow([
                i,
                read_ms,
                detect_ms,
                total_ms,
                int(result is not None),
                result["head"] if result else "",
                result["eye"] if result else "",
            ])
            f.flush()

            print(f"[FACE {i+1}/{args.runs}] detect={detect_ms:.2f} ms", flush=True)
            i += 1

    cap.release()
    done["face"] = True


def wheel_thread(args, done, control_latencies):
    path = find_wheel()
    if path is None:
        print("[WHEEL] wheel device not found", flush=True)
        return

    dev = InputDevice(path)
    print(f"[WHEEL] Using {dev.path}: {dev.name}", flush=True)

    xinfo = dev.absinfo(ecodes.ABS_X)
    xmin, xmax = xinfo.min, xinfo.max

    counts = defaultdict(int)
    ensure_parent_dir(args.control_csv)
    with open(args.control_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["control", "run", "raw_value", "normalized_value", "processing_ms"])

        for e in dev.read_loop():
            if all(counts[k] >= args.runs for k in ["steer", "brake", "throttle"]):
                break

            if e.type != ecodes.EV_ABS:
                continue

            t0 = time.perf_counter()

            control = None
            value = None

            if e.code == ecodes.ABS_X and counts["steer"] < args.runs:
                control = "steer"
                value = (2 * (e.value - xmin) / (xmax - xmin)) - 1.0
                value = float(clamp(value, -1.0, 1.0))

            elif e.code == ecodes.ABS_BRAKE and counts["brake"] < args.runs:
                control = "brake"
                value = float(clamp(norm(dev, ecodes.ABS_BRAKE, e.value), 0.0, 1.0))

            elif e.code == ecodes.ABS_GAS and counts["throttle"] < args.runs:
                control = "throttle"
                value = float(clamp(norm(dev, ecodes.ABS_GAS, e.value), 0.0, 1.0))

            if control is None:
                continue

            t1 = time.perf_counter()
            processing_ms = (t1 - t0) * 1000

            counts[control] += 1
            control_latencies[control].append(processing_ms)

            writer.writerow([
                control,
                counts[control],
                e.value,
                value,
                processing_ms,
            ])
            f.flush()

            print(
                f"[{control.upper()} {counts[control]}/{args.runs}] "
                f"latency={processing_ms:.4f} ms value={value:.3f}",
                flush=True,
            )

    done["wheel"] = True

def write_summary(summary_csv, face_latencies, control_latencies):
    rows = []

    def add_row(name, values):
        s = stats(values)
        if not s:
            rows.append([name, 0, "", "", ""])
        else:
            rows.append([
                name,
                s["runs"],
                s["mean_ms"],
                s["median_ms"],
                s["max_ms"],
            ])

    add_row("face_detection", face_latencies)
    add_row("steer", control_latencies["steer"])
    add_row("brake", control_latencies["brake"])
    add_row("throttle", control_latencies["throttle"])
    ensure_parent_dir(summary_csv)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "runs", "mean_ms", "median_ms", "max_ms"])
        writer.writerows(rows)

    print(f"Summary CSV: {summary_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--face-csv", default="face_detection_latency.csv")
    parser.add_argument("--control-csv", default="control_input_latency.csv")
    parser.add_argument("--summary-csv", default="input_latency_summary.csv")
    args = parser.parse_args()

    done = {"face": False, "wheel": False}
    face_latencies = []
    control_latencies = defaultdict(list)

    t_face = threading.Thread(target=face_thread, args=(args, done, face_latencies), daemon=True)
    t_wheel = threading.Thread(target=wheel_thread, args=(args, done, control_latencies), daemon=True)

    t_face.start()
    t_wheel.start()

    t_face.join()
    t_wheel.join()

    write_summary(args.summary_csv, face_latencies, control_latencies)

    print("\n===== SUMMARY =====")
    print("Face detection:", stats(face_latencies))
    print("Steer:", stats(control_latencies["steer"]))
    print("Brake:", stats(control_latencies["brake"]))
    print("Throttle:", stats(control_latencies["throttle"]))


if __name__ == "__main__":
    main()