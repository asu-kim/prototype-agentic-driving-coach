#!/usr/bin/env python3
"""Benchmark cold model loading and the first inference after each load."""

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
import types
import urllib.request
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "logs-trace" / "llm_inputs.csv"
DEFAULT_DETAIL = HERE / "llama3_1_8b_cold_load_first_inference_300.csv"
DEFAULT_SUMMARY = HERE / "llama3_1_8b_cold_load_first_inference_summary.csv"
EXISTING_BENCHMARK = HERE / "llm-inference-logged-user-behavior.py"


def load_existing_benchmark():
    """Reuse the existing input loader and system prompt without its Ollama client."""
    sys.modules.setdefault("ollama", types.ModuleType("ollama"))
    spec = importlib.util.spec_from_file_location(
        "llm_inference_logged_user_behavior", EXISTING_BENCHMARK
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def post_json(base_url, endpoint, payload, timeout_seconds):
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def percentile(values, percent):
    """Use the same nearest-rank convention as the existing benchmark."""
    ordered = sorted(values)
    index = int((percent / 100.0) * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def write_summary(path, model_load_values, first_inference_values):
    def row(name, values):
        return [
            name,
            len(values),
            statistics.mean(values),
            statistics.median(values),
            percentile(values, 95),
            max(values),
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["metric", "runs", "mean_ms", "median_ms", "p95_ms", "max_ms"])
        writer.writerow(row("model_load", model_load_values))
        writer.writerow(row("first_inference_after_model_load", first_inference_values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--num-predict", type=int, default=32)
    args = parser.parse_args()

    benchmark = load_existing_benchmark()
    inputs = benchmark.load_llm_inputs(str(args.inputs))

    # Unload a possibly resident model before the first measured request.
    post_json(
        args.ollama_url,
        "/api/generate",
        {"model": args.model, "prompt": "", "keep_alive": 0, "stream": False},
        args.timeout_seconds,
    )

    args.detail_csv.parent.mkdir(parents=True, exist_ok=True)
    model_load_values = []
    first_inference_values = []

    with args.detail_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "timestamp",
                "model",
                "run",
                "input_index",
                "model_load_ms",
                "first_inference_after_model_load_ms",
                "total_wall_ms",
                "ollama_total_ms",
                "prompt_eval_ms",
                "eval_ms",
                "status",
                "error",
            ]
        )

        for index in range(args.runs):
            input_index = index % len(inputs)
            prompt = "INPUT:\n" + json.dumps(inputs[input_index], indent=2) + "\n\nReturn output:"
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": benchmark.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0.1, "num_predict": args.num_predict},
                # Unload after every response so the next run is another cold load.
                "keep_alive": 0,
                "stream": False,
            }

            try:
                started = time.perf_counter()
                response = post_json(
                    args.ollama_url, "/api/chat", payload, args.timeout_seconds
                )
                total_wall_ms = (time.perf_counter() - started) * 1000.0
                model_load_ms = response.get("load_duration", 0) / 1e6
                first_inference_ms = max(0.0, total_wall_ms - model_load_ms)
                model_load_values.append(model_load_ms)
                first_inference_values.append(first_inference_ms)
                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        args.model,
                        index + 1,
                        input_index,
                        model_load_ms,
                        first_inference_ms,
                        total_wall_ms,
                        response.get("total_duration", 0) / 1e6,
                        response.get("prompt_eval_duration", 0) / 1e6,
                        response.get("eval_duration", 0) / 1e6,
                        "ok",
                        "",
                    ]
                )
                print(
                    f"[{index + 1}/{args.runs}] load={model_load_ms:.2f} ms "
                    f"first_inference={first_inference_ms:.2f} ms",
                    flush=True,
                )
            except Exception as error:
                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        args.model,
                        index + 1,
                        input_index,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "error",
                        str(error),
                    ]
                )
                print(f"[{index + 1}/{args.runs}] ERROR: {error}", flush=True)
            output.flush()

    if model_load_values:
        write_summary(args.summary_csv, model_load_values, first_inference_values)

    print(f"Detail CSV: {args.detail_csv}")
    print(f"Summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
