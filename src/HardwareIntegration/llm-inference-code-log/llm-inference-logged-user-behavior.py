#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime

import ollama


DEFAULT_MODELS = [
    "llama3.1:8b",
    "llama3.1:70b",
    "phi4:14b",
]


SYSTEM_PROMPT = """
 You are a real-time driving coach.

                            Treat every INPUT value like authoritative current measurement. Never invent vehicles, hazards, driver actions, or road phases.

                            Apply only the requirements for the current road_phase:

                            - HIGHWAY: Maintain 85-90 km/h. When distance_to_target_m is at most 1000, instruct the driver to prepare to change right for the exit. Do not request a right-lane check before that 1 km point.
                            - LANE_CHANGE: Look right before changing lanes. A right look is satisfied when RIGHT appears in either head_history or eye_history. The right lane is unsafe when right_car_present is 1.
                            - EXIT: Reduce speed toward 45 km/h. distance_to_target_m is distance to the stop sign at the crossroad.
                            - STOP_SIGN: Stop before the sign and look LEFT and RIGHT before moving ahead.
                            - YIELD: Yield until cross traffic has cleared, especially right-side cross traffic.
                            - MOVE: Move ahead for the final 100 m.

                            Choose exactly one control token using this priority:

                            1. ACTUATE for an immediate hazard requiring vehicle control:
                               - front_car_present is 1, or
                               - road_phase is LANE_CHANGE and right_car_present is 1, or
                               - road_phase is STOP_SIGN, distance_to_target_m is at most 5 m, and velocity_kmh is above 5.

                            2. WARNING for a recoverable unmet requirement:
                               - HIGHWAY distance_to_target_m is at most 1000 and the driver has not been told to prepare for the exit,
                               - HIGHWAY velocity is below 85 or above 90 km/h,
                               - LANE_CHANGE has no recent RIGHT look,
                               - EXIT velocity is outside 40-50 km/h, or
                               - STOP_SIGN requires a missing LEFT or RIGHT look.

                            3. NONE when no ACTUATE or WARNING condition applies.

                            Instruction rules:
                            - For low speed, say increase speed. Never say slow down.
                            - For high speed, say reduce speed.
                            - At the HIGHWAY 1 km point, say prepare to change right for the exit.
                            - Do not mention a right-lane vehicle when right_car_present is 0.
                            - For NONE, leave the instruction empty.

                            Output contract:
                            - Decide the control token first, then write the driver-facing instruction yourself.
                            - Return exactly one line and no explanation.
                            - Format: CONTROL|INSTRUCTION
                            - CONTROL must be exactly ACTUATE, WARNING, or NONE.
                            - INSTRUCTION must be a short natural-language command for the driver.
                            - Use exactly one | separator.
                            - Do not output placeholder words such as TOKEN, CONTROL, or INSTRUCTION.
                            - Do not put ACTUATE, WARNING, or NONE in the instruction field.
                            - For NONE, return exactly NONE| with nothing after the separator.

                            Valid examples:
                            WARNING|Reduce speed.
                            WARNING|Increase speed.
                            WARNING|Look right before changing lanes.
                            ACTUATE|Brake for the vehicle ahead.
                            NONE|
"""


def safe_json_loads(value, default):
    try:
        if value is None or value == "":
            return default
        return json.loads(value)
    except Exception:
        return default


def load_llm_inputs(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("event") != "llm_output_tick":
                continue

            env_values = safe_json_loads(row.get("environment_states_json"), [])
            head_history = safe_json_loads(row.get("head_history_json"), ["CENTER"])
            eye_history = safe_json_loads(row.get("eye_history_json"), ["CENTER"])

            environment_input = {
                "road_phase": env_values[0] if len(env_values) > 0 else "",
                "front_car_present": env_values[1] if len(env_values) > 1 else 0,
                "right_car_present": env_values[2] if len(env_values) > 2 else 0,
                "right_car_position": env_values[3] if len(env_values) > 3 else "",
                "cross_left_car_present": env_values[4] if len(env_values) > 4 else 0,
                "cross_right_car_present": env_values[5] if len(env_values) > 5 else 0,
            }

            llm_input = {
                "distance_to_target_m": float(row.get("environment_distance", 0.0)),
                "velocity_kmh": float(row.get("car_velocity", 0.0)),
                "steering": row.get("car_steer", ""),
                "head_history": head_history,
                "eye_history": eye_history,
                "environment": environment_input,
            }

            rows.append(llm_input)

    if not rows:
        raise RuntimeError(f"No llm_output_tick rows found in {path}")

    return rows


def run_once(model, llm_input, num_predict):
    prompt = f"""INPUT:
{json.dumps(llm_input, indent=2)}

Return output:"""

    t0 = time.perf_counter()

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.1,
            "num_predict": num_predict,
        },
        keep_alive="10m",
    )

    t1 = time.perf_counter()
    
    # print(response)
    raw = response.get("message", {}).get("content", "").strip()
    latency_ms = (t1 - t0) * 1000.0
    timing = {
        "ollama_total_ms": response.get("total_duration", 0) / 1e6,
        "model_load_ms": response.get("load_duration", 0) / 1e6,
        "prompt_eval_ms": response.get("prompt_eval_duration", 0) / 1e6,
        "eval_ms": response.get("eval_duration", 0) / 1e6,
    }
    timing["latency_after_model_load_ms"] = max(
        0.0,
        latency_ms - timing["model_load_ms"],
    )
    # print(raw)

    return latency_ms, raw, len(prompt), len(raw), timing


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int((p / 100.0) * len(values)) - 1
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def benchmark_model(model, inputs, runs, num_predict, writer):
    print(f"\n===== Benchmarking {model} =====", flush=True)

    # Warmup
    # Warmup
    print("[warmup]", flush=True)
    warmup_latency_ms, warmup_raw, warmup_prompt_chars, warmup_response_chars, warmup_timing = run_once(
        model, inputs[0], num_predict
    )

    writer.writerow([
        datetime.now().isoformat(),
        model,
        "warmup",
        0,
        "warmup",
        warmup_latency_ms,
        warmup_prompt_chars,
        warmup_response_chars,
        warmup_timing["model_load_ms"],
        warmup_timing["latency_after_model_load_ms"],
        warmup_timing["ollama_total_ms"],
        warmup_timing["prompt_eval_ms"],
        warmup_timing["eval_ms"],
        warmup_raw.replace("\n", " "),
        "",
    ])

    print(
        f"[{model}] warmup latency = {warmup_latency_ms:.2f} ms "
        f"(model load = {warmup_timing['model_load_ms']:.2f} ms, "
        f"after load = {warmup_timing['latency_after_model_load_ms']:.2f} ms)",
        flush=True,
    )

    latencies = []

    for i in range(runs):
        llm_input = inputs[i % len(inputs)]

        try:
            latency_ms, raw, prompt_chars, response_chars, timing = run_once(
                model, llm_input, num_predict
            )
            status = "ok"
            error = ""
        except Exception as e:
            latency_ms = -1.0
            raw = ""
            prompt_chars = 0
            response_chars = 0
            timing = {
                "model_load_ms": 0.0,
                "latency_after_model_load_ms": 0.0,
                "ollama_total_ms": 0.0,
                "prompt_eval_ms": 0.0,
                "eval_ms": 0.0,
            }
            status = "error"
            error = str(e)

        if latency_ms >= 0:
            latencies.append(latency_ms)

        writer.writerow([
            datetime.now().isoformat(),
            model,
            i,
            i % len(inputs),
            status,
            latency_ms,
            prompt_chars,
            response_chars,
            timing["model_load_ms"],
            timing["latency_after_model_load_ms"],
            timing["ollama_total_ms"],
            timing["prompt_eval_ms"],
            timing["eval_ms"],
            raw.replace("\n", " "),
            error,
        ])

        print(f"[{model}] {i+1}/{runs}: {latency_ms:.2f} ms {status}", flush=True)

    if not latencies:
        return {
            "model": model,
            "warmup_ms": warmup_latency_ms,
            "warmup_model_load_ms": warmup_timing["model_load_ms"],
            "warmup_after_model_load_ms": warmup_timing["latency_after_model_load_ms"],
            "runs_ok": 0,
            "mean_ms": 0,
            "median_ms": 0,
            "p95_ms": 0,
            "p99_ms": 0,
            "max_ms": 0,
            
        }
    max_ms = max(latencies)

    return {
        "model": model,
        "warmup_ms": warmup_latency_ms,
        "warmup_model_load_ms": warmup_timing["model_load_ms"],
        "warmup_after_model_load_ms": warmup_timing["latency_after_model_load_ms"],
        "runs_ok": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "max_ms": max_ms,
       
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-inputs", default="/home/asurite.ad.asu.edu/dprahlad/deterministic-prototyping-agenticCPS/src/HardwareIntegration/llm-inference-code-log/logs-trace/llm_inputs.csv")
    parser.add_argument("--output-dir", default="/home/asurite.ad.asu.edu/dprahlad/deterministic-prototyping-agenticCPS/src/HardwareIntegration/llm-inference-code-log")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--num-predict", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    inputs = load_llm_inputs(args.llm_inputs)

    detail_path = os.path.join(args.output_dir, "ollama_inference_detail.csv")
    summary_path = os.path.join(args.output_dir, "ollama_inference_summary_warmup.csv")

    summaries = []

    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model",
            "run_id",
            "input_index",
            "status",
            "latency_ms",
            "prompt_chars",
            "response_chars",
            "model_load_ms",
            "latency_after_model_load_ms",
            "ollama_total_ms",
            "prompt_eval_ms",
            "eval_ms",
            "response",
            "error",
        ])

        for model in args.models:
            summary = benchmark_model(
                model=model,
                inputs=inputs,
                runs=args.runs,
                num_predict=args.num_predict,
                writer=writer,
            )
            summaries.append(summary)
            f.flush()

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model",
            "warmup_ms",
            "warmup_model_load_ms",
            "warmup_after_model_load_ms",
            "runs_ok",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
  
        ])
        writer.writeheader()
        writer.writerows(summaries)

    print("\n===== SUMMARY =====")
    for s in summaries:
        print(s)

    print(f"\nDetail CSV: {detail_path}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()