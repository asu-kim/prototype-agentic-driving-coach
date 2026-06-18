import csv
import json
import os
import time
import statistics
import ollama

INPUT_CSV = os.getenv(
    "LLM_INPUT_LOG_PATH",
    "/home/asurite.ad.asu.edu/dprahlad/deterministic-prototyping-agenticCPS/src/HardwareIntegration/Logs/llm_inputs.csv",
)

OUTPUT_CSV = os.getenv(
    "LLM_BENCHMARK_LOG_PATH",
    "/home/asurite.ad.asu.edu/dprahlad/deterministic-prototyping-agenticCPS/src/HardwareIntegration/Logs/llm_inference_benchmark.csv",
)

MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
NUM_RUNS = int(os.getenv("NUM_RUNS", "300"))


def build_prompt(row):
    return f"""
You are an agentic driving coach.

Current driving state:
- Remaining distance: {row["environment_distance"]} m
- Velocity: {row["car_velocity"]} km/h
- Steering state: {row["car_steer"]}
- Head position history: {row["head_history_json"]}
- Eye position history: {row["eye_history_json"]}
- Environment states: {row["environment_states_json"]}

Give one short driving instruction.
"""


def load_inputs():
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["event"] == "llm_output_tick":
                rows.append(row)
    return rows


def run_inference(prompt):
    start = time.perf_counter()
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0,
        },
    )
    end = time.perf_counter()
    return (end - start) * 1000.0, response["message"]["content"]


def main():
    inputs = load_inputs()

    if not inputs:
        raise RuntimeError(f"No llm_output_tick rows found in {INPUT_CSV}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    latencies = []

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "model",
            "input_index",
            "latency_ms",
            "prompt_chars",
            "response_chars",
            "response",
        ])

        # Warmup
        print("[BENCHMARK] Warmup run...", flush=True)
        warmup_prompt = build_prompt(inputs[0])
        run_inference(warmup_prompt)

        for i in range(NUM_RUNS):
            row = inputs[i % len(inputs)]
            prompt = build_prompt(row)

            latency_ms, response = run_inference(prompt)
            latencies.append(latency_ms)

            writer.writerow([
                i,
                MODEL,
                i % len(inputs),
                latency_ms,
                len(prompt),
                len(response),
                response.replace("\n", " "),
            ])
            f.flush()

            print(f"[{i+1}/{NUM_RUNS}] latency = {latency_ms:.2f} ms", flush=True)

    latencies_sorted = sorted(latencies)

    mean_latency = statistics.mean(latencies)
    max_latency = max(latencies)
    p95 = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1]
    p99 = latencies_sorted[int(0.99 * len(latencies_sorted)) - 1]

    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"Model: {MODEL}")
    print(f"Runs: {NUM_RUNS}")
    print(f"Mean latency: {mean_latency:.2f} ms")
    print(f"P95 latency: {p95:.2f} ms")
    print(f"P99 latency: {p99:.2f} ms")
    print(f"Empirical max latency: {max_latency:.2f} ms")

    recommended_clock_ms = max_latency * 1.2
    print(f"Recommended LF clock period: {recommended_clock_ms:.2f} ms")
    print("=======================================")


if __name__ == "__main__":
    main()