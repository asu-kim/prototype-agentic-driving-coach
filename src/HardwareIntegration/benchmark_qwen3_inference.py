#!/usr/bin/env python3
"""Benchmark the Ollama model and prompt used by LLMInference.lf."""

import argparse
import statistics
import time

import ollama


SYSTEM_PROMPT = """
                            
                            You are a real-time driving coach. 
                            
                            The environment has tasks in this order:

                            1. The driver must do a lane change from a left lane to right lane before the exiting the highway. The the exit is in 1.6 kilometer.
                            The driver is maintaining a speed between 85 kilometer per hour to 90 kilometer per hour. 
                            The driver must look to the right before changing the lane to avoid any cars on the right lane.

                            2. After lane change, the driver must exiting the highway, and reduce the speed to 45 kilometer per hour.
                            There maybe cars ahead of the driver and the driver must maintain a safe distance.

                            3. There is a stop sign at 1 kilometer after exiting and the driver must come to a stop sign and look right and left before moving ahead.

                            Please guide the driver based on the Inputs from the driver and the environment.

                            You will generate the output in this format:

                            "TOKEN|INSTRUCTION"

                            where TOKEN is:
                                NONE = if the driver is following the instruction
                                WARNING = If the driver is in the recoverable speed region
                                ACTUATE = if there needs to be control taken over.

                            Instruction is the guiding message for warning or actuation.
                            """


def percentile(values, percent):
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def response_field(response, name, default=0):
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def build_messages(distance, velocity, head_history, eye_history):
    user_prompt = f"""INPUT:
                                            distance = {distance:.2f}
                                            velocity = {velocity:.2f}
                                            head_history = {head_history}
                                            eye_history = {eye_history}

                                            Return output:"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def run_once(model, messages):
    start = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.1, "num_predict": 30},
    )
    wall_ms = (time.perf_counter() - start) * 1000
    server_ms = response_field(response, "total_duration") / 1_000_000
    return wall_ms, server_ms


def print_summary(label, values):
    print(f"\n{label} ({len(values)} successful runs)")
    print(f"  Average:    {statistics.fmean(values):10.2f} ms")
    print(f"  Median:     {statistics.median(values):10.2f} ms")
    print(f"  P95:        {percentile(values, 95):10.2f} ms")
    print(f"  P99:        {percentile(values, 99):10.2f} ms")
    print(f"  Worst case: {max(values):10.2f} ms")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark qwen3:30b-a3b-fp16 using the LLMInference.lf prompt."
    )
    parser.add_argument("--model", default="qwen3:30b-a3b-fp16")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--distance", type=float, default=1200.0)
    parser.add_argument("--velocity", type=float, default=24.0)
    parser.add_argument("--head-history", default="['CENTER', 'LEFT', 'RIGHT']")
    parser.add_argument("--eye-history", default="['CENTER', 'LEFT', 'RIGHT']")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs must be at least 1 and --warmup cannot be negative")

    messages = build_messages(
        args.distance,
        args.velocity,
        args.head_history,
        args.eye_history,
    )

    print(f"Model: {args.model}")
    print(f"Warm-up runs: {args.warmup}; measured runs: {args.runs}")

    for run in range(args.warmup):
        run_once(args.model, messages)
        print(f"Warm-up {run + 1}/{args.warmup} complete")

    wall_times = []
    server_times = []
    failures = 0
    for run in range(1, args.runs + 1):
        try:
            wall_ms, server_ms = run_once(args.model, messages)
            wall_times.append(wall_ms)
            server_times.append(server_ms)
        except Exception as error:
            failures += 1
            print(f"Run {run}/{args.runs} failed: {error}")

        if run % 10 == 0 or run == args.runs:
            print(f"Completed {run}/{args.runs}")

    if not wall_times:
        raise SystemExit("All measured runs failed.")

    print_summary("End-to-end wall time (matches LF measurement)", wall_times)
    print_summary("Ollama server total_duration", server_times)
    print(f"\nFailures: {failures}")


if __name__ == "__main__":
    main()
