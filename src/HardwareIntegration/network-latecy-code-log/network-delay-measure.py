#!/usr/bin/env python3
import argparse
import csv
import socket
import statistics
import time


def run_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)

        print(f"[SERVER] Listening on {host}:{port}", flush=True)

        conn, addr = s.accept()
        with conn:
            print(f"[SERVER] Connected by {addr}", flush=True)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.sendall(data)


def run_client(host, port, runs, output_csv, payload_size):
    payload = b"x" * payload_size
    rtts_ms = []

    with socket.create_connection((host, port), timeout=10) as s:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "payload_bytes", "rtt_ms", "one_way_est_ms"])

            for i in range(runs):
                t0 = time.perf_counter_ns()
                s.sendall(payload)

                received = b""
                while len(received) < payload_size:
                    chunk = s.recv(payload_size - len(received))
                    if not chunk:
                        raise RuntimeError("Connection closed early")
                    received += chunk

                t1 = time.perf_counter_ns()

                rtt_ms = (t1 - t0) / 1e6
                one_way_est_ms = rtt_ms / 2.0
                rtts_ms.append(rtt_ms)

                writer.writerow([i, payload_size, rtt_ms, one_way_est_ms])
                f.flush()

                print(f"[{i+1}/{runs}] RTT={rtt_ms:.3f} ms, one-way≈{one_way_est_ms:.3f} ms", flush=True)

    rtts_sorted = sorted(rtts_ms)

    def pct(p):
        idx = int((p / 100) * len(rtts_sorted)) - 1
        idx = max(0, min(idx, len(rtts_sorted) - 1))
        return rtts_sorted[idx]

    print("\n===== NETWORK SUMMARY =====")
    print(f"Runs: {runs}")
    print(f"Payload: {payload_size} bytes")
    print(f"Mean RTT: {statistics.mean(rtts_ms):.3f} ms")
    print(f"Median RTT: {statistics.median(rtts_ms):.3f} ms")
    print(f"P95 RTT: {pct(95):.3f} ms")
    print(f"P99 RTT: {pct(99):.3f} ms")
    print(f"Max RTT: {max(rtts_ms):.3f} ms")
    print(f"Estimated one-way max: {max(rtts_ms)/2:.3f} ms")
    print("===========================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--payload-size", type=int, default=256)
    parser.add_argument("--output-csv", default="network_delay.csv")
    args = parser.parse_args()

    if args.mode == "server":
        run_server(args.host, args.port)
    else:
        run_client(args.host, args.port, args.runs, args.output_csv, args.payload_size)


if __name__ == "__main__":
    main()