#!/usr/bin/env python3
"""Measure TCP round-trip latency between two federate machines."""

import argparse
import csv
import socket
import statistics
import struct
import time

PACKET = struct.Struct("!Q")


def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data.extend(chunk)
    return bytes(data)


def receiver(host, port):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Listening on {host}:{port}", flush=True)
        connection, address = server.accept()
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Connected by {address}", flush=True)
            try:
                while True:
                    connection.sendall(recv_exact(connection, PACKET.size))
            except ConnectionError:
                pass


def percentile(values, percent):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sender(host, port, runs, warmup, interval_ms, output):
    samples = []
    with socket.create_connection((host, port)) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with open(output, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["sample", "round_trip_ms", "estimated_one_way_ms"])
            for sequence in range(warmup + runs):
                start = time.perf_counter_ns()
                connection.sendall(PACKET.pack(sequence))
                echoed = PACKET.unpack(recv_exact(connection, PACKET.size))[0]
                round_trip_ms = (time.perf_counter_ns() - start) / 1_000_000
                if echoed != sequence:
                    raise RuntimeError("mismatched echo")
                if sequence >= warmup:
                    samples.append(round_trip_ms)
                    writer.writerow([
                        sequence - warmup + 1,
                        f"{round_trip_ms:.6f}",
                        f"{round_trip_ms / 2:.6f}",
                    ])
                if interval_ms:
                    time.sleep(interval_ms / 1000)

    print(f"Samples: {len(samples)}")
    print(f"Average RTT: {statistics.fmean(samples):.3f} ms")
    print(f"Median RTT:  {statistics.median(samples):.3f} ms")
    print(f"P95 RTT:     {percentile(samples, 95):.3f} ms")
    print(f"P99 RTT:     {percentile(samples, 99):.3f} ms")
    print(f"Worst RTT:   {max(samples):.3f} ms")
    print(f"Estimated worst one-way: {max(samples) / 2:.3f} ms")
    print(f"CSV: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    receive_parser = modes.add_parser("receiver")
    receive_parser.add_argument("--host", default="0.0.0.0")
    receive_parser.add_argument("--port", type=int, default=9900)
    send_parser = modes.add_parser("sender")
    send_parser.add_argument("--host", required=True)
    send_parser.add_argument("--port", type=int, default=9900)
    send_parser.add_argument("--runs", type=int, default=1000)
    send_parser.add_argument("--warmup", type=int, default=20)
    send_parser.add_argument("--interval-ms", type=float, default=10)
    send_parser.add_argument("--output", default="network_communication_time.csv")
    args = parser.parse_args()
    if args.mode == "receiver":
        receiver(args.host, args.port)
    else:
        sender(args.host, args.port, args.runs, args.warmup, args.interval_ms, args.output)


if __name__ == "__main__":
    main()
