#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_ORIGINAL = Path("src/HardwareIntegration/data-collection/logs-trace-2")
DEFAULT_REPLAY = Path("src/HardwareIntegration/data-collection/logs-trace-2-replay")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tardy_value(row: dict[str, str]) -> str:
    for field in ("event", "reaction", "Event", "Trigger"):
        value = row.get(field, "")
        if "tardy" in str(value).lower():
            return str(value)
    return ""


def row_time(row: dict[str, str]) -> tuple[str, str, str]:
    logical_ns = row.get("logical_time_ns") or row.get("Elapsed Logical Time") or ""
    logical_ms = row.get("logical_time_ms") or ""
    if not logical_ms and logical_ns:
        try:
            logical_ms = str(float(logical_ns) / 1e6)
        except Exception:
            logical_ms = ""
    microstep = row.get("microstep") or row.get("Microstep") or ""
    return logical_ns, logical_ms, microstep


def collect_tardy_events(trace_dir: Path, trace_label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for csv_path in sorted(trace_dir.glob("*.csv")):
        for row_index, row in enumerate(read_rows(csv_path), start=2):
            tardy = tardy_value(row)
            if not tardy:
                continue
            logical_ns, logical_ms, microstep = row_time(row)
            rows.append({
                "trace": trace_label,
                "file": csv_path.name,
                "row_number": row_index,
                "tardy_kind": tardy,
                "federate": row.get("federate", ""),
                "reactor": row.get("reactor", row.get("Reactor", "")),
                "reaction": row.get("reaction", ""),
                "logical_time_ns": logical_ns,
                "logical_time_ms": logical_ms,
                "microstep": microstep,
                "physical_time_ms": row.get("physical_time_ms", ""),
                "execution_time_ms": row.get("execution_time_ms", ""),
                "control": row.get("control", row.get("control_token", "")),
                "instruction": row.get("instruction", ""),
                "act_command": row.get("act_command", row.get("actuate_in", "")),
                "reason": row.get("reason", ""),
            })
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter(
        (str(r["trace"]), str(r["file"]), str(r["federate"]), str(r["reactor"]), str(r["reaction"]), str(r["tardy_kind"]))
        for r in rows
    )
    summary = []
    for key, count in sorted(counts.items()):
        trace, file_name, federate, reactor, reaction, tardy_kind = key
        summary.append({
            "trace": trace,
            "file": file_name,
            "federate": federate,
            "reactor": reactor,
            "reaction": reaction,
            "tardy_kind": tardy_kind,
            "count": count,
        })
    return summary


def compare_summary(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str, str, str, str], dict[str, int]] = {}
    for row in summary:
        key = (
            str(row["file"]),
            str(row["federate"]),
            str(row["reactor"]),
            str(row["reaction"]),
            str(row["tardy_kind"]),
        )
        by_key.setdefault(key, {"Collected": 0, "Replay": 0})[str(row["trace"])] = int(row["count"])

    compared = []
    for key, counts in sorted(by_key.items()):
        original = counts.get("Collected", 0)
        replay = counts.get("Replay", 0)
        compared.append({
            "file": key[0],
            "federate": key[1],
            "reactor": key[2],
            "reaction": key[3],
            "tardy_kind": key[4],
            "collected_count": original,
            "replay_count": replay,
            "delta_replay_minus_collected": replay - original,
            "matches": int(original == replay),
        })
    return compared


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tardy events across collected and replay trace CSVs.")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (args.replay / "tardy-summary")
    detail = collect_tardy_events(args.original, "Collected") + collect_tardy_events(args.replay, "Replay")
    summary = summarize(detail)
    comparison = compare_summary(summary)

    detail_fields = [
        "trace", "file", "row_number", "tardy_kind", "federate", "reactor", "reaction",
        "logical_time_ns", "logical_time_ms", "microstep", "physical_time_ms", "execution_time_ms",
        "control", "instruction", "act_command", "reason",
    ]
    summary_fields = ["trace", "file", "federate", "reactor", "reaction", "tardy_kind", "count"]
    comparison_fields = [
        "file", "federate", "reactor", "reaction", "tardy_kind",
        "collected_count", "replay_count", "delta_replay_minus_collected", "matches",
    ]

    write_csv(out_dir / "tardy_events_detail.csv", detail_fields, detail)
    write_csv(out_dir / "tardy_events_summary.csv", summary_fields, summary)
    write_csv(out_dir / "tardy_events_comparison.csv", comparison_fields, comparison)

    print(f"Wrote {out_dir / 'tardy_events_detail.csv'}")
    print(f"Wrote {out_dir / 'tardy_events_summary.csv'}")
    print(f"Wrote {out_dir / 'tardy_events_comparison.csv'}")
    total_collected = sum(1 for row in detail if row["trace"] == "Collected")
    total_replay = sum(1 for row in detail if row["trace"] == "Replay")
    print(f"Collected tardy rows: {total_collected}")
    print(f"Replay tardy rows: {total_replay}")


if __name__ == "__main__":
    main()
