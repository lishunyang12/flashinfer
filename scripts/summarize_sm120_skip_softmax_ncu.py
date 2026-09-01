#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Summarize selected metrics from Nsight Compute raw CSV exports."""

import argparse
import csv
from pathlib import Path


METRICS = {
    "duration_ns": "gpu__time_duration.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_read_bytes": "dram__bytes_op_read.sum",
    "dram_write_bytes": "dram__bytes_op_write.sum",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_bytes": "lts__t_bytes.sum",
    "tensor_inst": "smsp__inst_executed_pipe_tensor.sum",
    "fma_inst": "smsp__inst_executed_pipe_fma.sum",
    "alu_inst": "smsp__inst_executed_pipe_alu.sum",
    "lsu_inst": "smsp__inst_executed_pipe_lsu.sum",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "active_warps": "smsp__warps_active.avg.per_cycle_active",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "stall_barrier_pct": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "stall_membar_pct": "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "stall_long_scoreboard_pct": (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
    ),
    "stall_wait_pct": "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "stall_math_pipe_pct": (
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
    ),
    "stall_mio_pct": "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "stall_not_selected_pct": (
        "smsp__warp_issue_stalled_not_selected_per_warp_active.pct"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def parse_number(value: str) -> float:
    return float(value.replace(",", "").strip())


def read_raw_csv(path: Path) -> tuple[str, dict[str, float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for start, line in enumerate(lines):
        if "Kernel Name" not in line or "gpu__time_duration.sum" not in line:
            continue
        reader = csv.DictReader(lines[start:])
        for row in reader:
            kernel_name = (row.get("Kernel Name") or "").strip()
            if not kernel_name:
                continue
            values: dict[str, float] = {}
            for metric_name in METRICS.values():
                metric_value = (row.get(metric_name) or "").strip()
                if not metric_value:
                    continue
                try:
                    values[metric_name] = parse_number(metric_value)
                except ValueError:
                    continue
            return kernel_name, values
    raise RuntimeError(f"Nsight Compute CSV header not found in {path}")


def format_value(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    modes = ("dense", "skip_no_tiles", "skip_active")
    records = []
    for mode in modes:
        kernel_name, raw = read_raw_csv(args.result_root / f"{mode}.raw.csv")
        record: dict[str, object] = {"mode": mode, "kernel_name": kernel_name}
        for output_name, metric_name in METRICS.items():
            record[output_name] = raw.get(metric_name)
        duration_ns = record["duration_ns"]
        record["duration_ms"] = (
            None if duration_ns is None else float(duration_ns) / 1e6
        )
        for source, output in (
            ("dram_read_bytes", "dram_read_gib"),
            ("dram_write_bytes", "dram_write_gib"),
            ("l2_bytes", "l2_gib"),
        ):
            value = record[source]
            record[output] = None if value is None else float(value) / (1 << 30)
        records.append(record)

    dense_duration = records[0]["duration_ms"]
    dense_tensor = records[0]["tensor_inst"]
    dense_dram_read = records[0]["dram_read_bytes"]
    for record in records:
        duration = record["duration_ms"]
        tensor = record["tensor_inst"]
        dram_read = record["dram_read_bytes"]
        record["speedup_vs_dense"] = (
            None
            if dense_duration is None or duration is None
            else float(dense_duration) / float(duration)
        )
        record["tensor_inst_vs_dense"] = (
            None
            if dense_tensor is None or tensor is None
            else float(tensor) / float(dense_tensor)
        )
        record["dram_read_vs_dense"] = (
            None
            if dense_dram_read is None or dram_read is None
            else float(dram_read) / float(dense_dram_read)
        )

    csv_path = args.result_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("SM120_SKIP_SOFTMAX_NCU_SUMMARY")
    print(
        "mode duration_ms speedup sm_pct dram_pct dram_read_gib l2_gib "
        "dram_read_vs_dense tensor_vs_dense issue_active eligible_warps "
        "barrier_pct membar_pct long_scoreboard_pct wait_pct math_pipe_pct "
        "mio_pct not_selected_pct"
    )
    for record in records:
        print(
            f"{record['mode']} "
            f"{format_value(record['duration_ms'])} "
            f"{format_value(record['speedup_vs_dense'])} "
            f"{format_value(record['sm_pct'])} "
            f"{format_value(record['dram_pct'])} "
            f"{format_value(record['dram_read_gib'])} "
            f"{format_value(record['l2_gib'])} "
            f"{format_value(record['dram_read_vs_dense'])} "
            f"{format_value(record['tensor_inst_vs_dense'])} "
            f"{format_value(record['issue_active_pct'])} "
            f"{format_value(record['eligible_warps'])} "
            f"{format_value(record['stall_barrier_pct'])} "
            f"{format_value(record['stall_membar_pct'])} "
            f"{format_value(record['stall_long_scoreboard_pct'])} "
            f"{format_value(record['stall_wait_pct'])} "
            f"{format_value(record['stall_math_pipe_pct'])} "
            f"{format_value(record['stall_mio_pct'])} "
            f"{format_value(record['stall_not_selected_pct'])}"
        )
    print(f"SUMMARY_CSV={csv_path}")


if __name__ == "__main__":
    main()
