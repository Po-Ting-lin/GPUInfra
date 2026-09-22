#!/usr/bin/env python3
"""Phase-separated summary for Nsight Systems SQLite exports (2025.5 schema)."""
import argparse
import pathlib
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=pathlib.Path)
args = parser.parse_args()
lines = [
    "# NVTX capture summary",
    "",
    "One capture per mode; these are profiler-affected observations, not a throughput benchmark.",
    "CPU phase ranges are on the main thread. Worker/GPU events are selected by",
    "full timestamp containment within each phase, not thread nesting.",
    "GPU durations are sums across operations and may overlap; do not add them to CPU times.",
    "Unfiltered nsys CSV summaries include initialization and both phases.",
    "",
    "| Mode | Phase | CPU phase ms | Executes | Kernels | GPU kernel sum ms | H2D count / MiB / ms | D2H count / MiB / ms |",
    "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
]
details = []
cache_lines = []
for mode in ("batched", "interleaved"):
    cache_lines.extend(line for line in (args.directory / (mode + ".log")).read_text().splitlines() if line.startswith("[CacheStatistics]"))
    with sqlite3.connect(args.directory / (mode + ".sqlite")) as database:
        phases = database.execute("SELECT text,start,end FROM NVTX_EVENTS WHERE text IN ('Phase.warmup','Phase.timed') ORDER BY start").fetchall()
        if len(phases) != 2:
            raise RuntimeError("Expected both phase markers; ensure NVTX was enabled")
        for name, start, end in phases:
            window = (start, end)
            tasks = database.execute("SELECT count(*) FROM NVTX_EVENTS WHERE text='Task.execute' AND start>=? AND end<=?", window).fetchone()[0]
            kernels, kernel_time = database.execute("SELECT count(*),coalesce(sum(end-start),0) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", window).fetchone()
            copies = []
            for kind in ("CUDA_MEMCPY_KIND_HTOD", "CUDA_MEMCPY_KIND_DTOH"):
                count, size, duration = database.execute("SELECT count(*),coalesce(sum(bytes),0),coalesce(sum(m.end-m.start),0) FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER k ON m.copyKind=k.id WHERE k.name=? AND m.start>=? AND m.end<=?", (kind, start, end)).fetchone()
                copies.append(f"{count} / {size / 1048576:.3f} / {duration / 1e6:.3f}")
            lines.append(f"| {mode} | {name} | {(end-start)/1e6:.3f} | {tasks} | {kernels} | {kernel_time/1e6:.3f} | {copies[0]} | {copies[1]} |")
            details.extend(["", f"## {mode}: {name}", "", "| CPU range | Count | Total ms | Average us |", "| --- | ---: | ---: | ---: |"])
            ranges = database.execute("SELECT text,count(*),sum(end-start) FROM NVTX_EVENTS WHERE start>=? AND end<=? AND text NOT LIKE 'Phase.%' GROUP BY text ORDER BY text", window).fetchall()
            for label, count, duration in ranges:
                details.append(f"| {label} | {count} | {duration/1e6:.3f} | {duration/count/1000:.3f} |")
lines += ["", "Cache counters below cover warmup+timed; absent Cache.wait ranges mean no actual waits were observed.",
          "H2D staging ranges include the CPU memcpy. Submission ranges do not measure GPU execution.",
          "Nested CPU ranges overlap and their totals must not be added together.",
          "", "```text"] + cache_lines + ["```"] + details
(args.directory / "SUMMARY.md").write_text("\n".join(lines) + "\n")
