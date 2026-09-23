# Optional NVTX profiling

Build annotations explicitly; the default is OFF:

```sh
cmake -S . -B cmake-build-nvtx-on -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGPUINFRA_ENABLE_NVTX=ON
cmake --build cmake-build-nvtx-on --parallel 4
ctest --test-dir cmake-build-nvtx-on --output-on-failure
```

The enabled build requires nvtx3/nvtx3.hpp, normally supplied by CUDA.
CMake exports GPUINFRA_ENABLE_NVTX=1 to all linked targets. The OFF build
does not search for NVTX headers or define the macro. Nvtx.h then excludes
NVTX entirely, and macro arguments are not evaluated.

Use GPUINFRA_NVTX_SCOPE("Fixed.name") inside a braced scope. Enabled scopes use
[NVIDIA's scoped_range RAII API](https://nvidia.github.io/NVTX/doxygen-cpp/index.html):
returns, breaks and exceptions end the range automatically. Avoid dynamic
strings and arguments with side effects. Scope names are fixed, and no extra
CUDA synchronization or cache/scheduler behavior was introduced.

## Ranges

| Range | CPU work covered |
| --- | --- |
| Phase.warmup / Phase.timed | Main-thread phase setup and waiting for completion |
| Task.execute | One task callback |
| Cache.getCacheData | Complete lookup including lock acquisition and bounded waits |
| Cache.wait | An actual condition-variable wait through mutex reacquisition |
| Frame.H2D.submit_with_staging | Pageable-to-pinned staging and H2D submission |
| Cel/Sdd/Mi.kernels.submit | Each algorithm's kernel submission function |
| Cel/Sdd/Mi.D2H.submit | Each algorithm's D2H submission function |
| Cache.freeCacheData | Synchronization and completion |
| Cache.stream_sync | Existing CUDA stream synchronization |
| Cache.publish_release | Metadata publication/rollback and lease release |
| Cache.RAII_abort_sync | Destructor synchronization and abort |

CPU ranges are not GPU durations. Use CUDA activities for actual transfer and
kernel times. Nested ranges and parallel workers overlap; do not sum their
percentages as a wall-clock decomposition. Main-thread phase ranges do not
lexically enclose worker ranges on other threads.

## Reproduce collection

Use a new output directory for each capture; the script refuses to overwrite it.
The collector passes an explicit environment allowlist to Nsight and its child;
ambient access tokens are not passed through that child environment. Nsight can
still capture environment information from other processes: the allowlist does
not sanitize a report. New captures use owner-only permissions. Raw artifacts
remain local-only. Enable the repository commit/push hooks as described in
[recorded results](../results/README.md).

```sh
python3 tools/collect_nvtx.py --output results/profiling/my-capture
python3 tools/summarize_nvtx.py results/profiling/my-capture
```

The first script runs 20 warmup + 200 timed frames per GPU, size factor 64,
for both batched and interleaved. It uses --trace=cuda,nvtx --sample=none
--cpuctxsw=none. Override workload through its command-line options.
It saves .nsys-rep, SQLite exports, CSV reports, application/profiler logs,
exact command arguments, and GPU/profiler identification. Ensure the supplied
binary was built with NVTX enabled; the summary rejects missing phase markers.

The summary script was verified with Nsight Systems 2025.5 SQLite schema.
It separates events by full containment within phase timestamps, omitting
initialization events. Each phase drains submitted work before returning.
The generic nsys CSV summaries cover the whole process, including initialization.
CacheStatistics log intervals cover warmup+timed, independently of phase summaries.

## Recorded capture

[2026-09-20 summary](../results/profiling/20260920-nvtx/SUMMARY.md).
Raw batched/interleaved traces and SQLite exports are local-only because they
can contain credentials from the captured environment.

- RTX 3080; driver 595.84; Nsight Systems 2025.5.2.
- RelWithDebInfo, CUDA architecture 86, NVTX ON.
- Multipliers: H2D 1, D2H 1, compute 3; four tasks/workers, four cache entries.
- Timed phase CPU ranges: batched 19.116 ms; interleaved 20.389 ms.
- Each timed phase: 200 executions, 600 kernels, 50 MiB H2D, 68.75 MiB D2H.
- Across both phases: each mode had 220 fills, 216 evictions and no hits,
  fallbacks or waits. This workload cannot evaluate the bounded-wait policy.

These are single profiler-affected observations, not evidence that one mode
has higher production throughput. Repeat representative workloads without
the profiler to measure throughput. The captures establish instrumentation
and let GPU operation timing be inspected independently of CPU submission time.

## Validation

Both ON and OFF RelWithDebInfo builds passed all seven CTests. A standalone
OFF compilation passed with an undeclared function as the scope argument,
confirming arguments are discarded. The OFF demo contained no NVTX symbols.
The collected traces contain both phase markers and the expected 220 task,
H2D, each-algorithm submission and completion ranges.
