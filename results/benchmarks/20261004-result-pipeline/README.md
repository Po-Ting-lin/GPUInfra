# A → CPU tasks → B measurements — 2026-10-04

RTX 3080; driver 595.84; GNU C++ 13.3.0; CUDA compiler 13.1.115;
RelWithDebInfo, architecture 86, NVTX OFF, graph NUMA node 0.

The core baseline is commit `ca63626`; the new simulation, tests and benchmark
were uncommitted when measured. See `metadata.json` for exact source SHA-256
hashes and `commands.json` for the four invocations. All commands ran serially,
after CTest completed; no other benchmark from this task ran concurrently.

Each configuration ran three trials for each policy/retention combination:
**48 trials total**. Every trial used 1,000 frames, 64 KiB per payload, a maximum
of 32 in-flight frames, three serial CPU tasks, and the default 50 ms cache
wait. There is one worker per A/CPU-chain/B stage. Each frame was checked
byte-for-byte. All trials completed without fallback. NVTX OFF and ON each
passed the full 7/7 CTest suite including the new pipeline regressions.

## Median of three trials

| Entries | CPU delay per stage (µs) | Policy | Retention | Frames/s | B hit rate | B result H2D (MiB) | p95 frame latency (µs) |
|---:|---:|---|---|---:|---:|---:|---:|
| 8 | 0 | fifo | discard | 19706 | 100.0% | 0.00 | 281 |
| 8 | 0 | fifo | keep | 18911 | 100.0% | 0.00 | 229 |
| 8 | 0 | lru | discard | 19258 | 100.0% | 0.00 | 319 |
| 8 | 0 | lru | keep | 18969 | 100.0% | 0.00 | 229 |
| 8 | 50 | fifo | discard | 1836 | 0.8% | 62.00 | 18340 |
| 8 | 50 | fifo | keep | 1834 | 0.2% | 62.38 | 18461 |
| 8 | 50 | lru | discard | 1835 | 0.8% | 62.00 | 18456 |
| 8 | 50 | lru | keep | 1880 | 0.1% | 62.44 | 18465 |
| 32 | 0 | fifo | discard | 19294 | 100.0% | 0.00 | 377 |
| 32 | 0 | fifo | keep | 19433 | 100.0% | 0.00 | 270 |
| 32 | 0 | lru | discard | 19637 | 100.0% | 0.00 | 326 |
| 32 | 0 | lru | keep | 19242 | 100.0% | 0.00 | 277 |
| 32 | 50 | fifo | discard | 1847 | 100.0% | 0.00 | 17552 |
| 32 | 50 | fifo | keep | 1859 | 100.0% | 0.00 | 18338 |
| 32 | 50 | lru | discard | 1809 | 100.0% | 0.00 | 18396 |
| 32 | 50 | lru | keep | 1843 | 0.8% | 62.00 | 18558 |

## Interpretation and limits

- With 32 entries and the CPU delay, LRU/Keep promotes terminal results and
  evicts still-needed results: median B hit rate was 0.8%. FIFO/Keep and both
  Discard combinations reached 100% in these trials. The capacity-2 unit test
  independently verifies this victim-order difference with fixed scheduling.
- With only 8 entries and 32 in-flight frames under CPU delay, even Discard
  cannot retain all pending results. CPU-atom recovery preserves correctness.
- At zero artificial delay, scheduling changes the actual number of in-flight
  frames. The CSV records observed peak concurrency; the configured maximum
  is not a guaranteed queue depth. Do not attribute all timing differences
  solely to victim selection.
- CPU-delayed throughput is roughly 1.8–2.3k frames/s across combinations.
  Reduced H2D traffic does not demonstrate a throughput gain in this CPU-limited
  synthetic workload. Three trials on one machine are not a statistical
  performance guarantee or an AOI integration measurement.
- Timings include CUDA copies/kernel/sync, CPU checksums, requested sleeps,
  queue overhead, atom copying, full output verification, pipeline fill/drain
  and final worker joins. They exclude load/allocation and teardown. Frame
  latency includes queueing after admission; the requested sleep can overshoot.
- Result H2D measures B recovery only. A uploads every input frame and saves
  every result to the CPU atom regardless of cache policy.

[Harness design, test coverage and reproduction](../../../docs/result_pipeline.md)

Artifacts: `measurements.csv` (48 trial rows), `commands.json` (argument lists),
and `metadata.json` (selected build/hardware/provenance fields). No raw
profiler databases or environment dumps are retained.
