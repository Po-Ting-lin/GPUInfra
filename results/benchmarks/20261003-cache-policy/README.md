# LRU/FIFO lookup cost with retained readers — 2026-10-03

The current FIFO scan has a measurable cost when many of the oldest entries
are held by readers. With no held readers, single-worker LRU/FIFO lookup
means are close. With 4,096 entries and 4,095 held readers, FIFO lookup
averages 13.237 us versus LRU 0.121 us; complete-request throughput is
59,469 versus 271,986 requests/s. These are medians across three trials.

At the same capacity with only 64 held readers and one worker, FIFO lookup
averages 0.349 us versus LRU 0.131 us, with throughput 254,072 versus
268,781 requests/s. With 256 held readers, FIFO lookup averages 1.081 us.
Four competing workers amplify contention in the held-reader cases.

This synthetic benchmark measures cache overhead, not AOI algorithm or graph
throughput. It does not measure hit-rate benefits or compare complete graph
scheduling strategies. No core eviction implementation was changed.

## Environment

- Intel Core i5-12400; NVIDIA GeForce RTX 3080; driver 595.84.
- NUMA node 0; all workers use the framework NUMA affinity helper. Individual
  CPU cores are not pinned, so operating-system scheduling remains a variable.
- GNU C++ 13.3.0, RelWithDebInfo, CUDA Runtime 13.1, driver API 13.2, NVTX OFF.
- Core under measurement: `e0a9d4d`; this change adds the benchmark and tests.
- Machine-readable setup: [metadata.json](metadata.json).

## Method

1. Allocate and fill K persistent 64-byte entries before timing.
2. Acquire reader leases for the oldest H entries and retain them throughout
   the measurement. These readers hold no pending GPU work; their lifetime
   reproduces the ineligible prefix that FIFO must skip.
3. Each worker owns a CUDA stream and fallback buffer. Warm each worker with
   100 replacement fills and wait at a common start barrier.
4. Each worker performs 5,000 requests for unique keys. Every request must
   return CacheFill, write 64 bytes with cudaMemsetAsync, and finish through
   `freeCacheData(true, CacheRetention::Keep)`, including its normal stream synchronization.
5. Keep at least one eligible slot per worker: `K - H >= workers`. All timed
   requests must be successful evictions, with zero fallback and invalid.
6. Run each policy three times, alternating which policy goes first. Use a
   fresh manager for every trial. Initialization, warmup, held-reader release,
   GPU allocation/free and percentile calculation are outside timed execution.

`lookup_mean_us` times the entire getCacheData call, including mutex wait,
hash lookup, victim selection and metadata changes. It is not an isolated
instruction-level traversal timer. Complete-request throughput includes the
64-byte memset, CUDA synchronization, completion, timing overhead and barrier
wake/thread-join overhead. It excludes cleanup. `wall_us_per_request` is the
reciprocal aggregate throughput, not per-thread latency.

Each table cell is the median of three per-trial statistics. Per-trial
P50/P95/P99 lookup statistics are in [measurements.csv](measurements.csv).
There are 16 configurations and 96 trial rows. Every row passed the expected
fill/success/eviction counts with zero fallback and invalid requests.

## Measurements

| Entries | Held readers | Workers | LRU lookup mean us | FIFO lookup mean us | LRU requests/s | FIFO requests/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0 | 1 | 0.123 | 0.127 | 274,345 | 273,007 |
| 64 | 63 | 1 | 0.108 | 0.308 | 273,441 | 260,034 |
| 64 | 0 | 4 | 0.217 | 0.488 | 1,029,865 | 837,908 |
| 64 | 60 | 4 | 0.191 | 0.787 | 1,057,800 | 802,332 |
| 1,024 | 0 | 1 | 0.130 | 0.133 | 271,854 | 269,349 |
| 1,024 | 1,023 | 1 | 0.122 | 4.205 | 276,039 | 128,635 |
| 1,024 | 0 | 4 | 0.553 | 1.408 | 736,588 | 495,927 |
| 1,024 | 1,020 | 4 | 1.325 | 4.804 | 505,826 | 152,397 |
| 4,096 | 0 | 1 | 0.130 | 0.133 | 269,090 | 269,108 |
| 4,096 | 64 | 1 | 0.131 | 0.349 | 268,781 | 254,072 |
| 4,096 | 256 | 1 | 0.130 | 1.081 | 269,431 | 210,582 |
| 4,096 | 4,095 | 1 | 0.121 | 13.237 | 271,986 | 59,469 |
| 4,096 | 0 | 4 | 0.378 | 0.287 | 846,404 | 906,182 |
| 4,096 | 64 | 4 | 0.521 | 2.404 | 747,104 | 436,540 |
| 4,096 | 256 | 4 | 0.328 | 3.272 | 881,754 | 400,302 |
| 4,096 | 4,092 | 4 | 0.203 | 102.648 | 1,015,683 | 18,619 |

## Interpretation and next decision

- The single-worker comparison shows that held-prefix length, rather than
  capacity alone, increases FIFO lookup cost. Four-worker results also
  include mutex contention, scheduling and CUDA driver contention.
- Three repetitions on one machine establish a cost to investigate, not a
  production throughput guarantee. Small differences in the no-held-reader
  cases should not be interpreted as a general policy ranking.
- A result waiting between Task A and Task B is normally cached with no live
  lease: A has already completed Keep. Time spent in intermediate CPU tasks
  alone does not create this held-reader prefix.
- If production FIFO workloads retain hundreds or thousands of old reader
  leases while requests churn, avoiding the scan is worth evaluating. An
  eligible-entry heap ordered by fill age is one candidate, with O(log K)
  reader-release/victim updates and no scan over held entries; it also adds
  bookkeeping and needs its own correctness and throughput comparison.
- Keep the current implementation for now. Before changing it, capture actual
  concurrent lease counts and cache time in the target workload. Discard
  continues to reclaim terminal results under both policies.

## Reproduce

```sh
cmake -S . -B cmake-build-nvtx-off -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGPUINFRA_ENABLE_NVTX=OFF -DGPUINFRA_BUILD_DEMO=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cmake-build-nvtx-off --parallel 4 --target gpuinfra_cache_policy_benchmark
./cmake-build-nvtx-off/gpuinfra_cache_policy_benchmark 4096 4095 1 5000 3 > policy.csv
```

Arguments are `entries held_readers workers iterations_per_worker [repeats]`.
The binary emits CSV on stdout and environment/error messages on stderr.
Use the CUDA architecture appropriate to the target GPU. The manual target
is excluded from default builds and is not a timing-gated CTest.
All measured invocations are in [commands.json](commands.json); adapt its
absolute binary path when using another checkout.

Source: [cache_policy_benchmark.cpp](../../../tools/cache_policy_benchmark.cpp).
