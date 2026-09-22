# NVTX capture summary

One capture per mode; these are profiler-affected observations, not a throughput benchmark.
CPU phase ranges are on the main thread. Worker/GPU events are selected by
full timestamp containment within each phase, not thread nesting.
GPU durations are sums across operations and may overlap; do not add them to CPU times.
Unfiltered nsys CSV summaries include initialization and both phases.

| Mode | Phase | CPU phase ms | Executes | Kernels | GPU kernel sum ms | H2D count / MiB / ms | D2H count / MiB / ms |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| batched | Phase.warmup | 3.163 | 20 | 60 | 1.771 | 20 / 5.000 / 1.265 | 60 / 6.875 / 1.016 |
| batched | Phase.timed | 19.116 | 200 | 600 | 18.361 | 200 / 50.000 / 12.540 | 600 / 68.750 / 10.512 |
| interleaved | Phase.warmup | 3.130 | 20 | 60 | 2.077 | 20 / 5.000 / 1.060 | 60 / 6.875 / 0.811 |
| interleaved | Phase.timed | 20.389 | 200 | 600 | 21.871 | 200 / 50.000 / 10.535 | 600 / 68.750 / 8.701 |

Cache counters below cover warmup+timed; absent Cache.wait ranges mean no actual waits were observed.
H2D staging ranges include the CPU memcpy. Submission ranges do not measure GPU execution.
Nested CPU ranges overlap and their totals must not be added together.

```text
[CacheStatistics] run=1789871789375994 graph=0 gpu=0 cache=frame interval=warmup+timed entries=4 wait_timeout_ms=50 warmup_frames_per_gpu=20 timed_frames_per_gpu=200 frame_bytes=262144 tasks=4 workers=4 execution_model=batched hit=0 fill=220 fallback=0 invalid=0 fill_succeeded=220 fill_failed=0 eviction=216 first_blocked_loading=0 first_blocked_full=0 capacity_zero=0 loading_timeout=0 full_timeout=0 loading_no_wait=0 full_no_wait=0 replica_unavailable=0 waited_requests=0 total_wait_ns=0 max_wait_ns=0 average_wait_ms=0 hit_ratio=0 fallback_ratio=0
[CacheStatistics] run=1789871791844925 graph=0 gpu=0 cache=frame interval=warmup+timed entries=4 wait_timeout_ms=50 warmup_frames_per_gpu=20 timed_frames_per_gpu=200 frame_bytes=262144 tasks=4 workers=4 execution_model=interleaved hit=0 fill=220 fallback=0 invalid=0 fill_succeeded=220 fill_failed=0 eviction=216 first_blocked_loading=0 first_blocked_full=0 capacity_zero=0 loading_timeout=0 full_timeout=0 loading_no_wait=0 full_no_wait=0 replica_unavailable=0 waited_requests=0 total_wait_ns=0 max_wait_ns=0 average_wait_ms=0 hit_ratio=0 fallback_ratio=0
```

## batched: Phase.warmup

| CPU range | Count | Total ms | Average us |
| --- | ---: | ---: | ---: |
| Cache.freeCacheData | 20 | 4.531 | 226.525 |
| Cache.getCacheData | 20 | 0.014 | 0.689 |
| Cache.publish_release | 20 | 0.008 | 0.383 |
| Cache.stream_sync | 20 | 4.513 | 225.659 |
| Cel.D2H.submit | 20 | 0.093 | 4.628 |
| Cel.kernels.submit | 20 | 0.686 | 34.282 |
| Frame.H2D.submit_with_staging | 20 | 2.104 | 105.183 |
| Mi.D2H.submit | 20 | 0.053 | 2.639 |
| Mi.kernels.submit | 20 | 0.640 | 31.992 |
| Sdd.D2H.submit | 20 | 0.052 | 2.582 |
| Sdd.kernels.submit | 20 | 0.549 | 27.439 |
| Task.execute | 20 | 11.945 | 597.264 |

## batched: Phase.timed

| CPU range | Count | Total ms | Average us |
| --- | ---: | ---: | ---: |
| Cache.freeCacheData | 200 | 52.498 | 262.492 |
| Cache.getCacheData | 200 | 0.071 | 0.356 |
| Cache.publish_release | 200 | 0.063 | 0.316 |
| Cache.stream_sync | 200 | 52.329 | 261.643 |
| Cel.D2H.submit | 200 | 0.575 | 2.873 |
| Cel.kernels.submit | 200 | 0.799 | 3.996 |
| Frame.H2D.submit_with_staging | 200 | 8.275 | 41.376 |
| Mi.D2H.submit | 200 | 0.442 | 2.212 |
| Mi.kernels.submit | 200 | 0.616 | 3.082 |
| Sdd.D2H.submit | 200 | 0.458 | 2.288 |
| Sdd.kernels.submit | 200 | 0.624 | 3.122 |
| Task.execute | 200 | 75.862 | 379.312 |

## interleaved: Phase.warmup

| CPU range | Count | Total ms | Average us |
| --- | ---: | ---: | ---: |
| Cache.freeCacheData | 20 | 4.735 | 236.757 |
| Cache.getCacheData | 20 | 0.013 | 0.673 |
| Cache.publish_release | 20 | 0.007 | 0.358 |
| Cache.stream_sync | 20 | 4.719 | 235.940 |
| Cel.D2H.submit | 20 | 0.147 | 7.346 |
| Cel.kernels.submit | 20 | 0.506 | 25.307 |
| Frame.H2D.submit_with_staging | 20 | 1.189 | 59.465 |
| Mi.D2H.submit | 20 | 0.064 | 3.200 |
| Mi.kernels.submit | 20 | 0.858 | 42.876 |
| Sdd.D2H.submit | 20 | 0.173 | 8.658 |
| Sdd.kernels.submit | 20 | 0.829 | 41.455 |
| Task.execute | 20 | 11.540 | 577.005 |

## interleaved: Phase.timed

| CPU range | Count | Total ms | Average us |
| --- | ---: | ---: | ---: |
| Cache.freeCacheData | 200 | 56.814 | 284.069 |
| Cache.getCacheData | 200 | 0.094 | 0.469 |
| Cache.publish_release | 200 | 0.067 | 0.337 |
| Cache.stream_sync | 200 | 56.644 | 283.218 |
| Cel.D2H.submit | 200 | 0.600 | 2.999 |
| Cel.kernels.submit | 200 | 0.779 | 3.897 |
| Frame.H2D.submit_with_staging | 200 | 8.878 | 44.390 |
| Mi.D2H.submit | 200 | 0.488 | 2.438 |
| Mi.kernels.submit | 200 | 0.607 | 3.037 |
| Sdd.D2H.submit | 200 | 0.488 | 2.438 |
| Sdd.kernels.submit | 200 | 0.625 | 3.124 |
| Task.execute | 200 | 80.548 | 402.742 |
