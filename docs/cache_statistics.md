# Cache statistics and legacy A/B comparison

Each GpuDataCache owns always-enabled CacheStatistics. Counters use
std::uint64_t; times use std::chrono::nanoseconds. All updates and snapshots
use the existing manager mutex. There is no logging, allocation, or separate
atomic counter on the cache hot path. Additional timestamps are taken only
when entering an actual condition-variable wait.

## Counting boundary

A normal getCacheData() return adds exactly one final outcome:

| Field | Meaning |
| --- | --- |
| hit | Returned CacheHit, including after waiting |
| fill | Returned CacheFill; counts a reservation, not successful publication |
| fallback | Returned TaskFallback |
| invalid | Returned Invalid from the manager |
| fillSucceeded | Fill completed successfully, including an explicit Discard completion |
| fillFailed | Fill failed, including RAII abandonment |
| eviction | An old Valid entry was replaced and a new CacheFill returned |
| discard | A cache-backed entry was removed because a caller completed with Discard |
| firstBlockedLoading | First obstruction in this request was a matching Loading entry |
| firstBlockedFull | First obstruction was no empty or evictable entry in a nonzero-capacity cache |

Retries and spurious wakeups do not add outcomes or repeat first-obstruction
counts. Zero capacity is not counted as Full. A request can first encounter
Loading and later finish with FullTimeout, or vice versa. The first-obstruction
fields and final fallback reasons deliberately answer different questions.

An empty-slot fill is not an eviction. Failure of a replacement fill does not
undo its eviction count. While fills are live, fillSucceeded + fillFailed can
be smaller than fill. They reconcile after normal completion/RAII cleanup.
Hit/fallback execution failures do not become fillFailed or rewrite the
already-returned lookup outcome.

Discard is counted once when the entry actually returns to Empty, not when a
reader first marks it. Remaining readers, including RAII cleanup, defer removal
until their GPU use finishes. Discard does not count fallback completion,
failed-fill rollback, reset or release. A successful fill completed with Discard
counts both fillSucceeded and discard, and is not retained for later hits.
See [eviction and retention policies](cache_policies.md).

Counters cover manager calls only. Rejection in StaticData validation or a
task before it calls the manager does not increment manager.invalid. Control
operations such as failed reset or initialize do not count as lookup errors.

## Fallback reasons

fallbackReasons contains mutually exclusive final-return counts; their sum
equals fallback.

| Field | Meaning |
| --- | --- |
| capacityZero | No cache entries configured |
| loadingTimeout | Matching entry still Loading at the shared deadline |
| fullTimeout | No empty/evictable entry at the shared deadline |
| loadingNoWait | Matching Loading entry with timeout configured as zero |
| fullNoWait | Full cache with timeout configured as zero |
| replicaUnavailable | Valid resident entry lacks the requested usable GPU replica |

A positive timeout exhausted before the first sleep is still a timeout reason,
with waitedRequests unchanged. ReplicaUnavailable is a defensive existing path;
it is not normally reachable under the current single-GPU initialization rules.

## Waiting time

- waitedRequests increases once when a request first enters wait_until().
- totalWaitTime sums all completed wait episodes, through mutex reacquisition.
- maxWaitTime is the largest accumulated wait for one request, not the largest
  individual sleep.
- Initial API lock contention, metadata lookup, and CUDA execution/synchronization
  are excluded. First-obstruction fields may increase even without sleeping.
- Average wait = totalWaitTime / waitedRequests; use zero when the count is zero.

A snapshot during execution is safe but provisional: sleepers have not yet
added their current episode's duration, and fills may still be outstanding.
Use a quiescent snapshot or successful reset output for final A/B totals.

## Snapshot and reset API

```cpp
const CacheStatistics snapshot = cache.statisticsSnapshot();
CacheStatistics completed;
const bool reset = cache.resetCache(&completed);
```

Snapshot returns a copy without clearing. Successful reset copies the previous
interval, clears statistics and residency under the same lock, and preserves
GPU allocations. Rejected reset changes neither counters nor completed.
Calling resetCache() without an output pointer still clears counters.

StaticData forwards these through cacheStatisticsSnapshot() and
resetCache(&completed). Static GPU regions are unaffected by frame-cache reset.
Successful initialize starts a fresh interval. Release retains the last counters
for inspection; a later successful initialize clears them. Owner lifetime still
must cover all calls and leases.

## Caller report

The demo caller prints one [CacheStatistics] line per graph after stopping,
outside timed execution and before shutdown. It includes run, graph, logical
GPU, cache name, capacity, timeout, frame size, task/worker counts, execution
model, configured workload, eviction policy, raw counters (including discard)
and ratios.

The explicitly labeled interval is warmup+timed; reading it does not reset
cache residency between phases. Production callers should label their own
matching run boundaries and report unsuccessful runs separately.

- Hit ratio = hit / (hit + fill + fallback).
- Fallback ratio = fallback / (hit + fill + fallback).
- Invalid is reported separately, excluded from those denominators.
- Average wait is reported in milliseconds; total/max also retain raw nanoseconds.
- No H2D-bytes-saved estimate is inferred from hits. A fallback may transfer only
  an ROI, and GPU-result caches may avoid computation instead of H2D.

## Legacy mapping

The supplied legacy GpuCache.cpp counts first lookups in cacheStatistics[0]
and after_stall calls in cacheStatistics[1]. resetCache() prints and clears
them once reset_counter reaches init_counter. Its debug path also periodically
prints a lookup summary. The supplied source contains OCR damage in log
expressions, so compare counter updates rather than trusting those expressions.

| Legacy counters | New interpretation |
| --- | --- |
| hit across initial/retry paths | Final hit; verify retries terminate on a hit |
| sum of miss[0..5] | Fill reservations |
| miss[0] + miss[1] | Empty-slot fills, comparable to fill - eviction |
| sum of miss[2..5] | Replacement fills, comparable to eviction |
| loading_skipped | A Loading obstruction per attempt, not necessarily final fallback |
| skipped | Both mapped ways busy per attempt; new Full concerns the entire eligible cache |
| cache_skipped | Legacy modality/camera policy exclusion; no new policy equivalent |
| total_call_counter | First legacy lookup count; normalize away after_stall retry attempts |

Legacy allows its own loading owner to reuse a slot as a hit; GPUInfra currently
waits/falls back on any matching Loading entry. This policy difference can alter
hit and wait rates and must not be disguised as a counting discrepancy.
Likewise, legacy two-way mapping and GPUInfra global eviction differ. A raw
legacy skip count cannot establish a final fallback rate: capture the final
caller-selected path when instrumenting the integration.

Use identical input identities/order, actual capacities in bytes/entries,
task/worker concurrency, GPU, execution model and run boundaries. State whether
wait/stall policies are matched or are the variable being compared. Record
LRU/FIFO and Keep/Discard settings explicitly; legacy has no equivalent
last-use discard counter. Compare final caller outcomes, end-to-end
throughput/latency and separately measured
transfer bytes; hit ratio alone is not evidence of higher throughput.
