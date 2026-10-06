# Cache eviction and retention

Eviction policy and retention are independent choices. Each manager selects
LRU or FIFO during initialization. Each caller chooses Keep or Discard when
finishing an access. Both policies use the same lease, waiting and fallback
contracts; neither can overwrite Loading data or a live reader's allocation.

## Selecting LRU or FIFO

```cpp
GpuDataCache frameCache;
bool frameInitialized = frameCache.initialize({gpuId}, frameBytes, frameCapacity);

GpuDataCache resultCache;
bool resultInitialized = resultCache.initialize({gpuId}, resultBytes, resultCapacity, std::chrono::milliseconds(50), CacheEvictionPolicy::FIFO);
```

LRU remains the default, so existing initialization calls behave as before.
The policy is fixed until release/reinitialization; reset clears entries and
statistics while preserving the configured policy and GPU allocations.
`StaticDataConfig::gpuCacheEvictionPolicy` and the simulation's corresponding
`GraphConfig` field also default to LRU and forward the choice to the manager.

| Policy | Ordering | Effect of a Hit completed with Keep |
| --- | --- | --- |
| LRU | Successful fills and the last reader's release enter MRU | Moves the entry to MRU when its last reader finishes |
| FIFO | Successful fill completion, oldest first | Keeps the original fill position |

Empty entries are always chosen first. At capacity, the manager evicts the
oldest **eligible** entry under the selected policy. FIFO skips entries still
being read, but retains their position: after the readers finish, an older
entry is still older than one filled later. Loading reservations have not
joined FIFO yet. If two fills finish in reverse reservation order, their
successful completion order determines FIFO age.

LRU uses the existing intrusive evictable list. FIFO additionally maintains
an intrusive list of all Valid entries, including active readers. Hits and
reader completion do not move entries in that list. Selection may scan past
active entries, up to O(K) in the worst case; list insertion/removal is O(1).
Both lists use entry-owned links and allocate nothing on the hot path.

## Last-use completion

```cpp
bool retained = access.freeCacheData(submittedSuccessfully);
// Equivalent explicit choice:
bool retainedExplicitly = anotherAccess.freeCacheData(submittedSuccessfully, CacheRetention::Keep);

// Task B knows there are no further consumers or retries of this result.
bool completed = resultAccess.freeCacheData(submittedSuccessfully, CacheRetention::Discard);
```

Use Discard only when the caller knows that this key has no future consumers
or retries. Existing reader leases may still be finishing. Discard applies
to the entry held by this access; it is not a key-based invalidation of some
other request's entry.

| Access | Keep | Discard |
| --- | --- | --- |
| CacheHit | Release the reader; retain immutable data | Mark the entry for removal; remove it when all current readers finish |
| Successful CacheFill | Publish and retain data | Complete the fill and return the slot to Empty without retaining the key |
| Failed CacheFill | Roll back to Empty | Same rollback; failed publication is counted as fillFailed |
| TaskFallback | Complete caller-owned work | Same completion; this access has no cache entry to discard |

Both choices synchronize the access stream before manager completion. A
pending discard is sticky: later Keep releases cannot cancel it. The final
reader honors it even when that reader exits through RAII cleanup or move
assignment. Ordinary RAII cleanup adds no new discard request and continues
to abort unfinished fills.

`submittedSuccessfully` and retention answer different questions. A failed
terminal consumer may call `freeCacheData(false, CacheRetention::Discard)`;
the call reports failure while still releasing/discarding its input. CUDA
synchronization errors retain the existing diagnostics and failure contract;
an ended lease does not establish that a failing GPU successfully drained.

Actual discard removes the residency mapping and returns the entry to Empty.
It does not call cudaFree, change the capacity or free caller buffers. It
wakes requests waiting for space. After removal, a lookup with the same identity is
a normal miss and follows the existing Fill/Fallback rules; no tombstone is
kept. Reset and release continue to reject all live accesses and waiters.

## A to B result workflow

For `Task A -> CPU tasks -> Task B`, A completes its D2D cache fill with Keep
and preserves the authoritative CPU atom through D2H. B uses the handed-off
identity, performs its normal Hit/Fill/Fallback handling, and completes its
final use with Discard. B can do this with either LRU or FIFO.

This releases completed results without making them MRU or leaving them in
FIFO. Pending results remain best-effort cache data: neither policy guarantees
that an A result survives the intervening CPU tasks. B still needs its CPU
atom recovery path. Graph ordering, scheduling and NUMA ownership do not change.

## Statistics and validation

`CacheStatistics::discard` counts actual explicit removals, once per entry.
It does not count a pending marker, fallback completion, failed-fill rollback,
reset or release. A successful Fill completed with Discard counts both
fillSucceeded and discard. Explicit removals are separate from eviction.
The demo's caller report includes `eviction_policy` and `discard`. See
[counter definitions](cache_statistics.md) for A/B comparison boundaries.

Protocol tests compare LRU/FIFO victims after hits, protect busy readers,
preserve FIFO age after release, order reversed fill completions, and cover
reset/reinitialization and abandoned fills. Both policies also test deferred
Discard, move/RAII cleanup, failed consumers, Fill/Fallback completion,
unchanged GPU allocation addresses, waiting-request wakeup, and StaticData
configuration forwarding.
The fixed-size result handoff test also runs both policies, verifying A's D2D
publication and CPU atom copy, B's cross-stream hit/recovery, and last-use
Discard allowing the next producer to reuse space without eviction.

A four-entry FIFO regression also discards the middle (with outstanding
readers), tail and head, checks surviving bytes, and forces subsequent
replacements to verify that the oldest surviving keys are evicted in order.

## Lookup-cost benchmark

The manual `gpuinfra_cache_policy_benchmark` CMake target compares LRU/FIFO
with a configurable prefix of long-held reader leases and one or more request
workers. It reports getCacheData latency separately from complete-request
throughput, including the normal CUDA fill and synchronization costs.
It is excluded from default builds and is not a timing-gated CTest.

[Recorded measurements and reproduction commands](../results/benchmarks/20261003-cache-policy/README.md)
show increasing FIFO cost as more old entries are held. Large capacity alone
does not imply a long scan. In the A-to-B workflow, results waiting between
tasks are normally idle cache entries after A finishes Keep, not live readers.
Use actual concurrent lease counts and workload timings to decide whether
replacing the FIFO scan is worthwhile.

## A → CPU tasks → B benchmark

The [result pipeline simulation](result_pipeline.md) uses separate A/B streams,
a CPU task chain, and bounded in-flight frames. It compares both policies with
Keep and Discard, measuring B hit rate, recovery traffic and end-to-end timing.
Unlike the held-reader lookup benchmark, no lease spans the CPU task chain.
