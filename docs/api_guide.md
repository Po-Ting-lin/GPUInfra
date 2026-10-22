# Cache and static-data API guide

How a caller uses `StaticData`, `GpuDataCache` and `GpuDataAccess`: lookup
statuses, miss handling, saved accesses, independent caches, static GPU data,
statistics, bounded waiting and run boundaries. [Architecture](architecture.md)
describes ownership, lifecycle and the scheduler boundary these APIs sit in.

## Logical frame versus GPU cache

```text
StaticData
  ├─ fixed frame layout + resetCache() run boundary
  └─ GpuDataCache
       ├─ fixed open-addressing GpuDataKey -> entry index table [~2K]
       ├─ O(1) empty stack + intrusive inactive-entry LRU
       └─ GpuDataCacheEntry[K]
            cached metadata + lease/LRU links
            └─ GpuReplica[gpuId] -> persistent d_data + validity
```

`StaticData::getCacheData()` accepts any incoming `GpuDataKey` whose metadata
matches the fixed frame layout. `GpuDataKey` contains `frameId`, `cameraId`, and `variantId`,
so the same frame ID from different cameras cannot alias. `StaticData` stores
no per-frame registry,
scheduler state, `FramePhase`, CPU bytes, `JobResult`, or NUMA identity: graph
collections own execution state and the owning `FrameCpuAtom` carries its data
and result.
`GpuDataCacheEntry` is only a reusable best-effort GPU cache entry. It may represent
different logical frames over time while retaining the same device allocation.

### Keeping an access across caller methods

`GpuDataAccess` is movable and non-copyable. A caller can preallocate member
slots such as `GpuDataAccess masterAccess;` and
`std::array<GpuDataAccess, 64> referenceAccesses;` during object construction.
No GPU data is copied or allocated by a move.

```cpp
// loadMaster(): save the lease into an initially Invalid member.
masterAccess = frameCache.getCacheData(metadata, request);
// Handle Invalid / CacheHit / CacheFill / TaskFallback, including caller fill.

// runRef(): use masterAccess.data() on masterAccess.getStream().
// Keep the member alive while any queued work may read the pointer.

// getDefects(): after submitting all consumers, finish and check the result.
bool succeeded = masterAccess.freeCacheData(submittedSuccessfully);
```

Moving invalidates the source and transfers its existing reservation. It does
not publish a fill or change the number of active leases. Destruction of a
moved-from object does nothing. The manager, stream and fallback storage must
outlive the destination, and moving is not a thread-synchronization mechanism.

Move assignment to an already active member synchronizes and aborts the old
lease using the same safety path as destruction, then takes the new lease.
An unfinished old fill is rolled back, not published; cleanup errors are logged.
Normal code should explicitly call `freeCacheData()` and check its result before
reusing a member. In `member = cache.getCacheData(...)`, lookup happens before
assignment cleans up the old member, so explicit completion also avoids waiting
on one's own old lease. Self-move leaves the access unchanged.

## Cache access

Distinct preprocessing results use distinct caller-defined `variantId` values
(default 0). See the [payload identity contract](integration/payload_identity.md)
for shared AOI mapping, reset boundaries and remaining adapter work.

`GpuDataCache::getCacheData()` uses its fixed residency table and returns an RAII
`GpuDataAccess`. `status()` returns `CacheStatus`; `getStream()` returns the
borrowed task stream supplied in `GpuCacheRequest`:

| Status | Path |
| --- | --- |
| `CacheHit` | Matching `Valid` entry; use its immutable device pointer without H2D |
| `CacheFill` | Reserve an empty entry or an inactive LRU/FIFO candidate; caller uploads or computes its payload |
| `TaskFallback` | Caller fills its supplied fallback buffer when capacity is zero, or a Loading/full-cache wait expires |
| `Invalid` | Metadata, GPU, stream, or fallback request is invalid; do not submit work |

Lookup and resident-key insertion/erasure are average `O(1)`. The table is
allocated at about twice `K`, uses linear probing and backward-shift deletion,
and never grows or rehashes. An empty-entry stack and intrusive LRU choose a
victim in `O(1)` under LRU; FIFO may scan past live readers, up to `O(K)`
(see [architecture](architecture.md#5-logical-lookup-and-cache-lookup)). Active entries cannot be evicted. A fill becomes `Valid` only
after the task stream synchronizes successfully. Failed or abandoned fills
return to `Empty`; a failed hit keeps the immutable cached payload.

## Caller-owned miss handling and independent caches

`GpuCacheRequest` explicitly supplies `gpuId`, `stream`, `d_fallback`, and
`fallbackBytes`. The fallback pointer and stream must be valid, and fallback
capacity must be at least the cache payload size, even on an expected hit.
The cache borrows these resources; it never allocates or frees caller fallback
buffers and does not perform H2D or run algorithms.

```cpp
GpuCacheRequest request;
request.gpuId = resources.gpuId;
request.stream = resources.stream;
request.d_fallback = resources.d_input;
request.fallbackBytes = resources.inBytes;
GpuDataAccess access = staticData.getCacheData(metadata, request);
const CacheStatus status = access.status();
if (status == CacheStatus::Invalid) return false;
const cudaStream_t stream = access.getStream();
bool submittedSuccessfully = true;
if (status == CacheStatus::CacheFill || status == CacheStatus::TaskFallback) {
    submittedSuccessfully = enqueueH2D(access.writableData(), stream);
}
if (submittedSuccessfully) {
    submittedSuccessfully = enqueueComputeAndD2H(access.data(), stream);
}
// Finish once after all work on stream, even if submission fails.
bool succeeded = access.freeCacheData(submittedSuccessfully);
```

Independent `GpuDataCache` instances may cache different fixed payload
sizes with separate capacities, indexes, and eviction policies. A frame cache and a
result cache may use the same key without sharing entries. Payloads whose GPU usage overlaps need non-overlapping fallback storage.
A caller may reuse one fallback allocation across outstanding leases only
when every read of the previous contents is ordered before the next overwrite
on the same stream. The old payload must not be read after that overwrite;
this exception never permits overwriting cache-owned data. Each request receives its own `GpuDataAccess`, including concurrent
readers of the same key on different task streams.

Fill publication occurs only in `freeCacheData()` after successful stream
synchronization, not as soon as H2D completes. A second request for a Loading
key waits within its request deadline, then uses fallback if still unavailable. With
all work on the returned stream, no extra H2D event is required. Each access
finishes independently; finishing two accesses on one stream currently causes
two synchronization calls and is not an atomic multi-cache publication.

For GpuI2I integration, the caller should preallocate one master and one
reference fallback buffer in load(), and release them in unload(). Keep the
master while comparisons need it; reuse the reference buffer between ordered
runRef() operations under the rule above. Special paths that retain older
references must be checked before assuming two buffers suffice. Buffer roles
stay in the caller. See [the integration contract](integration/old_vs_new.md#53-caller-owned-masterreference-fallback-buffers).
This is a production integration plan; the demo still has one input buffer.

Batch completion is a [future profiling-driven improvement](open_issues.md#batch-completion-backlog),
not an integration requirement. Keep per-access completion until measurements
show that repeated synchronization calls materially limit throughput.

The demo still uses one frame cache in `StaticData`; it does not automatically
cache CEL/SDD/MI outputs. A result-cache caller must define versioned identity
or reset at parameter changes and be able to recreate evicted data. The
existing frame-shaped metadata and immutable, best-effort eviction contract
remain in place.

Each manager can select `CacheEvictionPolicy::LRU` (default) or `FIFO` at
initialization. FIFO orders successful fill completions; hits never promote
an entry, and live readers remain protected. `StaticDataConfig` and the demo's
`GraphConfig` expose `gpuCacheEvictionPolicy` for the same choice.

Callers can finish their last use with
`access.freeCacheData(submittedSuccessfully, CacheRetention::Discard)` under
either policy. The entry is removed after all readers finish, keeping its
allocation available for another key. The default `Keep` preserves current
behavior. Only use Discard when no future consumers or retries need the key;
pending A-to-B results remain best-effort and still need CPU recovery. See
[cache policies and examples](cache_policies.md).
The [A → CPU tasks → B simulation](result_pipeline.md) adds concrete result tasks,
correctness tests, and a concurrent LRU/FIFO × Keep/Discard pipeline benchmark.
The optional `gpuinfra_cache_policy_benchmark` target measures lookup cost
with long-held readers; see [results and reproduction](../results/benchmarks/20261003-cache-policy/README.md).

## Shared static GPU data

`StaticData` owns `StaticGpuData`: separate detection-zone and distortion-table
allocations, each configured in bytes. They are ordinary device buffers, not
CUDA `__constant__` memory. They do not participate in LRU or frame-cache reset,
and work with `gpuCacheEntries = 0`.

The graph-copy owner prepares them once on the cold path:

```cpp
StaticDataConfig config;
config.runtime = runtime;
config.detectionZoneBytes = detectionZoneBytes;
config.distortionBytes = distortionBytes;
bool succeeded = staticData.init(config);
if (succeeded) {
    succeeded = staticData.uploadStaticData(StaticGpuDataType::DetectionZones, h_zones, detectionZoneBytes, 0);
}
if (succeeded) {
    succeeded = staticData.uploadStaticData(StaticGpuDataType::DistortionTables, h_tables, distortionBytes, 0);
}
if (succeeded) {
    succeeded = staticData.finalizeStaticData();
}
if (!succeeded) {
    staticData.release();
    return false;
}
const void* d_zones = staticData.staticGpuData(StaticGpuDataType::DetectionZones);
```

This example enables both regions. A zero-byte region is disabled: no allocation,
a null data pointer, and uploads to it are rejected. If both regions are disabled,
`init()` finalizes automatically, preserving the existing frame-only lifecycle;
otherwise call `finalizeStaticData()` explicitly. `isInitialized()` means resources
were initialized, while `execute()` also requires static data to be Ready.

Initialization enqueues zeroing on a private nonblocking upload stream. Each
`uploadStaticData(type, source, bytes, offset)` accepts a nonzero in-bounds segment
and synchronizes before returning, so pinned staging storage may immediately be
reused or freed. The caller computes alignment, table indexes, formats and offsets;
the owner checks byte bounds without overflowing. Unwritten gaps remain zero.
The caller must verify that all algorithm-required data has been supplied; the
owner does not track coverage or interpret zero bytes as missing data.

`finalizeStaticData()` waits for remaining preparation and publishes read-only
pointers. Before finalization, pointers are hidden and execution/cache requests
are rejected. Any invalid or failed upload during preparation makes the attempt
Failed: further uploads and finalization fail until release/reinitialization.
After Ready, uploads are rejected without modifying the published state.
Allocation failures abort initialization with no automatic fallback.

The framework must serialize initialization/uploads/finalization and ensure no
execute or outstanding GPU reads overlap release/reinitialization. There are no
static reader leases or runtime reader tracking. Multiple task streams may read
Ready buffers after the cold-path handoff. A frame-cache reset preserves their
addresses and contents. Changing static data requires releasing and reinitializing
the owner at a quiescent boundary; no in-place Ready updates are supported.
Normal `StaticData::release()` releases these buffers and their upload stream
after cache users finish and before GPU contexts are torn down.

The demo leaves both capacities zero; production GpuI2I must supply the real
payloads and cold-path calls. Protocol tests exercise enabled regions.

## Cache statistics

Each manager provides `statisticsSnapshot()` and
`resetCache(CacheStatistics* completedStatistics = nullptr)`.
StaticData forwards them as `cacheStatisticsSnapshot()` and `resetCache()`.
A successful reset returns and clears the previous interval; a rejected reset
preserves both counters and the output argument.

Counters cover final hit/fill/fallback/invalid, fill success/failure, eviction, discard,
stream synchronization failures, first obstruction, fallback reasons, and actual
condition-variable wait time. A synchronization failure marks the manager failed:
new requests return Invalid and reset is rejected until release and
reinitialization (`cacheFailed()` reports it; the framework must propagate it).
The demo caller prints a labeled warmup+timed report after execution. See
[statistics definitions and legacy A/B mapping](cache_statistics.md) before
comparing legacy skip counts with final fallback outcomes.

## Bounded cache waiting

Configure each manager at initialization; the default is **50 ms**:

```cpp
const std::chrono::milliseconds waitTimeout(50);
const bool initialized = cache.initialize(gpuIds, payloadBytes, cacheEntries, waitTimeout);
```

For the frame-cache wrapper, set `StaticDataConfig::gpuCacheWaitTimeout`.
There is no per-request override. Zero disables waiting; negative values are
rejected. A zero-capacity cache always returns fallback immediately.

A matching Loading entry or a full cache with no evictable entry waits on a
condition variable, releasing the mutex. Each call has one steady-clock
deadline, measured from entry to `getCacheData()`; notifications and transitions
between lookup branches never restart it. On wakeup, lookup runs again. A
successful fill permits a hit; failed/abandoned fill permits one contender to
claim CacheFill. The last reader releasing an entry also wakes contenders.
Loading entries and active readers never enter the evictable LRU.

At the deadline, lookup checks availability once more and falls back if still
blocked. This is a wait budget, not a hard wall-clock latency guarantee: thread
scheduling and mutex reacquisition can add delay. Waiters have no FIFO priority
or reserved entry. Holding another access while waiting can consume the entire
budget, so callers should avoid circular dependencies.

Reset/release return false while any request sleeps, even if a producer has
just finished. The manager and borrowed request resources must outlive all
calls and access objects. There is no early publication and no extra CUDA event.

The interactive HTML guides demonstrate the immediate-fallback configuration
(`waitTimeout = 0 ms`); their state transitions also describe the fallback path
after a bounded wait expires.

## Run boundaries and cache reset

`StaticData::resetCache()` is the single cold-path run-boundary operation. It
clears the fixed residency table, entry identities/validity, empty stack, and
global LRU state. It rejects active cache/fallback leases, waiting requests, or an in-progress
fill.

Reset does not call `cudaFree()` or `cudaMalloc()`: every
`GpuDataCacheEntry` device pointer remains allocated and is reused by later fills.
The graph-copy owner must call it after every old-run execution has finished
and before any new-run execution can start. Between two resets, one
`frameId + cameraId + variantId` identity must always represent the same immutable bytes.
If a run reuses an identity with new bytes and reset is skipped, stale cache
data can be returned.
