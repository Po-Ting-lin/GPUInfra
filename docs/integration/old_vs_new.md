# Old `GpuCache` vs. GPUInfra

## 1. Purpose

This document evaluates whether GPUInfra can replace the OCR-extracted cache
implementation and its caller contract:

- [`GpuCacheCaller.md`](../../../IosOCRServer/GpuCacheCaller.md)
- [`GpuCache.h`](../../../IosOCRServer/GpuCache.h)
- [`GpuCache.cpp`](../../../IosOCRServer/GpuCache.cpp)

The evaluation is read-only and is based on the intended behavior described by
the caller document and recoverable source logic. The supplied legacy source is
OCR output and is not a complete buildable IosOCRServer checkout.

## 2. Conclusion

GPUInfra can become the replacement cache core, but the current implementation
cannot directly replace the three legacy files without adaptation.

```text
Conceptual cache replacement       -> feasible
Drop-in source/API replacement     -> not currently feasible
Replacement without scheduler work -> feasible
Implemented first step             -> composite key, run reset,
                                      average O(1) lookup/victim selection
Remaining work                     -> dual fallback, multi-access completion,
                                      and integration adapter
```

The most important correctness gap is that one legacy task execution can hold
two adjacent frames, master and reference, at the same time. GPUInfra can hold
two cache-entry leases, but `TaskGpuResources` currently owns only one fallback
`d_input`. If both acquisitions fall back, the second H2D can overwrite the
first input before computation completes.

## 3. Legacy behavior

The legacy design has the following contract:

- One file-scope `GpuCache` object is shared by five task instances.
- Each task calls `register_thread()` from `notifyParameter()` only once across
  multiple runs.
- The first successful registration allocates the shared cache; later task
  instances register their CUDA streams.
- A task execution obtains both a master frame and a reference frame using the
  `(frame ID, camera ID)` identity.
- A cache fill uploads the complete frame because later users may request a
  different ROI.
- A cache skip uses a task-local device buffer and may upload only the required
  ROI.
- A cache hit skips H2D.
- If another stream is loading the same frame, `getCacheFrame()` can return the
  loading stream so the caller can stall and retry.
- `freeCacheFrame()` decrements the active reference count and publishes a
  completed fill.
- `resetCache()` invalidates cached identities between runs without freeing the
  device allocation.
- The cache reserves an additional constant-frame memory area.
- Cache modality settings can selectively skip frame IDs or camera IDs.

The legacy replacement policy is a direct-mapped, two-way cache. The set is
derived from the frame ID and the configured modality; two ways use counters
and an age bit for safe replacement.

## 4. Compatibility matrix

| Legacy requirement | Current GPUInfra behavior | Assessment |
| --- | --- | --- |
| One cache shared by multiple task instances | `StaticData` owns one graph-copy `GpuCacheManager` | Compatible and has clearer ownership |
| No hot-path `cudaMalloc()`/`cudaFree()` | Cache entries and task fallback buffers are allocated during initialization | Compatible |
| Cache hit, fill, active readers, and safe replacement | `GpuDataAccess`, `GpuCacheState`, and `activeAccesses` implement scoped leases | Compatible |
| Full-frame immutable cache payload | Each entry owns a fixed-size persistent GPU payload | Compatible if callers never modify cached input |
| Master and reference held simultaneously | Multiple cache leases are possible, but there is only one task fallback buffer | Correctness blocker |
| Identity is `(frame ID, camera ID)` | Explicit `GpuDataKey` contains frame ID and camera ID | Compatible |
| Frame IDs may be reused in a later run | `resetCache()` clears old residency before reused identities arrive | Compatible when the caller provides a quiescent run-boundary hook |
| Cache invalidation without freeing allocation | `resetCache()` clears residency/validity while retaining device pointers | Compatible |
| Frames may be discovered during a run | Any incoming key with the fixed layout may enter the cache; no registration list is required | Compatible |
| Loading frame can cause stall/retry | A matching `Loading` entry immediately returns task fallback | Correct but different performance and caller flow |
| Separate master/reference fallback buffers | `TaskGpuResources` has one `d_input` | Missing |
| ROI-only upload on fallback | The cache core lets the caller choose the copy, but the demo task uploads a full frame | Can be preserved in an adapter |
| Cache allocation failure disables only the cache | Cache initialization failure fails `StaticData::init()` | Intentional difference: fail initialization; no automatic capacity-zero fallback |
| Constant-frame GPU memory | StaticData owns separate byte-sized DZ/distortion buffers through StaticGpuData | Implemented; caller supplies uploads and cold-path handoff |
| Direct-mapped two-way O(1) lookup | A fixed open-addressing table gives average O(1) lookup; empty stack and intrusive global LRU give O(1) victim selection | Scalable alternative policy; exact legacy placement/age behavior is intentionally not preserved |
| Cache modality filters | Not implemented | Optional only if they are performance controls |
| Hit/miss/skip/debug counters | Planned but not implemented | Operational gap, not a correctness blocker |
| Many registered task streams | Every task already owns its stream; no fixed stream table is needed | Compatible and removes the legacy limit |
| Multiple GPUs in one NUMA graph copy | Temporary GPUInfra implementation accepts exactly one GPU | Future work |

## 5. Required changes and implementation status

### 5.1 General cache key

Implemented: `FrameMetadata` now carries an explicit immutable key:

```cpp
struct GpuDataKey {
    std::uint64_t frameId;
    std::uint32_t cameraId;
};
```

Identity distinguishes frames and cameras exactly as the legacy API does.
`unique_id` from the legacy API is a task/stream owner ID and must not be
treated as frame identity.

Packing frame and camera IDs into one `uint64_t` can work temporarily, but an
explicit key is clearer and safer for future non-frame data extensions.

### 5.2 Per-run state without device reallocation

Implemented as a cold-path run boundary:

```text
StaticData::resetCache()
  -> verify that no leases are active
  -> clear the fixed residency table and entry validity
  -> retain every device allocation
```

The operation rejects active cache or fallback leases and performs no
`cudaMalloc()`/`cudaFree()`.

This preserves the requirement that device allocations last until graph
teardown while preventing stale hits when IDs are reused in a later run.

If frame IDs are globally unique and their bytes never change for the entire
graph lifetime, reset is not needed for identity correctness. The legacy
`resetCache()` behavior indicates that the production caller should invoke it
at every quiescent run boundary.

### 5.3 Caller-owned master/reference fallback buffers

The GpuI2I integration plan is for each caller/task instance to preallocate
one master fallback buffer and one reference fallback buffer during load,
then free them during unload. These are caller-specific resources, not roles
to add to the generic GpuCacheManager or TaskGpuResources.

- loadMaster(): use the master cache access when available; on TaskFallback,
  upload into the caller's master buffer.
- runRef(): use the reference cache access when available; on TaskFallback,
  upload into the caller's reference buffer, then enqueue its consumers.
- Keep the master data available for every comparison that still needs it.
  Reuse the reference buffer only after all reads of its previous contents
  have been enqueued before the next overwrite on the same stream.
- Finish each access after all relevant GPU work has been submitted.

Supply the appropriate buffer through GpuCacheRequest::d_fallback and
fallbackBytes. Multiple outstanding fallback leases may borrow the same
reference allocation only under the ordering rule above; a fallback lease
does not preserve old contents after an explicitly ordered overwrite.
Never reuse a cache-owned pointer this way: cached payloads remain immutable.

The production review reports up to 64 reference cache leases. This is not a
requirement for 64 reference fallback allocations. Before integration, verify
GPU_OPTIMIZE_MASTER_AS_REF, m_bNeighborsDiff, mask-out-by-ref, and getDefects()
for any reads of an older reference after the next upload. Such paths need
additional buffers or a changed submission order. The full production source
has not yet been independently verified in this workspace.

activeFallbackAccesses counts leases, not distinct buffer addresses. Every
lease must finish separately, including leases borrowing the same allocation.
Reset/release remain blocked until all accesses and waiting requests finish.
The caller owns the allocation and keeps it alive through synchronization.

`GpuDataAccess` now supports move construction and assignment while remaining
non-copyable. Store master/reference accesses in caller-owned member slots
across loadMaster/runRef/getDefects. A move preserves the existing lease and
invalidates its source; finish each saved access explicitly. Assignment over a
live access aborts its previous lease through synchronized RAII cleanup, so it
must not replace normal success completion. Keep managers and CUDA resources
alive until all saved accesses finish.

### 5.4 Backlog: complete multiple accesses with one synchronization

Keep the current per-access freeCacheData() API for initial integration.
If the caller holds master plus 64 reference accesses and finishes all of
them individually, there are 65 cudaStreamSynchronize() calls, regardless
of the number of fallback allocations.

When all work has already been queued on one stream, the first successful
synchronization covers that queued work. Later calls do not repeat the GPU
computation; they still incur synchronization-call overhead. Its throughput
impact has not been measured, so batch completion is not an integration
prerequisite.

Profile the real caller before adding a batch/session API. If repeated
completion calls materially affect throughput, consider synchronizing once
per stream and then publishing/rolling back each access. This must preserve
failure handling and lease cleanup; it does not imply atomic publication
across caches or early publication after H2D.

### 5.5 Cache lookup policy and capacity

The legacy cache capacity is derived from MB and may contain many frames.
GPUInfra now uses a fixed open-addressing `GpuDataKey -> entry index` table for
average `O(1)` lookup, a reserved empty-entry stack, and an intrusive inactive-
entry global LRU for `O(1)` victim selection. The table stores only the at most K
resident/loading keys in preallocated storage sized to at least 2K slots.
Acquire may insert and erase keys, but it never allocates, grows, or rehashes the
table. Backward-shift deletion prevents tombstones from accumulating under
long-running cache churn.

The capacity conversion must use checked arithmetic:

```text
cache entries = floor(configured cache bytes / bytes per full frame)
```

Cache policy must remain invisible to scheduler selection.

### 5.6 Cache allocation failure

The legacy server continues with local buffers if cache allocation fails.
GPUInfra intentionally rejects initialization instead:

```text
requested cache allocation fails
  -> report the failure
  -> clean up partially initialized resources
  -> return initialization failure
```

Explicitly configuring capacity zero is a supported no-cache mode: callers use
their preallocated fallback buffers. Requesting nonzero capacity and failing to
allocate it is an initialization error, not an instruction to disable the cache.
Do not automatically retry with capacity zero or add a cacheRequired switch.

Failure to allocate mandatory task fallback buffers or configured static GPU
data must also fail initialization. These failures must not be deferred until
execute().

### 5.7 Constant GPU data

Implemented by StaticData-owned StaticGpuData with separate detection-zone and
distortion-table allocations. StaticDataConfig specifies each capacity in bytes;
zero disables that region. These allocations are independent of frame-cache
capacity and are retained across resetCache().

The graph-copy owner initializes once, uploads segments through
uploadStaticData(type, source, bytes, offset), then calls finalizeStaticData()
before execute. A dedicated stream handles zeroing and uploads. Each upload
synchronizes before returning so legacy pinned staging storage can be reused.
Finalization publishes read-only pointers; preparation failures prevent execution.

The caller retains responsibility for DZ alignment, distortion packing/indexes
and required-data completeness. The framework guarantees no GPU readers during
initialization or release/reinitialization. No instance-ID-zero convention,
runtime updates, LRU eviction or best-effort fallback is used for this storage.
Production caller wiring remains to be integrated; see README for the API.


## 6. Recommended replacement architecture

```text
Real graph copy
  └─ StaticData
       ├─ fixed data layout + resetCache() run boundary
       ├─ GpuCacheManager
       │    ├─ fixed open-addressing residency table
       │    ├─ empty stack + intrusive global LRU
       │    └─ persistent GpuCacheEntry allocations
       └─ StaticGpuData: independent DZ/distortion buffers + upload stream

Task instance
  └─ TaskGpuResources
       ├─ stream
       ├─ master fallback or caller-owned fallback slot 0
       ├─ reference fallback or caller-owned fallback slot 1
       ├─ scratch
       └─ algorithm-private buffers

Task::execute()
  ├─ acquire master access
  ├─ acquire reference access
  ├─ CacheFill: full-frame H2D
  ├─ TaskFallback: required ROI H2D
  ├─ run comparison algorithms
  ├─ synchronize once
  └─ call freeCacheData() on both accesses
```

There should be one `StaticData` instance per graph copy, not one
process-global mutable cache shared across unrelated NUMA/GPU graph copies.

## 7. Integration choices

### Recommended: change the task caller to RAII access

Modify `loadMaster()` and `runRef()` to retain `GpuDataAccess` objects until
`getDefects()` has finished. This removes manual pointer-to-cache-line recovery
and makes every early-return path release its lease safely.

The graph scheduler does not need to know cache residency and does not need to
change its ready-frame, task-instance, or worker selection.

Only the framework extension points are required:

- graph-copy initialization and release for `StaticData`;
- a quiescent run-boundary hook for `resetCache()`;
- `StaticData&` passed to task execution;
- workers stopped before cache release.

### Alternative: preserve the legacy `GpuCache` API

An adapter can expose `register_thread()`, `getCacheFrame()`, and
`freeCacheFrame()` while internally using `GpuCacheManager`. This is possible
but not preferred because the adapter must maintain an additional table of
outstanding RAII accesses indexed by task, pointer, frame ID, and camera ID.
It must also emulate loading-stream retry, constant memory, reset counters, and
legacy statistics if callers rely on them.

That adapter would preserve caller source compatibility but recreate much of
the legacy bookkeeping that GPUInfra is intended to remove.

## 8. Behavior that must be confirmed in the real caller

Before implementation, obtain the original task source and confirm:

1. Cached master/reference pointers are read-only after upload.
2. Whether frame IDs or `(frame ID, camera ID)` pairs are reused across runs.
3. Whether a quiescent run-boundary hook can call `resetCache()` exactly once
   per graph copy.
4. Whether `loading_stream` stall/retry is required for correctness or only for
   hit-rate optimization.
5. Whether constant-frame memory APIs are called.
6. Whether cache modality filters are product behavior or debug/performance
   controls.
7. The maximum cache size and resulting number of cache entries.
8. Where the legacy caller synchronizes CUDA work before `freeCacheFrame()`.
9. The actual number of GPUs per NUMA graph copy.
10. The server's C++ standard, CUDA version, build system, and existing context
    ownership.

## 9. Source limitations

The supplied OCR source cannot currently be compiled or used for an A/B test:

- `GpuCache.cpp` includes `GpuI2ICache.h` instead of the supplied
  `GpuCache.h`.
- `GPU_CACHE_NUM_CONST_FRAME8` does not match the declared constant.
- `CacheStatistics` member spellings differ between the header and source.
- `close()` and `addDefectsDebug()` are declared but have no definitions in the
  supplied file.
- The IosOCRServer directory contains no build configuration or actual Task
  caller implementation.

These appear consistent with OCR damage or an incomplete extraction. The
design-level feasibility result remains valid, but final API and synchronization
decisions require the original caller code.

## 10. Final decision

GPUInfra is a suitable foundation for replacing the legacy cache because its
graph-scoped ownership, persistent allocations, cache leases, fallback path,
and scheduler-independent selection match the intended direction.

The first implementation step is complete and tested:

- composite key including frame and camera identity;
- run-boundary reset without freeing device memory;
- arbitrary incoming-frame caching without a registration list;
- fixed-capacity average `O(1)` residency lookup and `O(1)` global-LRU victim
  selection.

Replacement should proceed only after the remaining correctness requirements
are implemented and tested:

- independent master/reference fallback storage;
- one-sync completion of multiple leases;
- graph-copy lifecycle integration;
- separate constant storage if used.

After those changes, the old cache can be removed without modifying scheduler
selection. Cache modality, loading stall behavior, statistics, and lookup
policy can then be matched or intentionally replaced based on benchmark and
product requirements.
