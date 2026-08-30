# Best-Effort Frame GPU Cache Plan: Temporary One-GPU-per-NUMA Scope

**Implementation status:** Implemented.

## 1. Scope and goal

The current graph copy explicitly supports exactly one eligible GPU per NUMA
node. The goal is to preserve GPU input across different task instances when
possible, while keeping correctness independent of cache residency:

```text
RAM / immutable FrameCpuAtom
  ├─ cache miss -> H2D -> task-private d_input -> algorithms
  └─ cache fill -> H2D -> GpuCacheEntry.GpuReplica.d_data
                              |
                    later task instance
                              |
                         cache hit
```

The cache is a performance bonus. A miss never changes scheduler selection and
never fails merely because all cache entries are busy.

## 2. Golden constraints

[`graph.md`](graph.md) remains authoritative.

- Scheduler selection still matches a ready frame, a free task instance, and a
  free NUMA-local worker without consulting GPU residency.
- Workers are NUMA-bound, not GPU-bound.
- Every task instance remains permanently bound to one GPU.
- `DummyTask::execute()` remains synchronous.
- The hot path performs no `cudaMalloc()` or `cudaFree()`.
- `TaskGpuResources` continues to own stream, pinned staging, scratch,
  task-private resources, and the new fallback input buffer.
- Algorithms read the input; they do not mutate it.

## 3. Two independent layers

```text
StaticData
  ├─ fixed frame layout + resetCache() run boundary
  └─ GpuCacheManager
       ├─ fixed open-addressing GpuDataKey -> entry index table [~2K]
       ├─ empty-entry stack + intrusive inactive-entry LRU
       └─ GpuCacheEntry[K]
            cached metadata + cache state + leases + LRU links
            └─ GpuReplica(gpuId, d_data, valid)
```

`K` is:

```text
GraphConfig::gpuCacheEntries
```

The default configured value is 4. Zero is valid and means pure task fallback.
There is no fixed 220-frame capacity or frame registration list.

### `FrameCpuAtom`

- Owns CPU bytes and intrinsic `FrameMetadata`.
- Owns the preallocated `JobResult` and CEL/SDD/MI output vectors.
- Remains graph-owned and valid for the configured logical frame lifetime.
- Is the reconstruction source on cache miss.

### `StaticData` frame validation

- Stores one fixed frame layout.
- Accepts any incoming `GpuDataKey(frameId, cameraId)` whose metadata matches
  that layout.
- Stores no per-frame key or metadata registry.
- Stores no execution state or `FramePhase`; graph-owned collections, queues,
  counters, and submitted flags define them.
- Stores no `JobResult`; result lifetime follows `FrameCpuAtom`.
- Owns no CPU bytes and no per-frame device allocation.

### `GpuCacheEntry`

- Is a reusable GPU cache entry, not a logical frame record.
- Temporarily stores cached metadata, `Empty`/`Loading`/`Valid` state, active
  lease count, and intrusive LRU links.
- Directly owns GPU-keyed persistent `GpuReplica` allocations and per-replica
  validity.
- Can represent different logical frames over graph lifetime without
  reallocating its device buffer.
- Does not own logical state, results, streams, scratch, or fallback storage.
- Currently allocates exactly one replica because the temporary topology has
  one GPU per NUMA graph copy.

### `TaskGpuResources`

- Owns `stream`, `h_in`, `d_input`, `d_scratch`, GPU/resource identity, and
  context reference.
- Allocates `d_input` once in `DummyTask::load()` and frees it in `unload()`.
- Uses `d_input` whenever the GPU cache cannot immediately provide an entry.

## 4. Cache capacity and allocation lifetime

Cold initialization:

```text
DummyTask::load()
  -> allocate one d_input fallback per task instance

StaticData::init()
  -> store fixed frame layout
  -> allocate K GpuCacheEntries
  -> allocate one persistent GpuReplica per entry
  -> allocate fixed residency table and empty/LRU victim structures
```

Later cold run boundaries do not reallocate the device buffers:

```text
StaticData::resetCache()
  -> reject active leases
  -> clear entry identities and replica validity
  -> clear residency, empty stack, and global LRU state
  -> retain every GpuReplica.d_data allocation
```

The framework must call `resetCache()` after all old-run executions finish and
before any new-run execution starts. Between two resets, one
`frameId + cameraId` must always represent the same immutable bytes. Reset is
mandatory before a new run reuses an identity with different bytes.

Teardown happens only after workers stop:

```text
StaticData::release()
  -> release all cache-entry device allocations
DummyTask::unload()
  -> release d_input, scratch, algorithm resources, staging, stream
```

No allocation, free, vector growth, or hash rehash occurs during cache acquire
or task execute. Cache misses insert into already allocated fixed table slots.

`DummyGraph` keeps separate `warmupAtoms` and `timedAtoms` collections. Those
collections are the only `FramePhase` authority. Phase-submitted flags and the
ready/in-flight/terminal bookkeeping own execution progress; `StaticDataConfig`
and `StaticData` contain no phase or execution-state field.

## 5. Access contract

`GpuDataAccess` reports its source:

| Source | Meaning | `needsUpload()` | Writable pointer |
| --- | --- | --- | --- |
| `CacheHit` | Matching valid cache entry | false | no |
| `CacheFill` | Miss reserved an inactive cache entry | true | cache-entry buffer |
| `TaskFallback` | No entry can be used immediately, or capacity is zero | true | task `d_input` |
| `Invalid` | Contract/metadata/GPU validation failed | n/a | no |

The scoped access owns neither allocation nor stream. `complete()` synchronizes
the captured stream once and then releases or publishes the lease. Destruction
of an incomplete access synchronizes submitted work and aborts it.

## 6. Lookup and replacement

`GpuCacheManager::acquire(metadata, resources)` holds a short cache metadata
mutex and performs an average `O(1)` lookup in a fixed open-addressing table.
The table holds at most K resident/loading keys in at least 2K slots. Misses
erase and insert keys with linear probing and backward-shift deletion. An
empty-entry stack or the head of an intrusive inactive-entry LRU supplies a
victim in `O(1)`. Acquire performs no allocation, rehash, container growth, or
`K`-entry array scan.

Rules are evaluated in order:

1. Any key with the fixed layout is eligible; a resident key with different
   complete metadata is invalid.
2. A matching `Valid` entry with a valid local replica returns `CacheHit` and
   increments its active-reader count.
3. A matching `Loading` entry returns `TaskFallback`; it does not wait.
4. On a miss, prefer an `Empty` entry.
5. Otherwise choose the least-recently-used `Valid` entry whose active-reader
   count is zero.
6. If every entry is active/loading, return `TaskFallback`.

CUDA work never runs while this mutex is held. Multiple immutable cache-hit
readers may coexist. An active entry cannot be evicted.

## 7. Fill publication and failure

A reserved cache entry transitions:

```text
Empty or inactive Valid
  -> Loading
  -> task enqueues H2D and algorithms
  -> stream synchronization succeeds
  -> Valid
```

Only successful `GpuDataAccess::complete(true)` publishes the new cached
metadata/replica. Submission failure, synchronization failure, or RAII abort
resets a fill entry to `Empty`.

A failed `CacheHit` releases its reader count but preserves the existing
immutable payload. A fallback completion never mutates cache state. Cache
release rejects all active cache and fallback leases.

## 8. Task execution

The scheduler still calls:

```cpp
bool DummyTask::execute(FrameCpuAtom& atom, StaticData& staticData);
```

The task performs:

```text
validate atom layout and preallocated outputs
  -> make the task GPU current
  -> StaticData::acquireGpuData(metadata, resources)
       -> validate fixed layout
  -> if needsUpload: atom.data -> h_in -> access.writableData()
  -> CEL / SDD / MI read access.data()
  -> algorithm D2H
  -> access.complete(result.ok), including one stream sync
  -> collect into FrameCpuAtom.result
```

CEL, SDD, and MI interfaces are unchanged. They do not know whether their
input came from a cache entry or `TaskGpuResources::d_input`.

## 9. Correctness boundary

This best-effort cache is correct because the current GPU input is immutable
and reproducible from `FrameCpuAtom` on every miss.

If a future TaskA produces a mutable GPU-only intermediate needed by TaskB,
that intermediate cannot use this eviction/fallback rule unless it has a
defined host spill or recomputation source. It needs a separate frame-owned
authoritative output plane or a different non-evictable lifetime contract.

## 10. Configuration and scheduler impact

`GraphConfig` contains:

```cpp
std::size_t gpuCacheEntries = 4;
```

`DummyGraph::initializeOnNumaNode()` copies this value into
`StaticDataConfig`. It does not add a CLI option. The ready-frame queue,
`freeTasks`, worker selection, checkout/return order, phase gate, and
cancellation logic are unchanged.

## 11. Verification

CUDA integration tests cover:

- more than 220 unregistered incoming frames with only two cache entries;
- fixed-table collision and backward-shift deletion;
- full-metadata/layout rejection;
- camera-ID separation and arbitrary incoming keys;
- reset rejection with an active lease;
- reset without releasing or reallocating device memory;
- configured capacity and capacity zero;
- first fill, later hit, and stable cache device pointer;
- same-frame `Loading` fallback;
- all-active fallback and release rejection with live leases;
- fill failure and RAII abort returning an entry to `Empty`;
- inactive LRU eviction;
- wrong-GPU rejection;
- cached input surviving a task-instance change;
- end-to-end correctness using only task fallback;
- both execution models, cancellation, task checkout, and teardown.

## 12. Future two-GPU-per-NUMA path

The public task API and scheduler stay unchanged. The future implementation
extends cache-entry internals:

1. Allocate one `GpuReplica` per eligible GPU for every cache entry.
2. Discover P2P capability and warm transfer paths during cold initialization.
3. On a matching frame with an invalid local replica, reserve a per-replica
   fill and lazily copy from another valid replica.
4. If migration cannot be reserved immediately, use task fallback.
5. Publish local replica validity only after stream synchronization.
6. Keep eviction at whole-entry granularity and forbid eviction while any
   replica lease/fill is active.

No future step may make scheduler selection residency-aware, bind workers to a
GPU, or allocate device memory in the hot path.
