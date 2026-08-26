# GPUInfra architecture

GPUInfra implements the one-task golden protocol in [`graph.md`](graph.md):

```text
start -> DummyTask -> end
```

CEL, SDD, and MI are private operations inside `DummyTask`. They are not
independent graph tasks. The current implementation supports exactly one GPU in
each NUMA graph copy while keeping GPU-keyed cache internals for a future
multi-GPU extension.

## 1. Runtime topology

```text
process
  |
  +-- GpuContextManager
  |     +-- GpuContext(gpu=0, numa=0)
  |     `-- GpuContext(gpu=1, numa=1)
  |
  +-- DummyGraph(numa=0)
  |     +-- NUMA-local worker pool
  |     +-- GPU-bound DummyTask pool for GPU 0
  |     +-- FrameCpuAtom collections
  |     `-- StaticData
  |           +-- registered FrameMetadata[N] + ID index
  |           `-- GpuCacheManager -> GpuCacheEntry[K]
  |
  `-- DummyGraph(numa=1)
        +-- NUMA-local worker pool
        +-- GPU-bound DummyTask pool for GPU 1
        +-- FrameCpuAtom collections
        `-- StaticData
              +-- registered FrameMetadata[N] + ID index
              `-- GpuCacheManager -> GpuCacheEntry[K]
```

`N` is the configured logical frame count. `K` is the smaller of configured
cache capacity and `N`; the default configured capacity is 4.

## 2. Scheduler boundary

The scheduler has three independent resources:

1. a ready graph-owned `FrameCpuAtom`;
2. a free `DummyTask` instance;
3. a free NUMA-local graph worker.

Under `schedulerLock`, the worker claims the frame and task, removes the task
from `freeTasks`, then releases the lock before execution. The task is returned
only after `execute()` completes. This prevents concurrent reuse of one task
instance at graph level.

Cache residency, LRU order, and fallback availability are not scheduler inputs.
The worker remains NUMA-bound; the selected task remains GPU-bound.

## 3. Ownership

| Component | Ownership and lifetime |
| --- | --- |
| `GpuContextManager` | Process-wide discovery, NUMA mapping, task registry, primary-context lifetime |
| `GpuContext` | One GPU's identity, NUMA node, retained context, active task table |
| `DummyGraph` | One NUMA graph copy, workers, task pool, warmup/timed CPU-atom collections, queues, `StaticData`, parameters |
| `FrameCpuAtom` | CPU input byte vector, intrinsic metadata, and preallocated result |
| `StaticData` | Graph NUMA, immutable registered metadata/index, and bounded GPU cache |
| `GpuCacheManager` | Fixed cache-entry array, short metadata mutex, lease counts, LRU |
| `GpuCacheEntry` | One reusable entry with GPU-keyed persistent replicas and validity bits |
| `GpuDataAccess` | Scoped non-owning cache/fallback view; synchronization and publish/abort |
| `DummyTask` | GPU-bound reusable execution lane and CEL/SDD/MI objects |
| `TaskGpuResources` | Stream, pinned `h_in`, fallback `d_input`, scratch, GPU/NUMA/context identity |
| CEL/SDD/MI | Private device output, pinned D2H staging, geometry, parameters |
| Graph worker | NUMA affinity and one temporary host call only |

The essential split is:

```text
CPU input and host result       -> FrameCpuAtom
logical execution state         -> DummyGraph collections/queues/counters
best-effort cached GPU input     -> GpuCacheEntry / GpuReplica
correctness fallback GPU input  -> TaskGpuResources::d_input
CUDA execution lane             -> DummyTask / TaskGpuResources
host execution                  -> temporarily selected graph worker
```

## 4. Cold path

`DummyGraph::initialize()` runs setup on a NUMA-pinned thread:

```text
create/load every DummyTask
  -> allocate stream, h_in, d_input, scratch, algo-private resources
  -> register shared parameter schema
  -> seal and notify immutable values
  -> create all FrameCpuAtoms and preallocate their result buffers
  -> StaticData::init()
       -> validate unique IDs/layouts
       -> copy registered FrameMetadata[N]
       -> build immutable ID-to-metadata hash
       -> create GpuCacheManager with GpuCacheEntry[K]
       -> allocate one replica per entry on the graph GPU
  -> start and affinity-check workers
```

There is no 220-frame limit. The demo default happens to configure 20 warmup
plus 200 timed frames, but those 220 registered frames use only four device
cache entries by default.

`FramePhase` is graph-owned: `warmupAtoms` and `timedAtoms` define membership.
The phase-submitted flags, ready queue, in-flight count, and terminal count own
execution progress. `StaticData` stores no phase, execution, or result state.

## 5. Logical lookup and cache lookup

`DummyTask::execute(atom, staticData)` first validates the registered metadata:

```text
frame ID -> immutable unordered_map -> metadata index -> full metadata check
```

This is average `O(1)` and provides logical correctness. After validating NUMA,
input layout, and result shape, the task requests GPU access:

```text
GpuCacheManager::acquire(metadata, TaskGpuResources)
```

The cache scans `K` preallocated entries under a short mutex. This is `O(K)`,
default `K=4`, with no allocation or CUDA call under the mutex.

## 6. Cache states and leases

Each `GpuCacheEntry` is in one state:

- `Empty`: immediately available for a miss;
- `Loading`: one cache fill owns the entry but has not published it;
- `Valid`: the cached metadata and local replica are readable.

`GpuDataAccessSource` describes the selected execution path:

- `CacheHit`: matching valid entry; immutable reader count increases;
- `CacheFill`: empty or inactive-LRU entry reserved for H2D;
- `TaskFallback`: no cache entry can be used immediately, or capacity is zero;
- `Invalid`: metadata/resource/GPU contract failed.

Multiple hit readers may coexist. An entry with active readers is not evicted.
If the same frame is loading or all entries are active, the request immediately
uses task fallback instead of waiting.

## 7. CUDA hot path

```text
FrameCpuAtom + StaticData registry validation
  -> make selected task GPU current
  -> acquire GpuDataAccess
  -> CacheHit: use cache pointer directly
     CacheFill/TaskFallback: atom.data -> h_in -> H2D to writableData()
  -> CEL / SDD / MI kernels read data()
  -> CEL / SDD / MI D2H
  -> GpuDataAccess::complete()
       -> cudaStreamSynchronize(task stream)
       -> publish fill or release lease
  -> collect into FrameCpuAtom.result
```

Hot-path invariants:

- no CUDA/host allocation or free;
- no container growth or hash mutation;
- no manager mutex and no `cudaDeviceSynchronize()`;
- one task-stream synchronization per synchronous `execute()`;
- no concurrent use of one task's mutable resources;
- algorithms receive one const input pointer and do not know its source.

The compile-time macros in `WorkloadSizing.h` independently control submitted
work without changing any function API or buffer layout:

```text
effective H2D traffic on upload = input bytes × GPUINFRA_H2D_SIZE_MULTIPLIER
effective D2H traffic per algo  = output bytes × GPUINFRA_D2H_SIZE_MULTIPLIER
compute repetitions per element = GPUINFRA_COMPUTE_SIZE_MULTIPLIER
```

All copies remain ordered on the selected task stream. A cache hit still
submits no H2D, regardless of the H2D multiplier. These controls affect traffic
and compute time only; they do not increase persistent GPU or host allocation.

## 8. Cache correctness

A cache fill is published only after successful stream synchronization. Failed
submission, failed synchronization, or RAII abandonment resets the entry to
`Empty`. A failed cache reader releases its lease without invalidating the
immutable cached payload. Fallback completion changes no cache entry.

Correctness is independent of cache size because the current input can always
be rebuilt from immutable `FrameCpuAtom`. Capacity zero is therefore a valid
baseline, and capacity changes should affect only H2D frequency and VRAM.

This rule does not cover mutable GPU-only intermediates. Such data needs a
separate authoritative frame-owned output plane or a defined spill/recompute
contract before eviction is safe.

## 9. Memory lifetime

```text
FrameCpuAtom / registered metadata:
  graph setup -> graph teardown

GpuCacheEntry replicas:
  StaticData::init() cudaMalloc -> repeated fill/hit/eviction -> release cudaFree

TaskGpuResources::d_input:
  DummyTask::load() cudaMalloc -> repeated fallback H2D -> unload cudaFree
```

The same entry replica pointer is reused when LRU replacement changes which logical
frame it caches. No cache replacement allocates or frees memory.

## 10. Teardown and failures

Execution failure raises shared cancellation. Queued frames become terminal,
in-flight calls return, all workers join, and teardown runs on a NUMA-pinned
thread:

```text
StaticData::release() cache allocations
  -> clear CPU atoms
  -> DummyTask::unload() task/algo allocations and streams
  -> unregister tasks
  -> GpuContextManager::shutdown() retained contexts
```

`GpuCacheManager::release()` rejects live cache or fallback leases. Normal graph
teardown reaches it only after workers have joined.

## 11. Current and future topology

Current initialization requires exactly one GPU ID. Future two-GPU support
keeps the scheduler and task API unchanged, but adds a persistent replica per
cache entry per GPU plus lazy P2P/staged local-replica fills. See
[`frame_gpu_data_plan.md`](frame_gpu_data_plan.md).

## 12. Verification

The CUDA tests verify:

- registered frames beyond 220 with a two-entry cache;
- duplicate ID and complete metadata rejection;
- effective-capacity clamping and capacity zero;
- cache fill/hit/loading fallback/busy fallback;
- stable pointers, fill failure, RAII abort, and LRU eviction;
- release rejection while any lease is active;
- wrong-GPU rejection;
- data reuse across task instances and pure-fallback result correctness;
- task lifecycle, thread mobility, exclusive graph checkout, both execution
  models, cancellation, and idempotent cleanup.
