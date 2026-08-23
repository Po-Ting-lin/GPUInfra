# Frame GPU Cache Plan: Future Multi-GPU-per-NUMA Extension

**Status:** Future design. The one-GPU best-effort cache foundation is
implemented in [`frame_gpu_data_plan_tmp.md`](frame_gpu_data_plan_tmp.md).

## 1. Goal

Allow successive stages of one immutable frame to run on task instances bound
to different GPUs, without changing scheduler selection:

```text
FrameCpuAtom
  -> TaskA on GPU 0
  -> best-effort FrameSlot cache entry
  -> TaskB on GPU 1
  -> TaskC on GPU 0
```

The scheduler remains unaware of residency. A local cache miss must always be
resolved inside frame-data access by migration, H2D fallback, or cache miss
fallback.

## 2. Constraints

- Workers remain NUMA-bound only.
- Task instances remain permanently GPU-bound.
- Different stages of one frame may select different GPUs.
- No hot-path `cudaMalloc()`/`cudaFree()`.
- Task streams, fallback input, scratch, and private buffers remain in
  `TaskGpuResources`.
- Logical execution state remains graph-owned; host results remain in
  `FrameCpuAtom`, while `StaticData` stores only immutable registered metadata.
- GPU cache entries remain bounded and best-effort.
- Current synchronous `execute()` semantics remain the first implementation
  boundary.

## 3. Extended cache entry

For `G` eligible GPUs and cache capacity `K`:

```text
FrameGpuCache
  └─ FrameSlot[K]
       ├─ cached FrameMetadata
       ├─ whole-entry LRU / active state
       └─ FrameGpuData
            ├─ Replica(gpu0): d_data + Empty/Loading/Valid
            ├─ Replica(gpu1): d_data + Empty/Loading/Valid
            └─ ...
```

Every replica allocation is created during `StaticData::init()` and retained
until `StaticData::release()`. Device cache VRAM becomes:

```text
K × G × FrameBytes
```

This intentionally trades VRAM for stable pointers and allocation-free access.

## 4. Access paths

### Local hit

```text
matching entry + local Valid replica
  -> CacheHit
  -> task reads local pointer
```

### Local replica fill

```text
matching entry + another GPU has Valid replica + local replica inactive
  -> reserve local replica Loading
  -> enqueue P2P or staged copy on selected task stream
  -> run task only after local copy ordering is established
  -> stream sync
  -> publish local replica Valid
```

This likely adds `FrameGpuAccessSource::ReplicaFill`. Other valid replicas stay
valid because the payload is immutable.

### Frame cache miss

Use the current policy:

1. reserve `Empty`, otherwise inactive whole-entry LRU;
2. H2D from immutable `FrameCpuAtom` into the selected GPU replica;
3. publish entry and replica only after successful synchronization;
4. if no entry is immediately available, use selected task's `d_input`.

### Migration unavailable

If the matching entry is being filled, the local replica is busy, P2P cannot
be reserved, or no safe staged-copy resource is available, use task fallback.
Do not wait inside the scheduler lock and do not redirect work to another task.

## 5. Transfer path initialization

Cold initialization must:

1. validate all GPU IDs belong to the graph copy's NUMA scope;
2. query peer-access capability for each ordered GPU pair;
3. enable supported peer access before workers start;
4. preallocate any staged-copy host buffers/events needed by fallback paths;
5. test/warm each selected path outside timed execution;
6. validate per-GPU VRAM budgets with checked arithmetic;
7. unwind all partial allocations and peer state on failure.

No capability discovery or resource creation belongs in `acquire()`.

## 6. Synchronization

The initial multi-GPU version retains synchronous task execution:

```text
enqueue migration/H2D on selected task stream
  -> enqueue algorithms on that stream
  -> enqueue D2H
  -> FrameGpuAccess::complete()
  -> cudaStreamSynchronize()
  -> publish replica/cache state
```

This makes successful return the inter-stage visibility boundary even when the
next stage uses another stream or GPU. A later asynchronous graph would require
CUDA events and explicit event lifetime; that is a separate design.

## 7. Concurrency state

The cache metadata mutex protects only metadata transitions. CUDA calls occur
after the lock is released.

- Multiple readers may hold different or identical valid replicas.
- A whole cache entry cannot be evicted while any reader or replica fill is
  active.
- At most one fill may write a particular replica.
- A same-frame concurrent request that cannot reserve the desired path uses
  task fallback.
- Fill failure invalidates only the destination replica unless it was the
  initial whole-entry fill; an initial fill failure returns the entry to
  `Empty`.

## 8. Immutable payload requirement

All replicas represent byte-identical original input. Task stages receive only
const input pointers. If a stage must produce downstream GPU data, add a
separate frame-owned output plane with its own authoritative lifetime. Do not
add implicit in-place mutation to the cache input plane.

## 9. Observability

Add counters per graph copy and per GPU pair:

- local cache hits;
- initial H2D fills;
- task fallbacks;
- P2P replica fills;
- staged replica fills;
- LRU evictions;
- transferred bytes and failures.

Use these counters to choose cache capacity and decide whether per-GPU replicas
outperform repeated H2D for the target topology.

## 10. Acceptance criteria

- TaskA/TaskB/TaskC may execute on different GPUs and observe identical input.
- Scheduler selection and `DummyTask::execute(FrameCpuAtom&, StaticData&)` stay
  unchanged.
- All replica and fallback buffers are allocated before workers start.
- Cache capacity affects only performance/VRAM, never correctness.
- No active entry or replica is evicted or overwritten.
- Failed/abandoned migration publishes no destination validity.
- Unsupported topology/path fails explicitly or safely uses H2D fallback.
- Capacity zero remains a valid correctness baseline.
