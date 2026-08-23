# FrameSlot Per-GPU Replica Plan

**Status:** Long-term design. The one-GPU-per-NUMA foundation is implemented
with graph-copy-scoped `StaticData` ownership as described in
`frame_gpu_data_plan_tmp.md`. Two-GPU replica allocation, lazy migration,
peer-path initialization, pool-capacity revision, and expanded VRAM budgeting
remain deferred.

## 1. Purpose

GPU-resident frame data must follow `FrameSlot` across task instances,
workers, and GPUs without making the scheduler GPU-residency aware:

```text
RAM
  -> Upload/H2D on TaskA's GPU
  -> TaskA reads
  -> FrameSlot GPU data
  -> TaskB reads, possibly on another GPU
  -> TaskC reads, possibly on another GPU
```

The selected strategy is one device allocation per eligible GPU for every
`FrameSlot`. Cross-GPU access lazily refreshes a stale local replica before a
task reads it.

## 2. Golden constraints

`graph.md` remains authoritative.

- The scheduler independently selects a ready frame, a free instance of the
  required task type, and a free NUMA-local worker.
- Scheduler selection does not inspect GPU residency.
- Workers bind to NUMA nodes, not GPUs.
- Every task instance remains permanently bound to one GPU.
- One frame may use different task instances, workers, and GPUs at successive
  stages.
- Stages of one frame execute in graph order and do not overlap.
- `execute()` remains synchronous for this design.
- No `cudaMalloc()` or `cudaFree()` occurs in execution or replica access.

## 3. Logical payload contract

The GPU payload is immutable for the lifetime of one logical frame.

- `Upload` completely initializes the payload from host data for a frame ID.
- Every graph task stage uses `Read`.
- A read on another GPU may create another byte-identical replica.
- A task stage never performs in-place mutation of the authoritative input.
- When a slot is recycled for a new logical frame, a new `Upload` overwrites
  its existing allocations and invalidates old residency metadata.

If a future stage produces data that another stage must consume, model it as a
separate frame-owned output plane or an explicit replacement payload. Do not
silently add in-place mutable access to the input plane.

## 4. Ownership and lifetime

```text
DummyGraph
  -> owns FrameCpuAtom collections and graph scheduler state
  -> ready records carry only FrameCpuAtom
  -> owns the eligible GPU list

StaticData
  -> one graph-copy-scoped instance
  -> owns the FrameSlot pool and frame state/results
  -> owns the frame-ID-to-slot index; immutable in the current fixed binding
  -> current temporary capacity is 220 slots

FrameCpuAtom
  -> owns CPU frame bytes and intrinsic metadata

FrameSlot
  -> owns an equal metadata copy, graph state, and results
  -> contains no FrameCpuAtom or FrameCpuAtom pointer
  -> embeds FrameGpuData for its complete lifetime

FrameGpuData
  -> owns one device allocation on every eligible GPU
  -> owns resident frame ID and per-replica validity
  -> resolves same-GPU reads and cross-GPU migration

FrameGpuAccess
  -> scoped, non-owning view of one replica
  -> owns synchronization and publish-or-abort handling

TaskGpuResources
  -> owns task GPU identity and stream
  -> owns scratch, pinned staging, and task-private buffers
```

Every replica is allocated during graph initialization and remains allocated
until graph teardown:

```text
graph initialize -> allocate replicas once
                 -> Upload/Read for any slot generations
graph teardown   -> free replicas once
```

The address is stable even when the slot begins representing another frame.
`Upload` changes contents and metadata only.

## 5. State model

Conceptual structure:

```cpp
struct FrameGpuReplica {
    int gpuId = -1;
    void* d_data = nullptr;
    std::uint64_t frameId = 0;
    bool valid = false;
};

class FrameGpuData {
public:
    bool initialize(const std::vector<int>& gpuIds, std::size_t bytes);
    FrameGpuAccess acquire(FrameGpuAccessMode mode,
                           std::uint64_t frameId,
                           const TaskGpuResources& resources);
    bool release();

    bool hasData(std::uint64_t frameId) const;

private:
    std::vector<FrameGpuReplica> replicas;
    std::size_t dataBytes = 0;
    std::uint64_t residentFrameId = 0;
    bool payloadValid = false;
    bool initialized = false;
};
```

State meanings:

- `payloadValid && residentFrameId == requestedId` means the object represents
  that logical frame.
- `replica.valid && replica.frameId == residentFrameId` means that GPU holds a
  readable copy.
- Multiple replicas may be valid for the same immutable frame.
- Frame ID 0 is valid, so explicit validity flags are required.
- GPU IDs are keys, not vector indexes; CUDA device IDs need not be dense.
- Frame IDs must uniquely identify logical frames during one graph lifetime.
  An integration that repeats visible IDs must supply a unique epoch as part of
  the residency key.

The state tracks frame identity, not a series of task-stage mutations.

## 6. Access contract

| Mode | Matching payload required | Mutable pointer | Cross-GPU behavior | Successful completion |
| --- | --- | --- | --- | --- |
| `Upload` | No | Yes, as complete H2D destination | No source copy | Publishes requested frame ID; uploaded replica becomes valid |
| `Read` | Yes | No | Refresh stale local replica | Local replica becomes valid for the unchanged frame ID |

`FrameGpuAccess` is stack-owned and does not own the stream or allocation.
`acquire()` performs lookup and may enqueue a replica transfer. It does not
publish pending state. `complete()` synchronizes the selected task stream and
updates metadata only after all submitted work succeeds.

Failure rules:

- Starting `Upload` invalidates all old replica metadata without freeing any
  allocation.
- A failed or abandoned upload leaves the logical payload invalid.
- A failed migrated read leaves the source replica valid and does not mark the
  destination valid.
- A read exposes only `const void*`; `writableData()` returns `nullptr`.
- Destruction of an incomplete access synchronizes already-submitted work and
  aborts metadata publication.

There is no storage-layer mutex, atomic, or `accessActive` flag. The graph
protocol serializes stages of one slot, `StaticData` exposes the corresponding
frame-state transitions, and the graph's free-task pool prevents simultaneous
use of one task instance. Concurrent direct access outside this protocol is
unsupported.

## 7. Data movement

### Initial upload

The first task receives `FrameCpuAtom&` and `StaticData&`, performs an average
O(1) frame-ID lookup with complete metadata validation, and requests `Upload`
on its bound GPU:

```text
FrameCpuAtom.data
  -> task-owned pinned staging
  -> H2D into selected FrameGpuReplica
  -> read-only task kernels on the same stream
  -> complete() synchronizes
  -> publish residentFrameId and replica validity
```

### Same-GPU read

If the selected GPU's replica is valid for the requested resident frame ID,
return its pointer without a transfer.

### Cross-GPU read

If the selected GPU's replica is stale:

1. Find a valid source replica for the requested resident frame ID.
2. Enqueue the initialized peer or staged transfer into the destination's
   preallocated replica.
3. Enqueue the task's read-only kernels after the copy on its task stream.
4. Synchronize once in `FrameGpuAccess::complete()`.
5. Mark the destination replica valid only after successful completion.

The resident frame ID never changes during this read. The scheduler is unaware
of both the source choice and transfer.

## 8. Transfer topology

During graph initialization:

1. Build a directed capability matrix for every eligible GPU pair.
2. Enable direct peer access where supported.
3. Select a deterministic supported fallback for every non-peer pair.
4. Preallocate all memory required by that fallback.
5. Exercise every path during initialization or warmup.
6. Reject unsupported pairs before workers start.

The hot path must not lazily create CUDA contexts, enable peers, allocate
staging, or acquire a bounded buffer that the scheduler cannot account for.
Diagnostics should distinguish same-GPU, direct-P2P, and staged copies.

## 9. Synchronization

Synchronous `execute()` is the inter-stage dependency boundary:

```text
stage N enqueues work
  -> stage N access.complete() synchronizes and validates state
  -> stage N execute() returns
  -> graph makes stage N+1 ready
  -> stage N+1 enqueues any migration and kernels
```

The source replica is complete before a later GPU copies from it. Same-stream
ordering makes destination kernels wait for their migration. No per-frame CUDA
event is required while this contract remains synchronous.

If `execute()` later becomes asynchronous, the design must add explicit CUDA
events and event lifetime management before removing the host synchronization.

## 10. Cold-path lifecycle

Initialization on the graph's NUMA-pinned setup thread:

1. Validate eligible GPUs and payload layout.
2. Initialize and warm transfer paths.
3. Compute checked per-GPU and graph-wide memory requirements.
4. Construct graph-owned `FrameCpuAtom` objects from the frame metadata.
5. Construct the graph-copy `StaticData` FrameSlot pool and bind one slot for
   each configured frame.
6. Build the frame-ID-to-slot index before workers start.
7. Allocate every bound slot replica on its target GPUs.
8. Set all payload and replica validity flags to false.
9. Start workers only after complete success; ready records carry atoms only.

Partial initialization must release every allocation already created.

Teardown:

1. Stop and join all graph workers.
2. Clear graph atom dispatch records.
3. Call `StaticData::release()` on the NUMA-pinned teardown thread.
4. Destroy the StaticData slot pool, then graph-owned CPU atoms.
5. Unload task-private CUDA resources.
6. Release retained contexts after frame and task allocations are gone.

No successful frame or stage completion frees a replica.

## 11. VRAM budget

Let:

- `G` be eligible GPUs in one NUMA graph copy;
- `S` be bound `FrameSlot` objects in that copy;
- `B` be bytes per GPU payload.

Preallocated replica memory is:

```text
frame replica VRAM = S * G * B
```

The graph creates frame counts per GPU. If `F` is warmup plus timed frames per
GPU:

```text
S = F * G
frame replica VRAM = F * G^2 * B
```

Example for `F = 220` and `B = 1 MiB`:

| GPUs in graph copy | Replica VRAM |
| ---: | ---: |
| 1 | 220 MiB |
| 2 | 880 MiB |
| 4 | 3,520 MiB |

This excludes task-private buffers, CUDA overhead, and additional frame planes.
Initialization must use checked arithmetic, a configured limit, and a safety
margin. The current one-GPU implementation uses a temporary fixed capacity of
220 slots per graph copy. Enabling multiple GPUs must explicitly revise or
configure that capacity when `F * G` can exceed 220.

## 12. Source responsibilities

### `FrameCpuAtom.h/.cpp` and `FrameMetadata.h`

- Own CPU frame bytes separately from `FrameSlot`.
- Store ID, byte count, dimensions, and dtype.
- Provide exact metadata comparison for graph/task boundary validation.

### `FrameGpuAccess.h/.cpp`

- Define `Upload` and `Read`.
- Hold the requested frame ID, target GPU/replica, and non-owning stream.
- Synchronize and complete-or-abort without owning allocations.

### `FrameGpuData.h/.cpp`

- Own fixed replica allocations for graph lifetime.
- Track resident frame ID and explicit replica validity.
- Implement lazy read migration internally.

### `FrameSlot.h/.cpp`

- Store a metadata copy but no CPU atom or CPU data vector.
- Embed `FrameGpuData`.
- Remain non-copyable while owning CUDA allocations.

### `StaticData.h/.cpp`

- Own the graph-copy FrameSlot pool and every bound slot's lifecycle.
- Build an immutable `frame ID -> slot index` hash during `init()`.
- Return a non-owning slot pointer after average O(1) ID lookup and complete
  metadata validation.
- Expose frame state, cancellation, and result plumbing without participating
  in scheduler selection.
- Allocate replicas during `init()` and release them after workers stop.

### `DummyGraph.h/.cpp`

- Own one atom per logical frame; `ReadyFrame` carries only the atom.
- Invoke `StaticData` lifecycle/state/result plumbing around task execution.
- Keep slot ownership, replica budgeting, and replica release in `StaticData`.
- Keep scheduler selection GPU-residency unaware.

### `DummyTask.cpp`

- Make its bound GPU current.
- Accept `execute(FrameCpuAtom& atom, StaticData& staticData)`.
- Request the bound slot through `StaticData::findFrameSlot(atom.metadata)`.
- Reject a missing ID or complete-metadata mismatch before CUDA work.
- Request access with `frame.metadata.id`, `resources.gpuId`, and
  `resources.stream`.
- Use `atom.data` as the only initial H2D source.
- Upload only when the requested frame ID is not resident.
- Pass only the const acquired view to algorithms.
- Complete the access before returning.

### `TaskGpuResources.h`

- Retain stream, GPU/NUMA identity, pinned staging, scratch, and private
  buffers.
- Never own the authoritative cross-stage frame payload.

### Algorithms

- Consume a const frame device view.
- Store later-stage outputs in explicit frame-owned output planes.
- Keep temporary algorithm buffers task-private.

### GPU topology component

- Discover, initialize, warm, and report directed transfer paths.
- Keep topology setup out of scheduler selection and timed execution.

## 13. Current source boundary

The checked-in repository currently contains:

```text
start -> DummyTask -> end
```

CEL, SDD, and MI are private operations inside one `DummyTask`; they are not
independent graph stages. `FrameGpuData` proves ownership and cross-instance
continuity, but adding `TaskA -> TaskB -> TaskC` graph topology is a separate
source prerequisite. That prerequisite still must not add GPU-aware scheduling.

## 14. Failure semantics

- Allocation, topology, or transfer-path initialization failure prevents graph
  startup and unwinds partial resources.
- Invalid GPU ID, wrong NUMA graph, missing requested frame ID, copy submission
  failure, kernel failure, or synchronization failure fails execution.
- A failed upload never publishes its requested frame ID.
- A failed migrated read never marks its destination replica valid.
- Existing fail-fast cancellation delivers every slot exactly once.
- Frame cleanup is idempotent after success, failure, or partial initialization.

## 15. Verification plan

State tests:

- Sparse frame IDs resolve through the hash index.
- A matching ID with different layout metadata is rejected.
- Initial read fails.
- Upload publishes the requested frame ID without allocating.
- Read returns a const view and preserves the frame ID.
- Same-GPU reads retain the same pointer.
- Upload for a new frame ID reuses every existing replica allocation.
- Failed and abandoned uploads publish no payload.
- A stale local replica is copied and marked valid only after completion.
- Read-only copies allow multiple replicas to remain valid.
- Non-contiguous GPU IDs resolve correctly.
- Partial and repeated cleanup are safe.

CUDA integration tests:

- Atom/slot metadata mismatch and malformed atom data are rejected.
- TaskA uploads on GPU 0 and TaskB verifies on GPU 1.
- TaskC can read the same immutable payload again on GPU 0 without replacement.
- Same-GPU transitions perform no cross-device copy.
- Different task instances share only frame-owned data.
- Direct-P2P and fallback paths run when hardware permits.
- No hot-path allocation/free call occurs.
- Cancellation releases all replicas only during teardown.

Performance checks:

- Count same-GPU, direct-P2P, and staged reads.
- Measure transfer bytes/time separately from task computation.
- Confirm initialization and path warmup are excluded from timed execution.

## 16. Implementation phases

### Phase 1: single-GPU ownership — complete

- Separate graph-owned CPU atoms from `StaticData`-owned GPU slots.
- Fixed 220-slot container with one bound slot per configured frame.
- Atom-only ready dispatch plus hash-indexed task-side metadata selection.
- Frame-owned persistent allocation.
- `Upload`/`Read` RAII state machine.
- Frame-ID residency and explicit validity.
- Task/algorithm integration without scheduler changes.

### Phase 2: multi-GPU read replicas

- Allocate every graph-local replica during initialization.
- Initialize and warm peer/fallback paths.
- Add lazy migration for stale `Read`.
- Add two-GPU state, transfer, failure, and VRAM tests.

### Phase 3: multi-stage graph integration

- Connect each task type to the same frame-owned input or explicit output plane.
- Preserve graph ordering and synchronous completion.
- Verify movement across task instances, workers, and GPUs.

### Phase 4: measurement and decision gate

- Measure VRAM and transfer costs on target hardware.
- Keep preallocated replicas as the correctness baseline.
- Consider pooling or remote peer reads only if measurements justify the added
  lifetime and bandwidth complexity.

## 17. Acceptance criteria

- GPU frame data survives task-instance, worker, and GPU changes.
- CPU data is owned by `FrameCpuAtom`; `FrameSlot` owns no atom reference.
- The graph passes atom and graph-copy `StaticData` to each task execution;
  the task uses the frame-ID index and requires a complete metadata match.
- `Upload` initializes or replaces a logical frame; graph stages are read-only.
- Residency uses a frame ID plus explicit validity, not a mutation counter.
- Every device pointer remains allocated until graph teardown.
- Reusing a slot changes payload metadata/content without allocation.
- Scheduler selection remains unchanged and residency unaware.
- Workers remain NUMA-bound; task instances remain GPU-bound.
- `TaskGpuResources` retains only execution and private resources.
- Cross-GPU reads are correct with no hot-path allocation.
- Synchronous completion provides the documented inter-stage boundary.
- Initialization fails cleanly for unsupported topology or memory requirements.
