# FrameSlot GPU Data Plan: Temporary One-GPU-per-NUMA Scope

**Implementation status:** Implemented for the repository's current
`start -> DummyTask -> end` graph. The implementation establishes frame-owned
GPU storage, `Upload`/`Read` access, frame-ID residency, graph-lifetime device
allocation, graph-owned `FrameCpuAtom` inputs, `StaticData`-owned FrameSlot
storage, hash-indexed task-side selection, synchronous RAII completion, and an
explicit one-GPU topology guard. Multi-stage scheduling, pool recycling, and
cross-GPU replica migration remain deferred.

## 1. Scope

The temporary implementation assumes:

```text
one NUMA graph copy -> exactly one eligible GPU
```

Its interfaces deliberately remain GPU-keyed so this can later become:

```text
one NUMA graph copy -> two eligible GPUs
```

without changing scheduler selection or task call sites. The long-term
multi-GPU extension is described in `frame_gpu_data_plan.md`.

## 2. Goal

GPU-resident frame data belongs to `FrameSlot`, not to a `DummyTask`
instance:

```text
RAM
  -> Upload/H2D on the graph GPU
  -> TaskA instance N reads
  -> same FrameSlot GPU payload
  -> TaskB instance M reads
  -> TaskC instance K reads
```

A task stage does not modify the logical frame payload. The only mutable
operation is `Upload`, which completely replaces the allocation's contents
when the slot begins representing a new logical frame. After a successful
upload, every stage uses `Read`.

## 3. Golden constraints

`graph.md` remains authoritative.

- The scheduler still selects a ready frame, a free task instance, and a free
  NUMA-local worker without consulting GPU residency.
- Workers remain NUMA-bound only.
- Each task instance remains permanently bound to one GPU.
- A frame may move between task instances and workers.
- Stages for one frame execute in graph order and do not overlap.
- GPU frame data has `FrameSlot` lifetime.
- `TaskGpuResources` continues to own the task stream, scratch, pinned
  staging, and task-private buffers.
- No `cudaMalloc()` or `cudaFree()` occurs in `execute()`,
  `FrameGpuData::acquire()`, or `FrameGpuAccess::complete()`.
- `execute()` remains synchronous.

## 4. Ownership and allocation lifetime

```text
DummyGraph
  -> owns one FrameCpuAtom per configured logical frame
  -> ReadyFrame carries only FrameCpuAtom during dispatch

StaticData
  -> one instance per graph copy
  -> owns a fixed pool of 220 FrameSlot objects
  -> binds one slot per configured logical frame
  -> owns an immutable frame-ID-to-slot-index hash
  -> owns frame state/result plumbing

FrameCpuAtom
  -> owns CPU frame bytes
  -> owns intrinsic FrameMetadata

FrameSlot
  -> owns an equal FrameMetadata copy, graph state, and results
  -> contains no FrameCpuAtom or FrameCpuAtom pointer
  -> embeds FrameGpuData

FrameGpuData
  -> owns GPU-keyed replica metadata
  -> owns the device allocation
  -> owns resident frame ID and validity state

FrameGpuAccess
  -> temporarily refers to one replica
  -> owns stream synchronization and publish-or-abort handling

TaskGpuResources
  -> owns GPU identity, stream, staging, scratch, and private buffers
```

For every bound slot:

```text
graph initialization
  -> cudaMalloc once
  -> zero or more Upload/Read accesses using the same pointer
  -> graph workers stop
  -> graph teardown
  -> cudaFree once
```

`Upload` never allocates or frees memory. It overwrites the existing device
buffer. `FrameGpuData::release()` is invoked by `StaticData` on the NUMA-pinned
teardown path, after workers have joined and before retained CUDA contexts are
released.

The checked-in graph currently owns one atom per configured frame. `StaticData`
constructs 220 slot objects, binds and allocates one for every configured frame,
and leaves the remaining entries unbound. More than 220 combined warmup/timed
frames in one graph copy is rejected. Slots are not recycled yet. The GPU API
nevertheless supports overwriting an existing allocation for a future recycled
slot.

## 5. Temporary topology contract

Initialization requires:

```text
config.gpuIds.size() == 1
```

The only GPU need not be CUDA device 0. Empty or multi-GPU lists fail before
slot allocation and before workers start. The implementation must not silently
drop extra GPUs, split a NUMA graph into GPU-specific schedulers, or defer this
error to the hot path.

## 6. Data model

The checked-in model is:

```cpp
struct FrameMetadata {
    std::uint64_t id = 0;
    std::size_t bytes = 0;
    int width = 0;
    int height = 0;
    int dtype = 0;
};

struct FrameCpuAtom {
    FrameMetadata metadata;
    std::vector<std::uint8_t> data;
};

struct FrameSlot {
    FrameMetadata metadata;
    FrameGpuData deviceData;
    // NUMA, phase/state, and result fields
};

class StaticData {
public:
    static constexpr std::size_t FRAME_SLOT_POOL_SIZE = 220;
    bool init(const StaticDataConfig& config);
    bool execute() const;
    std::size_t frameSlotCount() const;
    FrameSlot* frameSlotAt(std::size_t index);
    FrameSlot* findFrameSlot(const FrameMetadata& metadata);
    bool release();

private:
    std::unordered_map<std::uint64_t, std::size_t> frameIdToSlotIndex;
};

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
    std::uint64_t frameId() const;

private:
    std::vector<FrameGpuReplica> replicas;
    std::size_t dataBytes = 0;
    std::uint64_t residentFrameId = 0;
    bool payloadValid = false;
    bool initialized = false;
};
```

`FrameCpuAtom` remains graph-owned; `FrameSlot` is owned by the graph copy's
`StaticData`. Their metadata is copied from the same cold-path value.
`ReadyFrame` carries only the atom. `DummyTask::execute(FrameCpuAtom&,
StaticData&)` requests an average O(1) lookup by unique frame ID, after which
`StaticData` requires a complete metadata match. `FrameSlot` itself has no CPU
atom association.

`frameId` answers a concrete question: “which logical frame does this device
payload contain?” It avoids a generic mutation counter that does not match the
immutable payload contract.

`payloadValid` and `FrameGpuReplica::valid` remain separate from the ID
because frame ID 0 is legal. The ID is meaningful only while the corresponding
valid flag is true.

Frame IDs must uniquely identify logical frames during one graph lifetime. If
an integration reuses frame IDs while old slots can still exist, it must
provide a separate unique generation/epoch rather than treating a repeated ID
as the same payload.

Although there is currently one replica, lookup is by exact GPU ID rather than
using the CUDA device ID as a vector index. This keeps future multi-GPU
behavior inside `FrameGpuData`.

## 7. Access contract

Only two access modes exist:

| Mode | Existing matching payload required | Mutable pointer | Completion effect |
| --- | --- | --- | --- |
| `Upload` | No | Yes, only as the H2D destination | Publishes the requested frame ID after successful synchronization |
| `Read` | Yes | No | Validates the same resident frame ID; logical payload is unchanged |

Typical task usage:

```cpp
FrameSlot* selectedFrame = staticData.findFrameSlot(atom.metadata);
if (selectedFrame == nullptr) {
    return false;
}
FrameSlot& frame = *selectedFrame;

if (!atom.matchesMetadata(frame.metadata)) {
    return false;
}

const std::uint64_t frameId = frame.metadata.id;
const bool needsUpload = !frame.deviceData.hasData(frameId);
const FrameGpuAccessMode mode =
    needsUpload ? FrameGpuAccessMode::Upload : FrameGpuAccessMode::Read;

FrameGpuAccess access =
    frame.deviceData.acquire(mode, frameId, resources);
if (!access) {
    return false;
}

if (needsUpload) {
    enqueueH2D(access.writableData(), atom.data, resources.stream);
}
launchReadOnlyStage(access.data(), resources.stream);
return access.complete(submittedSuccessfully);
```

Rules:

- `Read` succeeds only when the global payload and selected replica are valid
  for the requested frame ID.
- `Read::writableData()` returns `nullptr`.
- Starting `Upload` invalidates metadata for the previous logical payload,
  but leaves all device pointers allocated.
- A successful `Upload` publishes its requested frame ID and marks its replica
  valid.
- A failed or abandoned `Upload` publishes no frame ID. The old payload stays
  invalid because slot reuse has already begun.
- `FrameGpuAccess::complete()` synchronizes the captured task stream before
  publishing or validating state.
- If the caller leaves scope without `complete()`, the destructor synchronizes
  already-submitted work and aborts the access.
- `FrameGpuAccess` owns neither the stream nor the allocation.

There is no mutable stage mode because downstream tasks have no requirement to
change the frame payload. If a future algorithm produces a distinct payload for
later stages, that output should be modeled as another frame-owned plane or an
explicit new payload contract, not as implicit in-place mutation.

## 8. Current data flow

### New logical frame

```text
FrameCpuAtom.data
  -> memcpy into task-owned pinned staging
  -> acquire Upload(frame.metadata.id)
  -> asynchronous H2D into the existing frame-owned allocation
  -> CEL / SDD / MI read that allocation on the same stream
  -> result D2H
  -> complete() synchronizes
  -> publish residentFrameId = frame.metadata.id
```

### Another task instance or stage

```text
same FrameSlot
  -> acquire Read(frame.metadata.id)
  -> same device pointer
  -> no repeated H2D
  -> read-only CUDA work
  -> complete() validates the unchanged frame ID
```

Task identity and worker identity do not affect the allocation or its lifetime.
Only a final stage that needs a host-visible result performs D2H; intermediate
stages do not round-trip frame payload data through RAM.

### Slot reused for another frame

```text
existing device allocation contains old frame ID
  -> acquire Upload(newFrameId)
  -> invalidate old residency metadata
  -> H2D overwrites the same device pointer
  -> complete()
  -> publish newFrameId
```

## 9. Synchronization and serialization

The temporary design retains one host synchronization per `execute()`:

```text
stage N enqueues its CUDA work
  -> FrameGpuAccess::complete()
  -> cudaStreamSynchronize(stage N stream)
  -> publish/validate frame ID
  -> execute() returns
  -> graph may dispatch stage N+1
```

This is sufficient when consecutive stages use different task streams because
the producer stream has completed before the next stage becomes eligible. An
asynchronous `execute()` design would require CUDA events and is out of scope.

`FrameGpuData` intentionally contains no scheduling mutex, atomic, or
`accessActive` flag. The graph protocol owns both protections, while
`StaticData` exposes frame-state transitions:

- one task instance is removed from `freeTasks` for the entire
  `execute()` call;
- stages of one `FrameSlot` do not overlap.

Direct concurrent access outside that graph contract is unsupported.

## 10. Initialization and teardown

Initialization on the NUMA-pinned setup thread:

1. Validate exactly one eligible GPU.
2. Validate at most 220 configured frames, unique IDs, payload sizes, and
   checked VRAM arithmetic.
3. Construct one graph-owned `FrameCpuAtom` for every configured frame.
4. `StaticData` constructs 220 slot objects and binds one per configured frame.
5. Build and reserve the immutable frame-ID-to-slot-index hash.
6. Allocate one replica for every bound slot; unbound slots allocate nothing.
7. Initialize all bound payload and replica validity flags to false.
8. Start workers only after `StaticData::init()` succeeds completely.

Partial initialization releases every successfully allocated replica.

Teardown:

1. Stop and join graph workers.
2. Clear pending graph atom dispatch records.
3. Call `StaticData::release()` on the NUMA-pinned teardown thread.
4. Destroy the StaticData slot pool, then graph-owned CPU atoms.
5. Unload task-private resources and synchronize/destroy their streams.
6. Release retained CUDA contexts only after all allocations are gone.

Cleanup remains idempotent and safe after partial initialization.

## 11. Memory model

Let:

- `S` be the number of bound slots in one graph copy, where `S <= 220`;
- `B` be the payload bytes per slot.

The temporary scope uses:

```text
frame replica VRAM = S * B
```

This memory is reserved for the entire graph lifetime. The cost is intentional:
it guarantees stable pointers and allocation-free execution.

The host-side index stores `S` entries. Lookup is average O(1), followed by a
full metadata comparison. It is built before workers start and is never
inserted into, erased from, or rehashed during execution.

## 12. Verification

The implemented CUDA tests cover:

- fixed 220-slot container capacity and duplicate frame-ID rejection;
- hash-indexed CPU-atom/slot metadata matching and mismatch rejection,
  including equal IDs with different layouts;
- malformed CPU atom size rejection;
- read rejection before the first successful upload;
- upload publication of the requested frame ID;
- frame ID 0 remaining a valid possible ID because validity is explicit;
- read-only access exposing no mutable pointer;
- read preserving the resident frame ID;
- abandoned and failed uploads publishing no new frame ID;
- upload for a new frame reusing the exact same device pointer;
- rejection of an old frame ID after replacement;
- rejection of a GPU without a replica;
- data continuity across different task instances on the same GPU;
- idempotent release and temporary topology rejection;
- graph-level task exclusivity with multiple workers.

The hot-path test contract also requires no `cudaMalloc()`, `cudaFree()`,
vector growth, or replica-table mutation during access or execution.

## 13. Future two-GPU-per-NUMA extension

The scheduler and `execute(FrameCpuAtom&, StaticData&)` API remain unchanged.
Extend only the cold-path topology and `FrameGpuData` internals:

1. Permit two eligible GPU IDs.
2. Allocate one persistent replica per slot on each GPU.
3. Discover and warm direct-P2P or staged-copy paths during initialization.
4. On `Read(frameId)`, return immediately if the local replica is valid for
   that ID.
5. Otherwise, copy from any valid replica for the same resident frame ID into
   the selected GPU's preallocated replica on the task stream.
6. Mark the copied replica valid only after successful synchronization.
7. On `Upload(newFrameId)`, invalidate all replica metadata, overwrite one
   existing allocation, and publish the new ID after completion.
8. Add checked per-GPU and graph-wide VRAM budgets plus transfer diagnostics.

Because a frame payload is immutable after upload, multiple GPU replicas may be
valid for the same frame ID. No replica becomes stale merely because another
task reads the frame.

The extension must not:

- prefer tasks based on residency;
- bind workers to GPUs;
- move stream or scratch ownership into `FrameGpuData`;
- allocate or free replicas in the hot path;
- introduce in-place task-stage mutation.

## 14. Acceptance criteria

- `FrameSlot` owns authoritative GPU frame data.
- `FrameCpuAtom` owns CPU bytes; `FrameSlot` contains no atom reference.
- `DummyGraph` owns CPU atoms; its graph-copy `StaticData` owns the fixed
  220-slot pool, binds one slot per configured frame, and owns the immutable
  frame-ID index.
- `ReadyFrame` carries only an atom; `DummyTask` selects the matching bound slot
  through `StaticData`'s average O(1) lookup without affecting scheduler choice.
- `Upload` is the only mutable access and means complete replacement for a
  logical frame.
- Later stages receive only `Read`.
- Resident state is identified by frame ID, not an abstract mutation counter.
- Device allocations remain stable until graph teardown and are reused when a
  slot receives a new frame.
- Task-instance and worker changes do not invalidate frame data.
- Scheduler selection remains unchanged and GPU-residency unaware.
- Workers remain NUMA-bound; task instances remain GPU-bound.
- `TaskGpuResources` retains streams, staging, scratch, and task-private
  buffers.
- Synchronous completion is the inter-stage dependency boundary.
- The current one-GPU limitation fails explicitly and leaves a contained
  two-GPU extension path.
