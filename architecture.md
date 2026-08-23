# GPUInfra architecture

GPUInfra implements the one-task form of the golden protocol in `graph.md`:

```text
start -> DummyTask -> end
```

CEL, SDD, and MI are private operations inside `DummyTask`. They are not
independent graph tasks. One `DummyGraph` is created for each NUMA node that has
at least one discovered CUDA GPU.

The temporary `FrameGpuData` implementation accepts exactly one GPU in each
NUMA graph copy. It keeps a GPU-keyed replica table so a future two-GPU extension
can add migration without changing task APIs or scheduler selection.

The corresponding visual views are `gpuinfra_class_diagram.html` for ownership
and `gpuinfra_resource_plot.html` for topology and frame-device data flow.
Major types use matching source files: `DummyGraph.*`, `StaticData.*`,
`FrameCpuAtom.*`, `FrameSlot.*`, `FrameGpuAccess.*`, and `FrameGpuData.*`. Header-only
`FrameMetadata.h` and `GraphTypes.h` contain shared value types and enums.

## 1. Runtime topology

```text
process
  |
  +-- GpuContextManager
  |     +-- GpuContext(gpu=0, numa=0)
  |     `-- GpuContext(gpu=1, numa=1)
  |
  +-- DummyGraph(numa=0)
  |     +-- NUMA-local graph-thread pool
  |     +-- GPU-bound DummyTask pool for GPU 0
  |     +-- warmup/timed FrameCpuAtom collections
  |     `-- StaticData -> fixed 220-slot FrameSlot pool
  |
  `-- DummyGraph(numa=1)
        +-- NUMA-local graph-thread pool
        +-- GPU-bound DummyTask pool for GPU 1
        +-- warmup/timed FrameCpuAtom collections
        `-- StaticData -> fixed 220-slot FrameSlot pool
```

The scheduler has three independent resources:

1. a ready graph-owned `FrameCpuAtom`;
2. a free task instance;
3. a free graph thread.

The selected scheduler objects are associated only while one
`DummyTask::execute()` call runs. The task selects the atom's unique
metadata-matching slot from `StaticData` through its immutable frame-ID hash
index; this lookup does not participate in scheduler selection. A task instance
can process many frames and can move between all graph threads in its NUMA copy.

## 2. Ownership

| Component | Ownership and lifetime |
| --- | --- |
| `GpuContextManager` | Process-wide GPU discovery, NUMA mapping, task registration, and primary-context lifetime |
| `GpuContext` | One GPU's identity, NUMA node, retained `CUcontext`, and active task-resource table |
| `DummyGraph` | One NUMA graph copy, workers, task pool, CPU-atom collections, `StaticData`, queues, lifecycle barriers, and parameters |
| `StaticData` | Graph-copy-scoped fixed pool of 220 FrameSlot objects, immutable frame-ID index, bound-frame state/results, cold allocation, and teardown |
| `FrameCpuAtom` | CPU frame bytes and intrinsic metadata; allocated and first-touched on the graph's NUMA setup thread |
| `FrameSlot` | Copied frame metadata, phase/state, frame-owned GPU data, and preallocated results; no CPU atom reference |
| `FrameGpuData` | One preallocated device replica plus resident-frame-ID and validity state |
| `FrameGpuAccess` | Scoped non-owning replica view; synchronizes and commits or aborts one access |
| `DummyTask` | One GPU-bound reusable CUDA lane and the internal algorithm objects |
| `TaskGpuResources` | Task stream, pinned input staging, optional scratch, and GPU/NUMA identity |
| CEL/SDD/MI | Algorithm parameters, geometry, device output, and pinned D2H staging output |
| Graph thread | NUMA affinity and a temporary execution call; no persistent CUDA lane |

The essential separation is:

```text
CPU frame data belongs to FrameCpuAtom
GPU frame data belongs to FrameSlot
FrameSlot pool ownership belongs to StaticData
CUDA execution resources belong to DummyTask
host execution belongs temporarily to a graph thread
```

`DummyGraph` owns one atom per logical frame. Its `StaticData` member owns a
fixed 220-object slot pool and binds one slot for each configured frame.
`ReadyFrame` carries only a `FrameCpuAtom*`. `DummyTask::execute(atom,
staticData)` performs an average O(1) lookup by unique frame ID, requires a
complete metadata match, and uses that non-owning slot pointer only during the
synchronous call.

`FrameGpuData` is a by-value member of `FrameSlot`, so composition preserves the
same lifetime. The extra type isolates CUDA allocation, replica frame-ID state,
validity, and access validation from the general frame record; it is not a
separate owner of the logical frame. `FrameGpuAccess` owns no allocation; it is
an RAII lease returned by `FrameGpuData::acquire()`.

## 3. GPU and NUMA rules

`GpuContextManager::init()` discovers all GPUs and records their PCI NUMA node.
It does not pin the startup thread. `main.cpp` groups the returned
`GpuLocation` records and creates one graph per distinct GPU-bearing node.

Setup, teardown, and graph workers run on threads pinned to their graph's NUMA
node. This ensures that frame first touch, pinned-host allocation, and task
execution use CPUs from the correct node.

Each `DummyTask` is permanently bound to one GPU so its CUDA allocations remain
stable. It has no permanent host-thread owner. On every assignment, the worker
calls `GpuContextManager::makeTaskCurrent()`, which selects the task's GPU with
`cudaSetDevice()`. CUDA streams and allocations can then be used by that worker
because `DummyGraph` keeps the selected task out of its free pool until the
call finishes.

All workers in one graph copy are eligible for every task bound to its sole
GPU. GPUs on another node are never eligible. A graph with zero or multiple
eligible GPUs fails initialization before workers start in the temporary scope.

## 4. Task lifecycle

`DummyTask` exposes the exact golden API:

```text
load
  -> registerParameters
  -> notifyParameters
  -> execute for each assigned frame
  -> unload
```

Its explicit lifecycle state rejects calls in the wrong order. It does not own
scheduling concurrency: callers must not invoke `execute()` or `unload()`
concurrently. `DummyGraph` enforces that contract through exclusive checkout
from its free-task pool and by joining workers before teardown.

### `load()`

`load()` registers task resources with the selected `GpuContext`, then creates:

- one nonblocking stream;
- one pinned input buffer;
- private CEL, SDD, and MI objects;
- one device and pinned-host output pair per algorithm;
- optional device scratch sized to the largest internal requirement.

The task synchronizes initialization work before becoming `Loaded`.

### Parameter registration and notification

Every task registers the same schema in one graph-owned `ParameterRegistry`:

| Name | Type |
| --- | --- |
| `dummy.name` | string |
| `dummy.blob` | byte vector |

Repeated registration succeeds only when the type agrees. The graph writes one
run value set, seals it, creates an immutable snapshot, and notifies every task
instance with that snapshot.

### `unload()`

After workers stop, `StaticData` releases frame-owned device data on the
NUMA-pinned teardown thread. Every task then synchronizes its stream, closes the
algorithms, frees scratch/pinned-staging/stream resources, and unregisters from
its `GpuContext`. Repeated unload is harmless. The graph joins every worker
before unloading tasks; direct concurrent `execute()` and `unload()` calls are
outside the `DummyTask` contract.

## 5. Graph-copy cold path

`DummyGraph::initialize()` starts a temporary NUMA-pinned setup thread and
performs copy-wide phases rather than complete setup per worker:

```text
create all DummyTask objects
  -> load all task instances
  -> register all task instances
  -> seal shared parameter values
  -> notify all task instances
  -> allocate and first-touch all FrameCpuAtoms
  -> StaticData constructs its fixed 220-slot pool
  -> bind and allocate one FrameSlot per configured frame
  -> build an immutable frame-ID-to-slot index
  -> start all NUMA-local graph workers
  -> wait until every worker reports successful affinity
```

Any failure sets the shared process cancellation flag and unloads all task
instances that reached any partial lifecycle state.

Task count and graph-thread count are independent configuration values. The
demo defaults to four tasks per GPU and the same aggregate number of workers in
each graph copy, but the scheduler supports unequal pools.

## 6. Frame-slot lifetime

Every graph copy gets a fixed pool of 220 `FrameSlot` objects. Every configured
warmup and timed logical frame gets one `FrameCpuAtom` and binds one pool slot
before workers start. Unused slots remain unbound and own no GPU/result
allocation. An atom contains:

- an owning CPU input byte vector;
- intrinsic metadata: frame ID, byte count, width, height, and dtype.

A slot contains:

- an equal metadata copy, but no atom or atom pointer;
- one frame-owned device replica on the graph copy's GPU;
- a resident frame ID, explicit validity state, and RAII access interface;
- its NUMA node and warmup/timed phase;
- a state: `Prepared`, `Ready`, `Executing`, `Completed`, `Failed`, or
  `Cancelled`;
- a `JobResult` with pre-sized CEL, SDD, and MI output vectors.

Slots have no task or worker affinity. Whichever task instance becomes free
first may process the first ready slot. The sole device replica does not
participate in scheduler selection. The graph state machine permits only one
stage to execute a given slot at a time, so `FrameGpuData` does not duplicate
that policy with an `accessActive` flag or lock.

Preallocating every atom and bound slot keeps `execute()` allocation-free.
Allocated host and device memory remains proportional to total configured
frames, while the slot-object capacity is always 220. More than 220 combined
warmup and timed frames in one graph copy is rejected during initialization.

The device pointer remains allocated from graph initialization through graph
teardown. `Upload` is used only when the slot receives a new logical frame; it
overwrites that allocation and publishes the new frame ID after stream
synchronization. The payload is immutable for the rest of that frame's stages,
which use `Read`. Frame ID 0 is legal, so validity is stored separately rather
than encoded as a sentinel ID.

## 7. Scheduler

Graph workers self-dispatch; there is no dedicated dispatcher thread.

```text
worker waits
  -> lock scheduler
  -> claim FIFO ReadyFrame(atom) and one free DummyTask atomically
  -> mark the StaticData frame Executing and task unavailable
  -> unlock scheduler
  -> wait for the phase gate
  -> StaticData::execute()
  -> task.execute(atom, staticData)
  -> task performs indexed FrameSlot lookup and validates full metadata
  -> graph obtains and publishes the result through StaticData
  -> lock scheduler
  -> return task to free pool
  -> complete phase when every slot is terminal
```

The scheduler lock is never held during CUDA work. A free task and ready frame
without a free worker cannot run; a free worker and ready frame without a free
task also cannot run.

Warmup and timed CPU atoms use separate collections; their slots are bound in
the same `StaticData` pool and retain a phase field. Main starts every graph
copy with the same `PhaseGate`; timing begins immediately before the timed gate
is released and ends after all graph copies report completion.

## 8. CUDA hot path

Each execution performs:

```text
FrameCpuAtom.data
  -> DummyTask uses StaticData's frame-ID hash and validates complete metadata
  -> task h_in
  -> FrameSlot.deviceData on first H2D
  -> CEL / SDD / MI submissions reading frame-owned device data
  -> algorithm pinned staging outputs
  -> FrameGpuAccess::complete()
  -> one cudaStreamSynchronize and frame-ID publish/validate
  -> FrameSlot.result.outputs
```

All algorithms read the acquired frame device pointer. Their names express
required submission order, not data chaining.

Batched mode:

```text
H2D
  -> CEL kernel -> SDD kernel -> MI kernel
  -> CEL D2H -> SDD D2H -> MI D2H
  -> sync and collect
```

Interleaved mode:

```text
H2D
  -> CEL kernel -> CEL D2H
  -> SDD kernel -> SDD D2H
  -> MI kernel  -> MI D2H
  -> sync and collect
```

Hot-path invariants are:

- no CUDA or host allocation;
- no vector resize or ownership transfer;
- no manager mutex;
- no `cudaDeviceSynchronize()`;
- exactly one host-side stream synchronization, owned by
  `FrameGpuAccess::complete()`, per `execute()` call;
- no concurrent use of a task's mutable resources.

## 9. Failure semantics

The graph uses fail-fast process-wide cancellation. If a task lifecycle or
execution operation fails:

1. the shared cancellation flag is set;
2. queued frames become `Cancelled` through `StaticData` with failed results;
3. in-flight slots finish as `Failed` or `Cancelled` through `StaticData`;
4. each configured frame result is delivered exactly once;
5. all workers wake and join;
6. every task unloads;
7. the executable returns nonzero without performance output.

Graph teardown is valid after success, cancellation, partial initialization,
or an explicit repeated shutdown.

## 10. Verification

`gpuinfra_protocol_tests` verifies on a real GPU:

- typed parameter schemas, values, sealing, and snapshots;
- fixed 220-slot capacity, duplicate-frame rejection, indexed task-side
  CPU-atom/slot metadata matching, and malformed-atom rejection;
- frame GPU `Upload`/`Read` modes, frame-ID publication, explicit/RAII aborts,
  and idempotent cleanup;
- frame device data surviving a change of task instance on the same GPU;
- lifecycle rejection and idempotent cleanup;
- one task moving sequentially across two live NUMA-local threads;
- graph-level exclusive checkout with one task and multiple workers;
- full CPU-reference matrices in Batched and Interleaved modes;
- malformed-frame rejection;
- one-task/two-worker and two-task/one-worker bottlenecks;
- fail-fast execution cancellation and terminal delivery;
- multi-NUMA graph creation when suitable hardware is present.

The existing executable tests also cover default execution, both models,
factor boundaries, and invalid factors.

Use:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```
