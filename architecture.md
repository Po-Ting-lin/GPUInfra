# GPUInfra architecture

GPUInfra implements the one-task form of the golden protocol in `graph.md`:

```text
start -> DummyTask -> end
```

CEL, SDD, and MI are private operations inside `DummyTask`. They are not
independent graph tasks. One `DummyGraph` is created for each NUMA node that has
at least one discovered CUDA GPU.

## 1. Runtime topology

```text
process
  |
  +-- GpuContextManager
  |     +-- GpuContext(gpu=0, numa=0)
  |     +-- GpuContext(gpu=1, numa=0)
  |     `-- GpuContext(gpu=2, numa=1)
  |
  +-- DummyGraph(numa=0)
  |     +-- NUMA-local graph-thread pool
  |     +-- GPU-bound DummyTask pool for GPUs 0 and 1
  |     `-- warmup/timed FrameSlot collections
  |
  `-- DummyGraph(numa=1)
        +-- NUMA-local graph-thread pool
        +-- GPU-bound DummyTask pool for GPU 2
        `-- warmup/timed FrameSlot collections
```

The scheduler has three independent resources:

1. a ready frame slot;
2. a free task instance;
3. a free graph thread.

The selected objects are associated only while one `DummyTask::execute()` call
runs. A task instance can process many frames and can move between all graph
threads in its NUMA copy.

## 2. Ownership

| Component | Ownership and lifetime |
| --- | --- |
| `GpuContextManager` | Process-wide GPU discovery, NUMA mapping, task registration, and primary-context lifetime |
| `GpuContext` | One GPU's identity, NUMA node, retained `CUcontext`, and active task-resource table |
| `DummyGraph` | One NUMA graph copy, workers, task pool, frame collections, queues, lifecycle barriers, and parameters |
| `FrameSlot` | One logical frame's identity, phase/state, input vector, and preallocated results |
| `DummyTask` | One GPU-bound reusable CUDA lane and the internal algorithm objects |
| `TaskGpuResources` | Task stream, pinned/device input, optional scratch, and GPU/NUMA identity |
| CEL/SDD/MI | Algorithm parameters, geometry, device output, and pinned D2H staging output |
| Graph thread | NUMA affinity and a temporary execution call; no persistent CUDA lane |

The essential separation is:

```text
frame data belongs to FrameSlot
CUDA execution resources belong to DummyTask
host execution belongs temporarily to a graph thread
```

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
because concurrent calls on the same task are prohibited.

All workers in one graph copy are eligible for all tasks whose GPUs belong to
that graph's NUMA node. GPUs on another node are never eligible.

## 4. Task lifecycle

`DummyTask` exposes the exact golden API:

```text
load
  -> registerParameters
  -> notifyParameters
  -> execute for each assigned frame
  -> unload
```

Its explicit lifecycle state rejects calls in the wrong order. An atomic
execution guard additionally rejects two simultaneous `execute()` calls on one
instance even if a caller bypasses the graph scheduler.

### `load()`

`load()` registers task resources with the selected `GpuContext`, then creates:

- one nonblocking stream;
- one pinned input buffer;
- one device input buffer;
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

After workers stop, every task synchronizes its stream, closes the algorithms,
frees scratch/input/stream resources, and unregisters from its `GpuContext`.
Repeated unload is harmless; unload during execution is rejected.

## 5. Graph-copy cold path

`DummyGraph::initialize()` starts a temporary NUMA-pinned setup thread and
performs copy-wide phases rather than complete setup per worker:

```text
create all DummyTask objects
  -> load all task instances
  -> register all task instances
  -> seal shared parameter values
  -> notify all task instances
  -> allocate and first-touch all FrameSlots
  -> start all NUMA-local graph workers
  -> wait until every worker reports successful affinity
```

Any failure sets the shared process cancellation flag and unloads all task
instances that reached any partial lifecycle state.

Task count and graph-thread count are independent configuration values. The
demo defaults to four tasks per GPU and the same aggregate number of workers in
each graph copy, but the scheduler supports unequal pools.

## 6. Frame-slot lifetime

Every warmup and timed `FrameSlot` is allocated before workers start. A slot
contains:

- an owning input byte vector;
- its NUMA node and warmup/timed phase;
- a state: `Prepared`, `Ready`, `Executing`, `Completed`, `Failed`, or
  `Cancelled`;
- a `JobResult` with pre-sized CEL, SDD, and MI output vectors.

Slots have no GPU affinity. Whichever local-GPU task becomes free first may
process the first ready slot. This permits dynamic balancing among GPUs on one
NUMA node.

Preallocating every slot keeps `execute()` allocation-free. It also makes host
memory proportional to total configured frames, which is an intentional demo
tradeoff.

## 7. Scheduler

Graph workers self-dispatch; there is no dedicated dispatcher thread.

```text
worker waits
  -> lock scheduler
  -> claim FIFO FrameSlot and one free DummyTask atomically
  -> mark frame Executing and task unavailable
  -> unlock scheduler
  -> wait for the phase gate
  -> task.execute(frame)
  -> graph publishes frame.result
  -> lock scheduler
  -> return task to free pool
  -> complete phase when every slot is terminal
```

The scheduler lock is never held during CUDA work. A free task and ready frame
without a free worker cannot run; a free worker and ready frame without a free
task also cannot run.

Warmup and timed slots use separate collections. Main starts every graph copy
with the same `PhaseGate`; timing begins immediately before the timed gate is
released and ends after all graph copies report completion.

## 8. CUDA hot path

Each execution performs:

```text
FrameSlot.input
  -> task h_in
  -> task d_in
  -> CEL / SDD / MI submissions on the task stream
  -> algorithm pinned staging outputs
  -> one cudaStreamSynchronize
  -> FrameSlot.result.outputs
```

All algorithms read `d_in`. Their names express required submission order, not
data chaining.

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
- exactly one host-side stream synchronization per frame;
- no concurrent use of a task's mutable resources.

## 9. Failure semantics

The graph uses fail-fast process-wide cancellation. If a task lifecycle or
execution operation fails:

1. the shared cancellation flag is set;
2. queued slots become `Cancelled` with failed results;
3. in-flight slots finish as `Failed` or `Cancelled`;
4. each slot is delivered exactly once;
5. all workers wake and join;
6. every task unloads;
7. the executable returns nonzero without performance output.

Graph teardown is valid after success, cancellation, partial initialization,
or an explicit repeated shutdown.

## 10. Verification

`gpuinfra_protocol_tests` verifies on a real GPU:

- typed parameter schemas, values, sealing, and snapshots;
- lifecycle rejection and idempotent cleanup;
- one task moving sequentially across two live NUMA-local threads;
- rejection of simultaneous calls on one task;
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
