# Graph Protocol Conformance Handover

## Scope and mapping

This document compares the current implementation with the golden protocol in
`graph.md`. GPUInfra implements its one-task graph:

| Golden concept | GPUInfra implementation |
| --- | --- |
| Graph copy | one `DummyGraph` per GPU-bearing NUMA node |
| `TaskA` | `DummyTask` |
| Task instance | one reusable, GPU-bound `DummyTask` object |
| Graph thread | one persistent NUMA-pinned `DummyGraph` worker |
| Frame | one graph-owned `FrameSlot` |

CEL, SDD, and MI are internal ordered operations of `DummyTask`. They do not
map to `TaskA`, `TaskB`, and `TaskC`. The multi-task section of `graph.md` is a
generic protocol example and is not a configurable topology in this demo.

## Conformance summary

The conflicts from the pre-refactor implementation are resolved:

1. graph topology is NUMA-node scoped;
2. `DummyTask` exposes the documented lifecycle API;
3. lifecycle operations run as graph-copy-wide phases;
4. task instances are no longer owned by graph threads;
5. one scheduling assignment is one task execution on one frame;
6. frame, task, and thread availability are independent constraints.

`graph.md` was not changed as part of the implementation refactor.

## Resolved conflicts

### 1. One graph copy per NUMA node

`GpuContextManager` returns every discovered `(gpuId, numaNode)` pair.
`main.cpp` groups those records and creates one `DummyGraph` for every distinct
GPU-bearing NUMA node. Each graph owns all task instances and workers eligible
for that node.

CPU-only NUMA nodes are intentionally outside this GPU demo because there is no
local GPU to bind a `DummyTask` to.

References: `GpuContextManager::gpuLocations()`, `main.cpp`, and
`DummyGraph::initialize()`.

### 2. Golden task API and lifecycle order

`DummyTask` provides:

```text
load
registerParameters
notifyParameters
execute
unload
```

An explicit `TaskLifecycle` state rejects out-of-order calls. `DummyGraph`
invokes all task instances in this order:

```text
load every task
  -> register every task
  -> notify every task
  -> execute frames
  -> unload every task
```

The old `registerParameters() -> load()` dependency is removed.

References: `DummyTask.h`, `DummyTask.cpp`, and
`DummyGraph::initializeOnNumaNode()`.

### 3. Real parameter registration

`registerParameters()` no longer sets a Boolean marker. Every task registers
the `dummy.name` string and `dummy.blob` byte-vector schema into one graph-owned
`ParameterRegistry`. Repeated registration must agree on type. The graph sets
values, seals the registry, creates a snapshot, and notifies every instance
with that shared immutable value set.

References: `ParameterRegistry.h`, `ParameterRegistry.cpp`, and
`DummyTask::registerParameters()`.

### 4. Task instances move between graph threads

`TaskGpuResources` and the private CEL/SDD/MI objects belong to `DummyTask`.
There is no `ownerTid`, thread-owned `ThreadSlot`, or same-thread execution
check. Each task remains GPU-bound but can be called by any graph worker in its
NUMA copy. The selected worker establishes the task's Runtime API device before
using its stream and allocations.

An atomic guard ensures that movement never becomes concurrent use: one task
instance still processes at most one frame at a time.

References: `TaskGpuResources.h`, `DummyTask::execute()`, and
`GpuContextManager::makeTaskCurrent()`.

### 5. Frames are independent from tasks and threads

Each `FrameSlot` owns its input and preallocated result buffers. It owns no CUDA
stream and has no GPU affinity. `DummyGraph` owns a FIFO ready queue and a free
task pool. A worker atomically claims one object from each pool, executes
outside the scheduler lock, then releases itself and the task independently.

This directly implements:

```text
ready frame + free task instance + free NUMA-local thread -> execute
```

References: `GraphTypes.h`, `GraphStuff.h`, and `DummyGraph::workerLoop()`.

### 6. Graph rather than task owns completion

`DummyTask::execute()` writes only the supplied frame slot and returns status.
It has no sink pointer. After execution, `DummyGraph` publishes the slot's
result and transitions the slot to `Completed`, `Failed`, or `Cancelled`.

Reference: `DummyGraph::workerLoop()`.

### 7. NUMA eligibility is enforced

Setup, teardown, and every graph worker call
`GpuContextManager::pinCurrentThreadToNumaNode()`. A graph contains only GPUs
mapped to that node. Any worker in the graph may execute any of those local-GPU
tasks, but no worker may execute a task from another graph copy.

References: `GpuContextManager.cpp`, `DummyGraph::initializeOnNumaNode()`, and
`DummyGraph::workerLoop()`.

## Intentional implementation choices

- The demo implements only `start -> DummyTask -> end`; it does not expose a
  runtime-selectable multi-task topology.
- CEL, SDD, and MI all read the original frame. Their order is CUDA submission
  order, not chained data flow.
- Task and worker counts default to equal values but remain independent fields
  in `GraphConfig`.
- Every configured frame and result payload is allocated before workers start.
  This preserves an allocation-free hot path at the cost of memory proportional
  to the total frame count.
- Results intentionally contain no task, GPU, or worker attribution metadata.
- The positional CLI and successful benchmark output remain backward-compatible.

None of these choices weakens the one-task scheduling and lifecycle rules in
`graph.md`.

## Verification evidence

`gpuinfra_protocol_tests` covers:

- typed registration and immutable snapshots;
- invalid lifecycle calls and repeated cleanup;
- one concrete task executed sequentially by two simultaneously live host
  threads;
- concurrent execution rejection;
- CPU-reference validation for CEL/SDD/MI in both execution modes;
- independent task-pool and worker-pool bottlenecks;
- malformed-frame rejection;
- cancellation with exactly one failed terminal result per frame;
- conditional multi-NUMA graph creation.

The existing CTest smoke cases continue to verify the default demo, both CUDA
submission modes, valid factor boundaries, and invalid factor rejection.
