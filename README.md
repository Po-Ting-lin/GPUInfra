# GPUInfra CUDA graph demo

This repository is a runnable CUDA model of the one-task graph described by
`graph.md`:

```text
start -> DummyTask -> end
```

`DummyTask` contains three ordered synthetic operations—CEL, SDD, and MI—but
they are not separate graph tasks. The graph scheduler independently matches a
ready CPU atom, a free `DummyTask` instance, and a free NUMA-local graph thread
for each `execute()` call. The task then finds the atom's matching `FrameSlot`
inside graph-copy-scoped `StaticData`; slot residency does not affect scheduling.

The implementation demonstrates:

- one `DummyGraph` copy per GPU-bearing NUMA node;
- a temporary topology contract of exactly one GPU per NUMA graph copy;
- independently sized task-instance and graph-thread pools;
- four GPU-bound `DummyTask` instances per GPU by default;
- task instances that move between graph threads on the same NUMA node;
- task-owned CUDA streams, pinned staging, scratch, and CEL/SDD/MI resources;
- graph-owned, fully preallocated `FrameCpuAtom` input objects;
- a `StaticData`-owned fixed pool of 220 `FrameSlot` objects;
- one bound slot with metadata, device input, state, and results for every
  configured logical frame;
- graph-wide lifecycle barriers in the golden order;
- fail-fast cancellation across every NUMA graph copy;
- one initial H2D transfer per frame payload and one
  `cudaStreamSynchronize()` per `execute()` call.

The synthetic algorithms launch real CUDA matrix-multiplication kernels over a
shared square byte input. For size factor `F`, the input is `(8F)x(8F)`, CEL
operates on `(2F)x(2F)`, and SDD and MI operate on `(3F)x(3F)`. Each result is a
row-major `uint32_t` matrix. All three operations consume the original input;
`CEL -> SDD -> MI` specifies submission order, not an output-to-input chain.

## Design references

- `graph.md` is the golden scheduler and task-lifecycle protocol.
- `architecture.md` documents the implemented ownership and execution model.
- `gpuinfra_class_diagram.html` visualizes class ownership and frame GPU access.
- `gpuinfra_resource_plot.html` visualizes NUMA topology and first/repeated
  frame-device access.
- `frame_gpu_data_plan_tmp.md` records the implemented one-GPU-per-NUMA scope.
- `frame_gpu_data_plan.md` remains the deferred multi-GPU replica/migration
  design.
- `open_issues.md` records the required real-framework integration hooks.

## Requirements

- Linux with NUMA topology under `/sys/devices/system/node`
- CMake 3.24 or newer
- A C++17 compiler
- NVIDIA CUDA Toolkit and compatible driver
- At least one CUDA GPU with a discoverable PCI NUMA node
- Exactly one discovered CUDA GPU per NUMA node for the temporary
  implementation

`libnuma-dev` is not required. CPU affinity uses pthreads and Linux NUMA sysfs.

The default CUDA architecture is `sm_86`. Override it for another GPU:

```bash
CUDA_ARCHITECTURES=89 ./build.sh
```

or:

```bash
./build.sh -DCMAKE_CUDA_ARCHITECTURES=89
```

## Build, run, and test

Build a Release configuration:

```bash
./build.sh
```

Equivalent commands are:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

Run the demo:

```bash
./build/gpuinfra_demo
./build/gpuinfra_demo 200 20 batched 128
./build/gpuinfra_demo 200 20 interleaved 128
```

The positional arguments remain:

```text
gpuinfra_demo [timed_frames_per_gpu] [warmup_frames_per_gpu]
              [batched|interleaved] [size_factor]
```

Defaults are 200 timed frames, 20 warmup frames, Batched execution, and size
factor 128. The factor must be a multiple of 16 from 16 through 256. The
temporary `StaticData` capacity limits each graph copy to at most 220 combined
warmup and timed frames.

Run the complete test suite:

```bash
ctest --test-dir build --output-on-failure
```

`gpuinfra_protocol_tests` uses a real CUDA GPU to verify lifecycle ordering,
typed parameter registration, `StaticData` capacity and unique-frame
validation, hash-indexed CPU-atom/slot matching, frame-ID residency and RAII access
behavior, device data surviving a task-instance change, task movement between
two live host threads, temporary topology rejection, graph-level exclusive task
checkout, CPU-reference results for both execution models, independent
task/thread bottlenecks, malformed input, cancellation, and cleanup. Multi-NUMA
coverage runs conditionally when each participating NUMA graph copy has exactly
one GPU.

Run the existing size-factor benchmark:

```bash
python3 scripts/benchmark_model_factor.py
```

The successful output contract is unchanged:

```text
[GPUInfra] shutdown complete
execution_model=batched
size_factor=128 input=1024x1024 cel=256x256 sdd=384x384 mi=384x384
warmup 20 frames/GPU, timed 200 frames/GPU (200 total) in ... ms
throughput=... frames/s, average=... ms/frame
delivered 220 frames across 1 CUDA GPU(s), failures=0
```

Performance lines are suppressed when initialization, execution, cancellation,
or teardown fails.

## Graph and task lifecycle

### 1. GPU discovery

`GpuContextManager::init()`:

1. initializes the CUDA Driver API;
2. discovers every Runtime API device;
3. maps each GPU to its PCI NUMA node;
4. primes each device's Runtime API primary context;
5. retains one `CUcontext` reference per GPU.

The startup thread is not NUMA-pinned. The manager groups no graph work itself;
`main.cpp` creates one `DummyGraph` for every distinct GPU-bearing NUMA node.
The temporary implementation rejects a graph copy containing more than one GPU
before allocating tasks or starting workers. The GPU-keyed frame-replica interface is retained so
two GPUs per NUMA can later be enabled without changing scheduler selection or
task call sites.

### 2. Graph-copy cold path

Each graph runs setup on a temporary NUMA-pinned setup thread. It creates the
configured number of GPU-bound `DummyTask` instances and then performs
copy-wide phases:

```text
load every DummyTask
  -> registerParameters on every DummyTask
  -> seal one shared parameter snapshot
  -> notifyParameters on every DummyTask
  -> preallocate one FrameCpuAtom per logical frame
  -> StaticData creates its fixed 220-slot pool
  -> bind and allocate one FrameSlot per configured logical frame
  -> build an immutable frame-ID-to-slot index
  -> start and pin graph workers
```

`ParameterRegistry` registers `dummy.name` as a string and `dummy.blob` as a
byte vector. Re-registration must agree on type. Values become immutable when
the graph seals the snapshot, and every task instance receives the same values.

### 3. Scheduling

Workers are persistent and pinned to their graph copy's NUMA node. Under the
graph scheduler lock, a free worker atomically claims the first ready
`FrameCpuAtom` and one free task. The lock is released before CUDA work begins.

```text
ready FrameCpuAtom + free DummyTask + free graph thread
                            |
                            v
                  StaticData::execute()
                            |
                            v
             DummyTask::execute(atom, staticData)
                            |
                            v
        task performs indexed FrameSlot lookup and validates full metadata
```

A task remains bound to one GPU because its reusable allocations belong to
that device. It is not bound to a host thread. Before each execution, the
selected worker calls `cudaSetDevice(taskGpu)` and then uses the task's stream
and private buffers. Any worker in the same NUMA graph copy may run any task
bound to the copy's sole GPU.

Exclusive task-instance use belongs to `DummyGraph`: the scheduler removes a
task from `freeTasks` while holding `schedulerLock` and returns it only after
`execute()` finishes. `DummyTask` has no independent concurrent-execution guard
and is not a thread-safe scheduling boundary.

Warmup completes on every graph copy before timed frames are released through a
shared `PhaseGate`.

### 4. Frame execution

`DummyTask::execute(FrameCpuAtom&, StaticData&)` is allocation-free:

```text
FrameCpuAtom.data
  -> average O(1) StaticData lookup by frame ID plus complete metadata validation
  -> memcpy to task-owned pinned input
  -> Upload performs asynchronous H2D into FrameSlot.deviceData for a new frame ID
  -> CEL, SDD, and MI CUDA work reading FrameSlot.deviceData
  -> D2H into each algorithm's task-owned pinned staging buffer
  -> FrameGpuAccess::complete() synchronizes and publishes or validates the frame ID
  -> memcpy into the FrameSlot's preallocated result buffers
  -> return task and thread independently to their pools
  -> DummyGraph publishes the result to GraphSink
```

Batched mode submits all kernels before all D2H transfers:

```text
H2D -> CEL kernel -> SDD kernel -> MI kernel
    -> CEL D2H -> SDD D2H -> MI D2H -> sync
```

Interleaved mode pairs each kernel with its transfer:

```text
H2D -> CEL kernel -> CEL D2H
    -> SDD kernel -> SDD D2H
    -> MI kernel  -> MI D2H -> sync
```

### 5. Failure and teardown

Any lifecycle or execution error sets a process-wide cancellation flag. Every
ready frame receives one cancelled terminal result, in-flight work completes or
fails, graph workers join, and every initialized task unloads. No throughput is
reported for a partial run.

Normal teardown is:

```text
stop and join graph workers
  -> StaticData releases frame-owned device inputs on the NUMA teardown thread
  -> synchronize every task stream during task unload
  -> close CEL/SDD/MI resources
  -> free task scratch, pinned staging, and stream
  -> unregister task resources
  -> release retained primary contexts
```

## Ownership summary

| Owner | Resources |
| --- | --- |
| `GpuContext` | GPU/NUMA identity, retained primary context, registered-task table |
| `DummyGraph` | NUMA-local workers, task pool, ready CPU-atom queue, CPU atoms, `StaticData`, parameter registry |
| `StaticData` | Fixed 220-slot container, immutable frame-ID index, frame state/results, and FrameSlot GPU lifetime |
| `FrameCpuAtom` | CPU frame bytes and intrinsic metadata: ID, byte count, dimensions, and dtype |
| `FrameSlot` | Copied frame metadata, scheduling state, frame-owned device data, and CEL/SDD/MI results |
| `FrameGpuData` | GPU-keyed replica table, device allocation, resident frame ID, and validity state |
| `FrameGpuAccess` | Scoped non-owning replica view with sync and commit-or-abort behavior |
| `DummyTask` | GPU binding, stream, pinned input staging, scratch, CEL/SDD/MI objects |
| CEL/SDD/MI object | Geometry, parameters, device output, pinned D2H staging output |
| Graph thread | CPU affinity and the duration of one `execute()` call only |

`DummyGraph` owns the configured `FrameCpuAtom` objects, while its `StaticData`
member owns every `FrameSlot`. `DummyGraph::ReadyFrame` carries only an atom.
`DummyTask` asks `StaticData` for the atom's frame ID using an immutable hash
index, then requires a complete metadata match. The returned pointer is
non-owning and lives only for the current synchronous call. `FrameSlot` stores
neither an atom nor an atom pointer.

`FrameGpuData` is embedded directly in `FrameSlot`. The separate class
encapsulates CUDA allocation, GPU-keyed replica metadata, frame-ID residency,
and access validation; it does not introduce a lifetime independent of the frame.
`FrameGpuAccess` is an RAII lease, not a buffer owner. The graph protocol
serializes stages of a frame, so frame storage does not contain its own
scheduling guard.

The task and worker defaults are equal, but `GraphConfig::taskInstancesPerGpu`
and `GraphConfig::graphThreads` are independent. Either pool may limit
concurrency.

## Memory policy

Every graph copy constructs a 220-object `FrameSlot` pool. Each configured
warmup or timed logical frame binds one slot and receives one device input
replica plus three pageable result matrices before workers start. Unbound pool
entries own no device/result allocation. This keeps the hot path free of
allocation while making allocated host/device storage proportional to the
configured frame count, up to the fixed capacity.

Each device replica is allocated once during graph initialization and retained
until graph teardown. Reusing a slot for a new logical frame performs `Upload`
into the same device pointer; it does not call `cudaMalloc()` or `cudaFree()`.
After that upload completes, stages may only request `Read` for that frame ID.

Task-owned CUDA resources scale with task count. Atom-owned CPU storage and
bound-slot device/result storage scale one-to-one with configured frames.
Neither an atom nor a slot owns a CUDA stream.

## Source layout

```text
src/
  GpuContext.*            retained per-GPU context and task registration table
  GpuContextManager.*     discovery, affinity, task registration, device binding
  FrameCpuAtom.*          CPU frame bytes and intrinsic metadata
  FrameMetadata.h         metadata shared by CPU atoms and GPU slots
  FrameGpuAccess.*        scoped RAII replica access and completion handling
  FrameGpuData.*          frame-owned replicas, resident frame ID, validity state
  FrameSlot.*             frame metadata, GPU storage, scheduling state, results
  StaticData.*            fixed FrameSlot pool, immutable ID index, frame state API
  TaskGpuResources.h      one task's reusable CUDA lane
  ParameterRegistry.*     typed registration and immutable run snapshot
  GraphTypes.h            shared execution, phase, and frame-state enums
  DummyTask.*             golden lifecycle and CEL/SDD/MI execution
  DummyGraph.*            NUMA graph copy, scheduler, workers, sink, phase gate
  IAlgo.h                  internal algorithm contract
  Cel.*, Sdd.*, Mi.*      synthetic CUDA algorithms
  main.cpp                topology grouping and benchmark coordination
tests/
  gpuinfra_tests.cpp      protocol and CUDA integration tests
```
