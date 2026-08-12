# GPUInfra CUDA graph demo

This repository is a runnable CUDA model of the one-task graph described by
`graph.md`:

```text
start -> DummyTask -> end
```

`DummyTask` contains three ordered synthetic operations—CEL, SDD, and MI—but
they are not separate graph tasks. The graph scheduler independently matches a
ready `FrameSlot`, a free `DummyTask` instance, and a free NUMA-local graph
thread for each `execute()` call.

The implementation demonstrates:

- one `DummyGraph` copy per GPU-bearing NUMA node;
- independently sized task-instance and graph-thread pools;
- four GPU-bound `DummyTask` instances per GPU by default;
- task instances that move between graph threads on the same NUMA node;
- task-owned CUDA streams, staging buffers, device input, scratch, and
  CEL/SDD/MI resources;
- graph-owned, fully preallocated frame inputs and results;
- graph-wide lifecycle barriers in the golden order;
- fail-fast cancellation across every NUMA graph copy;
- one H2D transfer and one `cudaStreamSynchronize()` per frame.

The synthetic algorithms launch real CUDA matrix-multiplication kernels over a
shared square byte input. For size factor `F`, the input is `(8F)x(8F)`, CEL
operates on `(2F)x(2F)`, and SDD and MI operate on `(3F)x(3F)`. Each result is a
row-major `uint32_t` matrix. All three operations consume the original input;
`CEL -> SDD -> MI` specifies submission order, not an output-to-input chain.

## Requirements

- Linux with NUMA topology under `/sys/devices/system/node`
- CMake 3.24 or newer
- A C++17 compiler
- NVIDIA CUDA Toolkit and compatible driver
- At least one CUDA GPU with a discoverable PCI NUMA node

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
factor 128. The factor must be a multiple of 16 from 16 through 256.

Run the complete test suite:

```bash
ctest --test-dir build --output-on-failure
```

`gpuinfra_protocol_tests` uses a real CUDA GPU to verify lifecycle ordering,
typed parameter registration, task movement between two live host threads,
concurrent-use rejection, CPU-reference results for both execution models,
independent task/thread bottlenecks, malformed input, cancellation, and cleanup.
Multi-NUMA coverage runs conditionally and reports a skip when the topology is
not available.

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

### 2. Graph-copy cold path

Each graph runs setup on a temporary NUMA-pinned setup thread. It creates the
configured number of GPU-bound `DummyTask` instances and then performs
copy-wide phases:

```text
load every DummyTask
  -> registerParameters on every DummyTask
  -> seal one shared parameter snapshot
  -> notifyParameters on every DummyTask
  -> preallocate every warmup and timed FrameSlot
  -> start and pin graph workers
```

`ParameterRegistry` registers `dummy.name` as a string and `dummy.blob` as a
byte vector. Re-registration must agree on type. Values become immutable when
the graph seals the snapshot, and every task instance receives the same values.

### 3. Scheduling

Workers are persistent and pinned to their graph copy's NUMA node. Under the
graph scheduler lock, a free worker atomically claims the first ready frame and
one free task. The lock is released before CUDA work begins.

```text
ready FrameSlot + free DummyTask + free graph thread
                         |
                         v
                DummyTask::execute(slot)
```

A task remains bound to one GPU because its reusable allocations belong to
that device. It is not bound to a host thread. Before each execution, the
selected worker calls `cudaSetDevice(taskGpu)` and then uses the task's stream
and buffers. Any worker in the same NUMA graph copy may run any local-GPU task.

Warmup completes on every graph copy before timed slots are released through a
shared `PhaseGate`.

### 4. Frame execution

`DummyTask::execute()` is allocation-free:

```text
FrameSlot input
  -> memcpy to task-owned pinned input
  -> one asynchronous H2D copy
  -> CEL, SDD, and MI CUDA work
  -> D2H into each algorithm's task-owned pinned staging buffer
  -> one stream synchronization
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
  -> synchronize every task stream
  -> close CEL/SDD/MI resources
  -> free task scratch, input, and stream
  -> unregister task resources
  -> release frame payloads
  -> release retained primary contexts
```

## Ownership summary

| Owner | Resources |
| --- | --- |
| `GpuContext` | GPU/NUMA identity, retained primary context, registered-task table |
| `DummyGraph` | NUMA-local workers, task pool, ready queue, frame slots, parameter registry |
| `FrameSlot` | Frame identity/state, input bytes, CEL/SDD/MI result vectors |
| `DummyTask` | GPU binding, stream, pinned/device input, scratch, CEL/SDD/MI objects |
| CEL/SDD/MI object | Geometry, parameters, device output, pinned D2H staging output |
| Graph thread | CPU affinity and the duration of one `execute()` call only |

The task and worker defaults are equal, but `GraphConfig::taskInstancesPerGpu`
and `GraphConfig::graphThreads` are independent. Either pool may limit
concurrency.

## Memory policy

Every warmup and timed frame owns its full input and three pageable result
matrices before workers start. This deliberately keeps the hot path free of
host allocation, but memory grows with total frame count rather than maximum
concurrency. Large frame counts or factors should be sized with that tradeoff
in mind.

Task-owned CUDA resources scale with task count. Frame-owned pageable storage
scales with total frames. No frame owns a CUDA stream or device allocation.

## Source layout

```text
src/
  GpuContext.*            retained per-GPU context and task registration table
  GpuContextManager.*     discovery, affinity, task registration, device binding
  TaskGpuResources.h      one task's reusable CUDA lane
  ParameterRegistry.*     typed registration and immutable run snapshot
  GraphTypes.*            frame slots, phases, states, execution model
  DummyTask.*             golden lifecycle and CEL/SDD/MI execution
  GraphStuff.*            NUMA graph copy, scheduler, workers, sink, phase gate
  IAlgo.h                  internal algorithm contract
  Cel.*, Sdd.*, Mi.*      synthetic CUDA algorithms
  main.cpp                topology grouping and benchmark coordination
tests/
  gpuinfra_tests.cpp      protocol and CUDA integration tests
```
