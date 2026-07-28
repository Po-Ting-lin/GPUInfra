# GPUInfra architecture

This document describes the implemented ownership and execution model in the
current GPUInfra demo. The design relies on Graph framework threads for
parallelism and does not create an internal queue, dispatcher, or worker pool.

The current demo uses:

- one CUDA `GpuContext` per discovered GPU;
- four Graph threads per GPU;
- one `DummyGraph` and `ThreadSlot` per Graph thread;
- three private algorithms per `DummyGraph`: CEL, SDD, and MI;
- one stream and one input pair per slot;
- one device and pinned-host output pair per algorithm object.

The algorithm count and thread count are configuration choices. The ownership
model remains the same when either count changes.

## 1. Design decisions

| Topic | Decision |
| --- | --- |
| Parallelism owner | External Graph framework |
| GPU binding | Fixed for the lifetime of a registered Graph worker |
| Algorithm ownership | One private algorithm object set per `DummyGraph` |
| Input and shared scratch ownership | `ThreadSlot` |
| Output buffer ownership | Each private algorithm object |
| Primary context and per-GPU shared data | `GpuContext` |
| Streams | One nonblocking stream per slot |
| Input transfer | One H2D per frame, shared by all algorithms |
| Default ordering | Kernel batch followed by D2H batch |
| Host wait | One `cudaStreamSynchronize()` per frame |
| Hot-path allocation | Forbidden |
| NUMA policy | Pin before pinned-host allocation and first touch |

The ownership boundary follows lifetime and mutation:

- per-thread mutable execution state belongs to `DummyGraph` or `ThreadSlot`;
- output resources used by one algorithm belong to that algorithm;
- GPU resources shared by every algorithm in one worker belong to `ThreadSlot`;
- large immutable resources shared by every worker on one GPU belong to
  `GpuContext`.

## 2. Ownership model

### `GpuContextManager`

`GpuContextManager` is the process-wide discovery and registration service. It:

- discovers CUDA devices and PCI NUMA nodes;
- creates one `GpuContext` per GPU;
- retains each device primary context;
- validates the common runtime frame geometry;
- registers and unregisters thread slots;
- allocates and frees optional shared slot scratch;
- releases primary-context references after every worker has stopped.

It does not own algorithm objects.

### `GpuContext`

`GpuContext` owns per-GPU infrastructure:

- CUDA device and NUMA identity;
- retained `CUcontext primaryCtx`;
- the fixed `ThreadSlot*` table;
- input-size and registration limits;
- future large immutable per-GPU resources.

The current CEL, SDD, and MI implementations do not have large immutable model
data. If a future algorithm adds weights, lookup tables, or modules that should
not be duplicated per thread, those resources should be represented by a
separate immutable object owned by `GpuContext`. Per-Graph algorithm instances
may hold non-owning or shared-const references to it.

### `ThreadSlot`

One registered Graph worker owns one slot for its lifetime. The slot contains:

- `cudaStream_t stream`;
- NUMA-local pinned input `h_in`;
- device input `d_in`;
- optional shared device scratch `d_scratch`.

`GpuContext` owns the slot allocation. The Graph worker holds a non-owning
pointer and must unregister it from the same host thread.

### `DummyGraph`

One `DummyGraph` runs on one Graph worker thread. It owns:

- `vector<unique_ptr<IAlgo>> algos`;
- algorithm execution order;
- one reusable `JobResult`;
- the pageable `AlgoOutput` matrices prepared during `load()`.

Every algorithm object is thread-confined. No algorithm instance is shared
between Graph workers.

### `IAlgo`

An `IAlgo` implementation owns its configuration, device output buffer, and
pinned-host D2H buffer. It uses the slot stream, input, and optional scratch
without owning them.

The lifecycle is:

```text
factory.make()
  -> initStatic(gpu, numa)
  -> configure(runtime)
  -> allocateOutputBuffers(slot)
  -> prepareOutput(output)
  -> launchKernels / launchD2H / collectResult
  -> close()
```

CEL, SDD, and MI store the expected GPU and slot ID when allocating. Calls
using a different slot are rejected.

## 3. Startup and worker lifecycle

### Manager startup

The main thread calls:

```cpp
GpuContextManager::init(config);
GpuContextManager::configure(runtime);
```

`init()` discovers devices, establishes NUMA placement, primes the Runtime API,
and retains primary contexts. `configure()` validates the input dimensions and
marks contexts ready. Neither call creates algorithm objects or output buffers.

### Graph worker load

Each external worker constructs one `DummyGraph` and calls `load()`:

```text
register ThreadSlot
  -> create private algorithm objects
  -> configure each algorithm
  -> collect the maximum shared-scratch requirement
  -> allocate optional slot scratch
  -> allocate each algorithm's output buffers
  -> prepare reusable pageable outputs
```

All allocation occurs before warmup and timing.

If any step fails, `DummyGraph` destroys the objects it already created and
unregisters the slot. Each algorithm frees any output buffers it allocated;
slot teardown frees input and scratch resources.

### Graph worker unload

Teardown order is:

```text
synchronize slot stream
  -> close private algorithms and free their output buffers
  -> destroy private algorithms
  -> release pageable JobResult storage
  -> unregister ThreadSlot
  -> free slot input, scratch, and stream
```

After every worker has joined, manager shutdown releases the primary-context
references.

## 4. Algorithm-owned output buffers

Every private algorithm allocates two persistent buffers during
`DummyGraph::load()`:

```text
CEL -> d_outputMatrix + h_outputMatrix
SDD -> d_outputMatrix + h_outputMatrix
MI  -> d_outputMatrix + h_outputMatrix
```

The device buffer holds the kernel result. The pinned-host buffer is the target
of the algorithm's asynchronous D2H copy. The buffers remain allocated and are
reused for every frame until `close()`.

Separate buffers are required in Batched mode:

```text
kernel(CEL)
kernel(SDD)
kernel(MI)
D2H(CEL)
D2H(SDD)
D2H(MI)
```

All three device outputs remain live until their D2H copies are enqueued. All
three pinned-host results remain live until the single stream synchronization
and result collection complete.

`d_scratch` is different: it may be sized to the maximum scratch requirement
and reused by algorithms because all kernels in one slot are ordered on the
same stream. Scratch contents are undefined when the next algorithm begins.

### Memory scaling

For `K` threads and `M` algorithms on one GPU:

```text
device input       = K * inputBytes
pinned input       = K * inputBytes
device outputs     = K * sum(deviceOutputBytes[i])
pinned D2H output  = K * sum(hostOutputBytes[i])
device scratch     = K * max(scratchBytes[i])
pageable results   = K * sum(hostOutputBytes[i])
```

Moving algorithm objects from `GpuContext` to `DummyGraph` does not multiply
the output-buffer count. The old design had one per-GPU object with `K`
internal states; the current design has `K` private objects that each own one
buffer pair. Both require `K * M` output-buffer pairs.

Large per-object immutable data would be multiplied by `K`, which is why such
data belongs in a `GpuContext`-owned shared resource.

## 5. Frame execution

The frame hot path is allocation-free:

```text
caller frame
  -> memcpy into slot.h_in
  -> cudaMemcpyAsync H2D into slot.d_in
  -> algorithm kernel operations
  -> algorithm D2H operations into algorithm-owned pinned buffers
  -> one cudaStreamSynchronize(slot.stream)
  -> memcpy pinned buffers into prepared AlgoOutput vectors
  -> GraphSink::deliver(result)
```

### Batched model

```text
H2D
  -> CEL kernel
  -> SDD kernel
  -> MI kernel
  -> CEL D2H
  -> SDD D2H
  -> MI D2H
  -> sync
```

This groups compute and copy submissions while preserving stream order.

### Interleaved model

```text
H2D
  -> CEL kernel -> CEL D2H
  -> SDD kernel -> SDD D2H
  -> MI kernel  -> MI D2H
  -> sync
```

Both models use the same algorithm-owned output buffers and exactly one
host-side CUDA wait per frame.

## 6. NUMA and context discipline

- Explicit `numaHint` pins the Graph worker before slot allocation.
- Pinned input and algorithm output allocations occur after pinning.
- `registerThread()` binds the selected CUDA device on the calling thread.
- Algorithm allocation calls `cudaSetDevice()` for the slot GPU.
- Slot registration, scratch preparation, and teardown are cold paths protected
  by the manager mutex.
- Kernel launch and transfer operations use no manager mutex.
- A slot must be used and unregistered by its recorded `ownerTid`.

NUMA node and CUDA context are separate concepts. NUMA placement controls CPU
affinity and host-memory locality; CUDA context identity controls GPU resource
visibility.

## 7. Hot-path rules

The following rules are non-negotiable during `DummyGraph::execute()`:

- no `cudaMalloc`, `cudaFree`, `cudaHostAlloc`, or `cudaFreeHost`;
- no vector resize, reserve, or result ownership transfer;
- no `cudaDeviceSynchronize`;
- no process-wide queue or manager mutex;
- no synchronous CUDA copies;
- no algorithm-shared mutable state across Graph threads;
- exactly one `cudaStreamSynchronize()` per frame.

CUDA allocations, vector growth, and output preparation are allowed only
during `load()` or teardown.

## 8. Adding an algorithm

To add algorithm D:

1. implement `IAlgo`;
2. compute output sizes in `configure()`;
3. allocate owned device and pinned-host outputs in
   `allocateOutputBuffers()`;
4. enqueue compute in `launchKernels()`;
5. enqueue D2H in `launchD2H()`;
6. copy into the prepared pageable destination in `collectResult()`;
7. register its factory in `main.cpp`.

No change to `ThreadSlot` or the execution loops is required.

If the algorithm needs temporary storage, report it through
`scratchBytesNeeded()` and use `slot.d_scratch`. If it needs immutable
per-GPU model data, add an explicit shared-resource type owned by
`GpuContext`; do not place a separate copy in every `DummyGraph`.

## 9. Verification

The repository verifies:

- the default Batched run;
- Interleaved execution;
- minimum and maximum valid size factors;
- rejection of unaligned and oversized factors.

Use:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For performance validation:

1. inspect the `nsys` timeline for one row per slot;
2. confirm one H2D, three kernels, three D2H operations, and one synchronization
   per frame;
3. use `numastat -p <pid>` to check pinned-host locality;
4. use `nvidia-smi dmon` to compare SM utilization and PCIe traffic;
5. sweep thread count and size factor rather than assuming more streams improve
   throughput.

## 10. Mental model

`GpuContext` represents one GPU and its process-wide context identity.
`ThreadSlot` represents one Graph worker's CUDA lane and owns every mutable CUDA
buffer shared by that worker's algorithms. `DummyGraph` owns private algorithm
objects and drives them in order. Each algorithm owns its output buffers and
frees them during `close()`. This keeps state and teardown local while leaving
a clear place in `GpuContext` for future immutable resources that should be
shared across workers.
