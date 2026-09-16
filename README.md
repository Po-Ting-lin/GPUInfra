# GPUInfra CUDA graph demo

This repository is a runnable CUDA model of the golden protocol in
[`graph.md`](graph.md):

```text
start -> DummyTask -> end
```

`DummyTask` contains three ordered synthetic CUDA operations—CEL, SDD, and MI.
They are private algorithms, not separate scheduler tasks. The scheduler still
selects a ready frame, a free task instance, and a free NUMA-local worker. GPU
cache residency never participates in that decision.

The implementation demonstrates:

- one `DummyGraph` copy per GPU-bearing NUMA node;
- an explicit temporary limit of one GPU per NUMA graph copy;
- GPU-bound task instances that may move between NUMA-local workers;
- graph-level exclusive task checkout;
- `FrameCpuAtom` ownership of CPU data, metadata, and preallocated result;
- `DummyGraph` ownership of warmup/timed membership and execution state;
- `StaticData` validation of one fixed frame layout and explicit run-boundary
  cache reset;
- a bounded, best-effort `GpuCacheManager` with persistent `GpuCacheEntry` device
  buffers and a fixed-capacity open-addressing residency table;
- task-private persistent `d_input` fallback on cache miss;
- no hot-path `cudaMalloc()`/`cudaFree()`;
- explicit register/load/start/execute/stop/unload lifecycle barriers;
- change-driven parameter notification at quiescent boundaries;
- graph-wide fail-fast cancellation.

The cache avoids repeated H2D when possible, but correctness does not depend on
a hit. Because the current input is immutable, a miss re-uploads from
`FrameCpuAtom` into the selected task's fallback buffer.

## Design references

- [`graph.md`](graph.md) is the authoritative scheduler/lifecycle reference,
  corrected to match the reviewed real-framework callbacks.
- [`architecture.md`](architecture.md) documents the implemented model.
- [`frame_gpu_data_plan_tmp.md`](frame_gpu_data_plan_tmp.md) specifies the
  implemented one-GPU cache.
- [`frame_gpu_data_plan.md`](frame_gpu_data_plan.md) specifies the future
  multi-GPU replica extension.
- [`num_of_gpu_cache_entry_issue.md`](num_of_gpu_cache_entry_issue.md) explains
  why logical frame count and GPU cache capacity are independent.
- [`unordered_map_vs_fixed_open_addressing.md`](unordered_map_vs_fixed_open_addressing.md)
  compares the two resident-key index implementations for N=1024, S=300, and
  K=150.
- [`open_issues.md`](open_issues.md) records real-framework integration and
  future payload constraints.
- [`gpuinfra_class_diagram.html`](gpuinfra_class_diagram.html) and
  [`gpuinfra_resource_plot.html`](gpuinfra_resource_plot.html) visualize
  UML ownership, multiplicity, resources, and hit/fallback flow.
- [`gpu_cache_explained.html`](gpu_cache_explained.html) gives the complete
  incoming-frame walkthrough for K, residency slots, global LRU, cache
  hit/fill/fallback, and allocation lifetime.

## Requirements

- Linux NUMA topology under `/sys/devices/system/node`
- CMake 3.24 or newer
- C++17 compiler
- NVIDIA CUDA Toolkit and compatible driver
- at least one CUDA GPU with a discoverable PCI NUMA node
- exactly one discovered GPU per participating NUMA node for the temporary
  implementation

The default CUDA architecture is `sm_86`. Override it when building:

```bash
./build.sh -DCMAKE_CUDA_ARCHITECTURES=89
```

## Build, run, and test

```bash
./build.sh
./build/gpuinfra_demo
ctest --test-dir build --output-on-failure
```

Equivalent build commands:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

Demo arguments remain:

```text
gpuinfra_demo [timed_frames_per_gpu] [warmup_frames_per_gpu]
              [batched|interleaved] [size_factor]
```

Defaults are 200 timed frames, 20 warmup frames, Batched execution, and size
factor 128. The factor must be a multiple of 16 from 16 through 256. There is no
220-frame implementation cap anymore.

`GraphConfig::gpuCacheEntries` defaults to 4 and is intentionally not exposed
as a demo CLI argument. Setting it to zero programmatically disables caching
and exercises the task fallback path.

## Independent H2D, D2H, and compute sizing

Three positive compile-time macros independently scale the submitted work:

| CMake setting / macro | Default | Effect |
| --- | ---: | --- |
| `GPUINFRA_H2D_SIZE_MULTIPLIER` | 1 | Full-buffer H2D copies per upload |
| `GPUINFRA_D2H_SIZE_MULTIPLIER` | 1 | Full-buffer D2H copies per algorithm |
| `GPUINFRA_COMPUTE_SIZE_MULTIPLIER` | 3 | Matrix calculations per output element |

For example:

```bash
./build.sh \
    -DGPUINFRA_H2D_SIZE_MULTIPLIER=2 \
    -DGPUINFRA_D2H_SIZE_MULTIPLIER=4 \
    -DGPUINFRA_COMPUTE_SIZE_MULTIPLIER=6
```

These multipliers change transfer traffic and compute work without changing
function APIs, allocation sizes, image dimensions, or result contents. H2D is
still skipped on a cache hit. The defaults reproduce the previous one H2D,
one D2H, and three compute repetitions.

## Runtime topology

`GpuContextManager::init()` discovers Runtime API devices, maps their PCI NUMA
nodes, primes primary contexts, and retains one Driver API primary-context
reference per GPU. The simulation in `main.cpp` creates a `NumaExecutor` and a
graph copy for each GPU-bearing NUMA node. NUMA placement belongs to this
framework execution environment; `GraphConfig` has no `numaNode` or `gpuIds`.

The framework establishes NUMA CPU affinity before every task callback,
including registration, load, parameter notification, start, execute, stop,
and unload. `Grape/NumaExecutor` implements that guarantee for the simulation
and tests. Individual callbacks and `GpuContextManager` do not set affinity.

`GpuContext` is the authoritative GPU-to-NUMA mapping. During `load()`, a task
observes its current NUMA node and asks `GpuContextManager` for the local GPU.
Zero or multiple GPUs on that node are errors; no arbitrary first GPU is
selected. The selected GPU is stored in `TaskGpuResources`, not supplied to
the task constructor. `StaticData::init()` uses the same NUMA lookup policy.
`DummyGraph` resolves its GPU list inside the execution environment before
allocating tasks and retains `taskInstancesPerGpu * gpuIds.size()` and the
existing frame/worker count calculations.

Tasks remain GPU-bound for the lifetime of their loaded resources, but may
execute on different threads within the same NUMA node. `makeTaskCurrent()`
is still required: CPU affinity does not select a CUDA device for a host
thread. Tasks, task resources, `StaticData`, and private algorithms do not
retain duplicate NUMA identity.

## Cold initialization

```text
framework NUMA executor resolves exactly one local GPU
  -> construct each DummyTask and immediately register its parameter schema
  -> define initial parameter values and seal the shared schema
  -> load all tasks
       -> resolve the current NUMA node and register the unique local GPU
       -> stream + h_in + d_input fallback + scratch + algo-private buffers
       -> apply initial parameter values inside load
  -> create separate warmup/timed FrameCpuAtom collections
       -> preallocate each atom's input and CEL/SDD/MI result buffers
  -> StaticData::init()
       -> resolve the same local GPU through the NUMA lookup
       -> store the fixed frame layout
       -> create exactly K persistent GpuCacheEntry cache entries
       -> allocate a fixed open-addressing table plus empty/LRU structures
  -> start NUMA-local workers
  -> DummyGraph::start() calls DummyTask::start() on every task
```

`DummyTask::start()` and `stop()` are lifecycle guards in this simulation; they
do not move or resize CUDA resources. Warmup and timed phases share one graph
execution cycle. A later `DummyGraph::changeParameters()` call is accepted only
when no phase is active and no execution is in flight, and only a changed
revision is delivered to task instances.

`K = GraphConfig::gpuCacheEntries`. The default workload has 220
`FrameCpuAtom` objects but only four GPU cache entries per graph copy. No list
of those 220 frame keys is copied into `StaticData`.

## Scheduling

Under `DummyGraph::schedulerLock`, a worker atomically claims:

```text
ready FrameCpuAtom + free DummyTask + free graph thread
```

The task is removed from `freeTasks` until `execute()` finishes, preventing two
threads from using the same task instance concurrently. The lock is released
before CUDA work. No cache lookup, hit/miss, or GPU pointer affects which task
the graph selects.

The call remains:

```cpp
bool DummyTask::execute(FrameCpuAtom& atom, StaticData& staticData);
```

## Logical frame versus GPU cache

```text
StaticData
  ├─ fixed frame layout + resetCache() run boundary
  └─ GpuCacheManager
       ├─ fixed open-addressing GpuDataKey -> entry index table [~2K]
       ├─ O(1) empty stack + intrusive inactive-entry LRU
       └─ GpuCacheEntry[K]
            cached metadata + lease/LRU links
            └─ GpuReplica[gpuId] -> persistent d_data + validity
```

`StaticData::getCacheData()` accepts any incoming `GpuDataKey` whose metadata
matches the fixed frame layout. `GpuDataKey` contains `frameId` and `cameraId`,
so the same frame ID from different cameras cannot alias. `StaticData` stores
no per-frame registry,
scheduler state, `FramePhase`, CPU bytes, `JobResult`, or NUMA identity: graph
collections own execution state and the owning `FrameCpuAtom` carries its data
and result.
`GpuCacheEntry` is only a reusable best-effort GPU cache entry. It may represent
different logical frames over time while retaining the same device allocation.

## Cache access

`GpuCacheManager::getCacheData()` uses its fixed residency table and returns an RAII
`GpuDataAccess`. `status()` returns `CacheStatus`; `getStream()` returns the
borrowed task stream supplied in `GpuCacheRequest`:

| Status | Path |
| --- | --- |
| `CacheHit` | Matching `Valid` entry; use its immutable device pointer without H2D |
| `CacheFill` | Reserve an empty/inactive-LRU entry; caller uploads or computes its payload |
| `TaskFallback` | Caller fills its supplied fallback buffer when capacity is zero, or a Loading/full-cache wait expires |
| `Invalid` | Metadata, GPU, stream, or fallback request is invalid; do not submit work |

Lookup and resident-key insertion/erasure are average `O(1)`. The table is
allocated at about twice `K`, uses linear probing and backward-shift deletion,
and never grows or rehashes. An empty-entry stack and intrusive LRU choose a
victim in `O(1)`. Active entries cannot be evicted. A fill becomes `Valid` only
after the task stream synchronizes successfully. Failed or abandoned fills
return to `Empty`; a failed hit keeps the immutable cached payload.

## Caller-owned miss handling and independent caches

`GpuCacheRequest` explicitly supplies `gpuId`, `stream`, `d_fallback`, and
`fallbackBytes`. The fallback pointer and stream must be valid, and fallback
capacity must be at least the cache payload size, even on an expected hit.
The cache borrows these resources; it never allocates or frees caller fallback
buffers and does not perform H2D or run algorithms.

```cpp
GpuCacheRequest request;
request.gpuId = resources.gpuId;
request.stream = resources.stream;
request.d_fallback = resources.d_input;
request.fallbackBytes = resources.inBytes;
GpuDataAccess access = staticData.getCacheData(metadata, request);
const CacheStatus status = access.status();
if (status == CacheStatus::Invalid) return false;
const cudaStream_t stream = access.getStream();
bool submittedSuccessfully = true;
if (status == CacheStatus::CacheFill || status == CacheStatus::TaskFallback) {
    submittedSuccessfully = enqueueH2D(access.writableData(), stream);
}
if (submittedSuccessfully) {
    submittedSuccessfully = enqueueComputeAndD2H(access.data(), stream);
}
// Finish once after all work on stream, even if submission fails.
bool succeeded = access.freeCacheData(submittedSuccessfully);
```

Independent `GpuCacheManager` instances may cache different fixed payload
sizes with separate capacities, indexes, and LRU lists. A frame cache and a
result cache may use the same key without sharing entries. Each live payload
needs fallback storage that will not be overwritten by another live request;
using the same task input buffer for simultaneous frame and result fallback
is unsafe. Each request receives its own `GpuDataAccess`, including concurrent
readers of the same key on different task streams.

Fill publication occurs only in `freeCacheData()` after successful stream
synchronization, not as soon as H2D completes. A second request for a Loading
key waits within its request deadline, then uses fallback if still unavailable. With
all work on the returned stream, no extra H2D event is required. Each access
finishes independently; finishing two accesses on one stream currently causes
two synchronization calls and is not an atomic multi-cache publication.

The demo still uses one frame cache in `StaticData`; it does not automatically
cache CEL/SDD/MI outputs. A result-cache caller must define versioned identity
or reset at parameter changes and be able to recreate evicted data. The
existing frame-shaped metadata and immutable, best-effort eviction contract
remain in place.

## Bounded cache waiting

Configure each manager at initialization; the default is **50 ms**:

```cpp
const std::chrono::milliseconds waitTimeout(50);
const bool initialized = cache.initialize(gpuIds, payloadBytes, cacheEntries, waitTimeout);
```

For the frame-cache wrapper, set `StaticDataConfig::gpuCacheWaitTimeout`.
There is no per-request override. Zero disables waiting; negative values are
rejected. A zero-capacity cache always returns fallback immediately.

A matching Loading entry or a full cache with no evictable entry waits on a
condition variable, releasing the mutex. Each call has one steady-clock
deadline, measured from entry to `getCacheData()`; notifications and transitions
between lookup branches never restart it. On wakeup, lookup runs again. A
successful fill permits a hit; failed/abandoned fill permits one contender to
claim CacheFill. The last reader releasing an entry also wakes contenders.
Loading entries and active readers never enter the evictable LRU.

At the deadline, lookup checks availability once more and falls back if still
blocked. This is a wait budget, not a hard wall-clock latency guarantee: thread
scheduling and mutex reacquisition can add delay. Waiters have no FIFO priority
or reserved entry. Holding another access while waiting can consume the entire
budget, so callers should avoid circular dependencies.

Reset/release return false while any request sleeps, even if a producer has
just finished. The manager and borrowed request resources must outlive all
calls and access objects. There is no early publication and no extra CUDA event.

The interactive HTML guides demonstrate the immediate-fallback configuration
(`waitTimeout = 0 ms`); their state transitions also describe the fallback path
after a bounded wait expires.

## Run boundaries and cache reset

`StaticData::resetCache()` is the single cold-path run-boundary operation. It
clears the fixed residency table, entry identities/validity, empty stack, and
global LRU state. It rejects active cache/fallback leases, waiting requests, or an in-progress
fill.

Reset does not call `cudaFree()` or `cudaMalloc()`: every
`GpuCacheEntry` device pointer remains allocated and is reused by later fills.
The graph-copy owner must call it after every old-run execution has finished
and before any new-run execution can start. Between two resets, one
`frameId + cameraId` identity must always represent the same immutable bytes.
If a run reuses an identity with new bytes and reset is skipped, stale cache
data can be returned.

## Frame execution

```text
FrameCpuAtom metadata/layout/result validation
  -> make task GPU current
  -> StaticData::getCacheData()
       -> validate fixed layout
       -> acquire CacheHit / CacheFill / TaskFallback
  -> when status is CacheFill / TaskFallback: atom.data -> task h_in -> selected device buffer
  -> CEL / SDD / MI read access.data()
  -> algorithm D2H staging
  -> GpuDataAccess::freeCacheData()
       -> one cudaStreamSynchronize()
       -> publish/release cache state
  -> copy results into FrameCpuAtom.result
```

Batched mode submits all kernels before all D2H copies. Interleaved mode submits
each algorithm kernel followed by its D2H. All three algorithms consume the
original square byte input; the ordering does not form an output-to-input
chain.

For size factor `F`, input is `(8F)x(8F)`, CEL is `(2F)x(2F)`, and SDD/MI are
`(3F)x(3F)`. Results are row-major `uint32_t` matrices.

## Ownership summary

| Owner | Resources |
| --- | --- |
| `GpuContext` | GPU/NUMA identity, retained primary context, registered task table |
| `DummyGraph` | workers, task pool, ready queue, CPU atoms, `StaticData`, phases, cancellation |
| `FrameCpuAtom` | CPU input bytes, intrinsic metadata, and preallocated `JobResult` |
| `StaticData` | fixed frame layout, reset boundary, and one bounded `GpuCacheManager` |
| `GpuCacheManager` | fixed cache array, fixed open-addressing residency table, O(1) victim structures, metadata lock, and leases |
| `GpuCacheEntry` | one reusable cache entry with persistent GPU-keyed replicas and validity |
| `GpuDataAccess` | scoped non-owning cache/fallback lease and stream completion |
| `DummyTask` / `TaskGpuResources` | GPU binding, stream, `h_in`, fallback `d_input`, scratch, algorithms |
| CEL/SDD/MI | private device outputs, pinned D2H staging, geometry, parameters |

## Memory policy

GPU input storage now scales independently:

```text
cache input VRAM = effective cache entries × frame bytes × replicas per entry
fallback input VRAM = task instances × frame bytes
```

CPU atoms, logical results, task scratch, and algorithm-private resources keep
their own lifetimes. See
[`gpu_mem_consumption.md`](gpu_mem_consumption.md) for the three summary
formulas.

The current cache is correct only for immutable input reproducible from the CPU
atom. A future mutable GPU-only intermediate needs a separate authoritative
frame-owned plane; it cannot rely on best-effort eviction.

## Failure and teardown

Any lifecycle/execution failure raises process-wide cancellation. Queued frames
receive terminal failed results, in-flight work finishes or fails, and workers
join before resource teardown.

```text
finish all in-flight execute calls
  -> DummyTask::stop() for every task in the execution cycle
  -> stop and join workers
  -> StaticData releases cache-entry device allocations
  -> clear CPU atoms
  -> unload tasks
       -> sync stream, close algorithms, free scratch/d_input/h_in/stream
  -> unregister tasks
  -> release retained contexts
```

Cleanup is idempotent after success, cancellation, or partial initialization.

## Verification coverage

The real-CUDA protocol test caches more than 220 unregistered incoming frames
through two entries and covers fixed-table collisions/backward-shift deletion,
capacity zero, layout rejection, reset-boundary lease rejection, reset without
device reallocation, fill/hit/fallback, loading and busy
fallback, RAII abort, failed fill, LRU eviction, stable device pointers,
cross-task reuse, pure-fallback correctness, both execution models, load-time GPU discovery,
framework affinity rejection, graph-level task exclusivity, cancellation, and
cleanup, independent frame/result caches with unequal payload sizes, and
concurrent readers on separate task streams. Synthetic topology tests also reject zero/multiple local GPUs and
verify selection when GPU IDs differ from NUMA IDs; multi-NUMA execution is
checked when the hardware supports it.

## Source layout

```text
src/
  Algo/
    IAlgo.h               algorithm contract and shared result types
    Cel.*, Sdd.*, Mi.*    synthetic CUDA algorithms
  DataCache/
    GpuDataKey.h          frame/camera cache identity
    GpuCacheManager.*     bounded cache lookup, LRU, leases, fallback choice
    GpuCacheEntry.*       reusable entry with persistent per-GPU replicas
    GpuDataAccess.*       explicit status, borrowed stream, scoped lease completion
    GpuCacheRequest.h     caller GPU, stream, fallback pointer and capacity
    GpuResidencyTable.*   fixed open-addressing resident-key index
  Grape/
    README.md             immutable graph-simulation boundary rules
    DummyGraph.*          NUMA graph copy and unchanged scheduler selection
    NumaExecutor.*        framework CPU-affinity and callback dispatch boundary
    FrameCpuAtom.*        CPU bytes, metadata, and preallocated result
    GraphTypes.h          simulated graph-owned execution types
  DummyTask.*             task lifecycle and CEL/SDD/MI execution
  ParameterRegistry.*     sealed schema, mutable values, and revisions
  StaticData.*            graph-copy layout/reset/cache owner
  TaskGpuResources.h      task CUDA lane including fallback d_input
  GpuContextManager.*     GPU discovery, NUMA lookup, task registration
  GpuTopology.*           current NUMA detection and unique local GPU selection
  WorkloadSizing.h        independent H2D/D2H/compute compile-time controls
tests/
  gpuinfra_tests.cpp      protocol and CUDA integration tests
```
