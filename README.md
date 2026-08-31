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
- graph-wide lifecycle barriers and fail-fast cancellation.

The cache avoids repeated H2D when possible, but correctness does not depend on
a hit. Because the current input is immutable, a miss re-uploads from
`FrameCpuAtom` into the selected task's fallback buffer.

## Design references

- [`graph.md`](graph.md) is the untouched golden scheduler/lifecycle reference.
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
  ownership and hit/fallback flow.
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
reference per GPU. `main.cpp` groups devices by NUMA node and creates one graph
copy per node. A graph copy with zero or more than one GPU is rejected in the
temporary scope.

Setup, teardown, and workers run on threads pinned to the graph copy's NUMA
node. A `DummyTask` remains bound to one GPU because its stream and allocations
belong to that device, but it has no permanent host-thread binding.

`GpuContext` is the authoritative GPU-to-NUMA mapping. Each `DummyGraph`
validates its complete GPU list against that mapping once before creating
resources. NUMA identity is not copied into `DummyTask`, `TaskGpuResources`,
`StaticData`, or private algorithms.

## Cold initialization

```text
validate graph GPU IDs belong to the graph NUMA node
  -> create all DummyTask instances
  -> load all tasks
       -> stream + h_in + d_input fallback + scratch + algo-private buffers
  -> register one shared parameter schema
  -> seal and notify one immutable parameter snapshot
  -> create separate warmup/timed FrameCpuAtom collections
       -> preallocate each atom's input and CEL/SDD/MI result buffers
  -> StaticData::init()
       -> store the fixed frame layout
       -> create exactly K persistent GpuCacheEntry cache entries
       -> allocate a fixed open-addressing table plus empty/LRU structures
  -> start NUMA-local workers
```

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

`StaticData::acquireGpuData()` accepts any incoming `GpuDataKey` whose metadata
matches the fixed frame layout. `GpuDataKey` contains `frameId` and `cameraId`,
so the same frame ID from different cameras cannot alias. `StaticData` stores
no per-frame registry,
scheduler state, `FramePhase`, CPU bytes, `JobResult`, or NUMA identity: graph
collections own execution state and the owning `FrameCpuAtom` carries its data
and result.
`GpuCacheEntry` is only a reusable best-effort GPU cache entry. It may represent
different logical frames over time while retaining the same device allocation.

## Cache access

`GpuCacheManager::acquire()` uses its fixed residency table and returns an RAII
`GpuDataAccess`:

| Source | Path |
| --- | --- |
| `CacheHit` | Matching `Valid` entry; use its immutable device pointer without H2D |
| `CacheFill` | Reserve an empty/inactive-LRU entry and H2D into its existing allocation |
| `TaskFallback` | H2D into `TaskGpuResources::d_input` when capacity is zero, the matching entry is loading, or all entries are active |

Lookup and resident-key insertion/erasure are average `O(1)`. The table is
allocated at about twice `K`, uses linear probing and backward-shift deletion,
and never grows or rehashes. An empty-entry stack and intrusive LRU choose a
victim in `O(1)`. Active entries cannot be evicted. A fill becomes `Valid` only
after the task stream synchronizes successfully. Failed or abandoned fills
return to `Empty`; a failed hit keeps the immutable cached payload.

## Run boundaries and cache reset

`StaticData::resetCache()` is the single cold-path run-boundary operation. It
clears the fixed residency table, entry identities/validity, empty stack, and
global LRU state. It rejects active cache/fallback leases or an in-progress
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
  -> StaticData::acquireGpuData()
       -> validate fixed layout
       -> acquire CacheHit / CacheFill / TaskFallback
  -> when needsUpload(): atom.data -> task h_in -> selected device buffer
  -> CEL / SDD / MI read access.data()
  -> algorithm D2H staging
  -> GpuDataAccess::complete()
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
stop and join workers
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
cross-task reuse, pure-fallback correctness, both execution models, one-time
GPU/NUMA topology rejection, graph-level task exclusivity, cancellation, and
cleanup.

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
    GpuDataAccess.*       scoped access source and completion
    GpuResidencyTable.*   fixed open-addressing resident-key index
  Grape/
    README.md             immutable graph-simulation boundary rules
    DummyGraph.*          NUMA graph copy and unchanged scheduler selection
    FrameCpuAtom.*        CPU bytes, metadata, and preallocated result
    GraphTypes.h          simulated graph-owned execution types
  DummyTask.*             task lifecycle and CEL/SDD/MI execution
  StaticData.*            graph-copy layout/reset/cache owner
  TaskGpuResources.h      task CUDA lane including fallback d_input
  GpuContextManager.*     GPU discovery, NUMA affinity, task registration
  WorkloadSizing.h        independent H2D/D2H/compute compile-time controls
tests/
  gpuinfra_tests.cpp      protocol and CUDA integration tests
```
