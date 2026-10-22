# GPUInfra CUDA graph demo

This repository is a runnable CUDA model of the golden protocol in
[`graph.md`](docs/graph.md):

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
- a bounded, best-effort `GpuDataCache` with persistent `GpuDataCacheEntry` device
  buffers and a fixed-capacity open-addressing residency table;
- task-private persistent `d_input` fallback on cache miss;
- no hot-path `cudaMalloc()`/`cudaFree()`;
- explicit register/load/start/execute/stop/unload lifecycle barriers;
- change-driven parameter notification at quiescent boundaries;
- graph-wide fail-fast cancellation.

The cache avoids repeated H2D when possible, but correctness does not depend on
a hit. Because the current input is immutable, a miss re-uploads from
`FrameCpuAtom` into the selected task's fallback buffer.

## Documentation

Start with the [documentation index](docs/README.md). The main pages:

- [Architecture](docs/architecture.md): ownership, scheduler boundary,
  lifecycle, cache states, hot path, teardown and test coverage.
- [Cache and static-data API guide](docs/api_guide.md): lookup statuses, miss
  handling, saved accesses, independent caches, static GPU data, bounded waiting
  and run boundaries.
- [`graph.md`](docs/graph.md): the authoritative scheduler/lifecycle reference,
  corrected to match the reviewed real-framework callbacks.
- [Cache policies](docs/cache_policies.md), [cache statistics](docs/cache_statistics.md)
  and the [A → CPU → B result pipeline](docs/result_pipeline.md).
- [Open issues](docs/open_issues.md): real-framework integration work, future
  payload constraints and the follow-ups from the latest review.
- Design notes: [`num_of_gpu_cache_entry_issue.md`](docs/design/num_of_gpu_cache_entry_issue.md)
  (logical frame count versus cache capacity),
  [`unordered_map_vs_fixed_open_addressing.md`](docs/design/unordered_map_vs_fixed_open_addressing.md)
  (resident-key index choices) and [`gpu_mem_consumption.md`](docs/design/gpu_mem_consumption.md).
- Visual guides: [class diagram](docs/guides/gpuinfra_class_diagram.html),
  [resource plot](docs/guides/gpuinfra_resource_plot.html) and the
  [cache walkthrough](docs/guides/gpu_cache_explained.html).

Historical plans in [`docs/archive`](docs/archive/README.md) (the one-GPU cache
plan and the future multi-GPU replica extension) are kept for background only.
They may describe superseded APIs; treat the pages above as the current contract.

## Local commit and push protection

Run `git config core.hooksPath .githooks` once in each clone. The hooks reject raw
profiler files and recognizable credentials in staged content and outgoing
history. See [artifact handling](results/README.md). Raw captures stay local.

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

Every named test is its own ctest entry (`core.*` and `demo.*`), so a failure
names the test and `ctest -R cache` selects a group. The executables accept the
same names directly: `build/gpuinfra_core_tests --list` prints them and
`build/gpuinfra_core_tests lru` runs one. With no argument they run everything.

Optional host sanitizers: configure with `-DGPUINFRA_SANITIZER=thread`
(or `address`, `undefined`). ThreadSanitizer needs ASLR disabled on recent
kernels (`setarch $(uname -m) -R build/gpuinfra_core_tests`); AddressSanitizer
needs `ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0` for
CUDA to initialize. Device code is not covered; run a test under
`compute-sanitizer` instead. Use `-DGPUINFRA_BUILD_DEMO=OFF` for sanitizer
builds (the sanitizer applies to host C++ targets only).

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

For AOI integration, `-DGPUINFRA_BUILD_DEMO=OFF` builds only the C++17 host
core, without enabling CUDA-language compilation or building the demo tests. It
still builds and registers `gpuinfra_core_tests` (cache, StaticData, logger,
topology and result-cache tests) unless `-DBUILD_TESTING=OFF` is given. These
tests need a CUDA GPU and its NUMA topology but not nvcc.
See [toolchain compatibility](docs/toolchain_compatibility.md) for toolkit
selection, optional cuFFT enum detection and the remaining production matrix.

## How it fits together

Details live in [architecture](docs/architecture.md); this is the short version.

- **Topology.** `GpuContextManager::init()` discovers GPUs, maps each to its PCI
  NUMA node and retains one primary context per GPU. The framework (in the demo,
  `NumaExecutor`) sets CPU affinity before every task callback; `GpuContext` is
  the authoritative GPU-to-NUMA mapping. A task resolves exactly one local GPU in
  `load()`. `makeTaskCurrent()` is still required because CPU affinity does not
  select a CUDA device for a host thread.
- **Scheduling.** Under `DummyGraph::schedulerLock` a worker atomically claims a
  ready `FrameCpuAtom`, a free `DummyTask` and a free NUMA-local worker. The task
  stays checked out until `execute()` finishes, and cache residency never takes
  part in that decision. The call is
  `bool DummyTask::execute(FrameCpuAtom& atom, StaticData& staticData)`.
- **Logical frames versus cache.** `N` frames are owned by the graph; `K =
  GraphConfig::gpuCacheEntries` is the independent cache capacity. The default
  workload has 220 `FrameCpuAtom` objects but four entries per graph copy.

```text
StaticData
  ├─ fixed frame layout + resetCache() run boundary
  └─ GpuDataCache
       ├─ fixed open-addressing GpuDataKey -> entry index table [~2K]
       ├─ O(1) empty stack + intrusive inactive-entry LRU (FIFO optional)
       └─ GpuDataCacheEntry[K]
            cached metadata + lease/LRU links
            └─ GpuReplica[gpuId] -> persistent d_data + validity
```

  `GpuDataKey` is `frameId + cameraId + variantId`, so the same frame ID from
  different cameras cannot alias. The API is described in the
  [API guide](docs/api_guide.md).
- **Frame execution.**

```text
FrameCpuAtom metadata/layout/result validation
  -> make task GPU current
  -> StaticData::getCacheData()      CacheHit / CacheFill / TaskFallback
  -> on CacheFill / TaskFallback: atom.data -> task h_in -> selected device buffer
  -> CEL / SDD / MI read access.data()
  -> algorithm D2H staging
  -> GpuDataAccess::freeCacheData()  one cudaStreamSynchronize, publish/release
  -> copy results into FrameCpuAtom.result
```

- **Memory.** GPU input scales as cache entries × frame bytes (cache) plus task
  instances × frame bytes (fallback); see [`gpu_mem_consumption.md`](docs/design/gpu_mem_consumption.md).
  The cache is correct only for immutable input reproducible from the CPU atom.
- **Failure and teardown.** Any lifecycle or execution failure raises
  process-wide cancellation; queued frames receive terminal failed results,
  in-flight work finishes or fails, and workers join before resources are torn
  down (idempotently). A stream-synchronization failure marks the cache failed.
- **Tests.** `gpuinfra_core_tests` covers the library and `gpuinfra_demo_tests`
  the simulated framework; the coverage list is in
  [architecture §12](docs/architecture.md#12-verification).

## Source layout

```text
src/                       reusable gpuinfra library
  Context/                 discovery, primary contexts and NUMA lookup
  DataCache/               cache entries, leases, statistics and residency index
  StaticData/              graph-copy cache owner and read-only GPU regions
  Types/                   runtime layout, frame metadata and task GPU resources
  CudaCheck.h              public Runtime/Driver/cuFFT diagnostics API
  GpuDiagnostics.cpp       diagnostic implementation
  Nvtx.h                   optional profiling macros
demo/                      gpuinfra_demo_support library and demo executable
  Algo/                    IAlgo and synthetic CEL/SDD/MI algorithms
  Grape/                   simulated framework, scheduler and NUMA executor
  ResultPipeline/          A/CPU/B tasks and bounded concurrent simulation
  DummyTask.*              example caller lifecycle
  ParameterRegistry.*      simulated parameter delivery
  ImageSizing.h            synthetic image dimensions
  WorkloadSizing.h         demo-only workload multipliers
  main.cpp                 simulation entry point
tests/                     two test runners; core tests do not depend on the demo
  core/                    gpuinfra_core_tests (links only gpuinfra)
    core_tests.cpp         test invocation and infrastructure lifetime
    cache_tests.cpp        leases, waits, fallback, LRU and statistics
    cache_policy_tests.cpp LRU/FIFO ordering, Discard, completion failures
    result_cache_tests.cpp D2D handoff, CPU-backed recovery and terminal Discard
    static_data_tests.cpp  layout validation, device guard and read-only GPU data
    diagnostics_tests.cpp  logger and cuFFT integration
    topology_tests.cpp     NUMA placement and unique-GPU selection
    core_test_support.*    shared fixtures (no demo headers)
  demo/                    gpuinfra_demo_tests (links gpuinfra_demo_support)
    demo_tests.cpp         test invocation and infrastructure lifetime
    graph_tests.cpp        task/framework lifecycle and NUMA boundaries
    result_pipeline_tests.cpp A/CPU/B interleaving, recovery and policy outcomes
    demo_test_support.*    task, graph and output-verification fixtures
docs/                      documentation index and current contracts
  integration/             legacy behavior and migration comparison
  design/                  design rationale and capacity analysis
  guides/                  standalone visual HTML guides
  archive/                 historical plans (not current API contracts)
tools/                     benchmark drivers and NVTX collection/summary scripts
results/                   retained benchmark and profiling artifacts
```

Link production callers to `gpuinfra`; its public include root is `src/`.
For example, include `CudaCheck.h`, `Context/GpuContextManager.h`, or
`DataCache/GpuDataCache.h`. The core does not include demo headers.
`gpuinfra_demo_support` links the core and adds the `demo/` include root and
synthetic workload definitions. Tests and `gpuinfra_demo` link this support target.
See the [documentation index](docs/README.md) for reading order.

## Optional NVTX profiling

Build with `-DGPUINFRA_ENABLE_NVTX=ON` to enable fixed-name RAII ranges.
The default OFF build excludes NVTX and does not evaluate annotation arguments.
See [profiling instructions and recorded traces](docs/profiling.md) for
CUDA + NVTX collection, phase-separated summaries and interpretation limits.

## GPU error diagnostics

Include `CudaCheck.h` for CUDA Runtime/Driver/cuFFT checks and diagnostic scopes.
See [CUDA logger usage and output examples](docs/cuda_logging.md) for the unified
logger, explicit caller information, error handling and field definitions.
