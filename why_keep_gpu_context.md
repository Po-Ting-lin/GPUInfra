# Why keep `CUcontext primaryCtx`?

## Short answer

Keep `primaryCtx` only when GPUInfra uses it for a concrete function: Driver
API interoperability, scoped restoration of an incoming context, or exact
context-identity validation. Do not retain it merely because it may become
useful later.

Keeping the handle does **not** make kernels faster and does **not**
automatically produce more detailed CUDA error messages.

The primary context is a shared, per-device, per-process resource. Retaining
its handle gives GPUInfra a reference and an identity to validate; it does not
give GPUInfra exclusive ownership of the context.

GPU, NUMA node, task instance, graph worker, task-resource ID, stream, frame,
and algorithm attribution do not require a `CUcontext`. GPUInfra can carry
Runtime API metadata through one unified diagnostic path; NUMA comes from the
graph configuration or the authoritative GPU topology rather than duplicated
task-resource state. Exact context identity is an optional extension to that
path, not the reason to create a separate logging architecture.

## CUDA 10 through CUDA 12 compatibility

GPUInfra intends to keep its context-management design compatible with CUDA
10, CUDA 11, and CUDA 12. The context behavior relevant to this design differs
at the CUDA 12 boundary:

- In CUDA 10 and CUDA 11, `cudaSetDevice()` makes the selected device's primary
  context current to the calling thread, but Runtime initialization on that
  device remains lazy.
- Starting with CUDA 12, `cudaSetDevice()` also initializes the Runtime and the
  selected primary context.

GPUInfra may use the following common initialization sequence across all three
major versions:

```cpp
CUDA_CHECK(cudaSetDevice(gpuId), return false);
CUDA_CHECK(cudaFree(nullptr), return false);
```

`cudaFree(nullptr)` is an intentional no-op initialization call for CUDA 10 and
CUDA 11. It is redundant but valid with CUDA 12, which allows one source path
without version-dependent behavior. If the project later drops CUDA 10 and
CUDA 11, the call can be removed.

The compatibility baseline should not require APIs that are absent from CUDA
10 headers. For example, logging the opaque `CUcontext` handle and its
`CUdevice` is sufficient for the invariant described below. Newer diagnostics
such as a context ID may be added only behind an appropriate compile-time
availability check.

This context policy does not by itself guarantee that the complete project
currently builds with every Toolkit in that range. CUDA 10 `nvcc` supports
language modes only through C++14, while the current CMake configuration
requests C++17 for CUDA sources. The default `CMAKE_CUDA_ARCHITECTURES` value
must also be supported by the selected Toolkit. A real CUDA 10 build target
therefore needs a C++14-compatible CUDA source configuration, a compatible GPU
architecture, and CI coverage with CUDA 10 headers and `nvcc`.

## Benefits

### 1. Runtime and Driver API interoperability

The CUDA Runtime API normally operates on the device's primary context unless
another Driver API context is current to the calling thread. By calling
`cuDevicePrimaryCtxRetain()`, GPUInfra obtains the Driver API handle for the
shared primary context.

Runtime and Driver API operations can use resources in the same primary
context when that context is current to the calling thread:

- `cudaMalloc()` device allocations
- `cudaStreamCreateWithFlags()` streams
- Runtime API kernel launches
- Driver API modules loaded with `cuModuleLoad()`
- Driver API kernels launched with `cuLaunchKernel()`

Using the primary context avoids accidentally creating a separate context with
`cuCtxCreate()`, where resources and context state would be isolated from the
Runtime API's primary context.

Storing `primaryCtx` does not make it current automatically. Before a
current-context-dependent Driver API operation, GPUInfra must bind or validate
the expected context.

### 2. Explicit retained-reference lifetime

`cuDevicePrimaryCtxRetain()` obtains an explicit reference to the primary
context. `cuDevicePrimaryCtxRelease()` releases GPUInfra's reference during
shutdown.

This gives GPUInfra a clear participation boundary in the shared primary
context lifetime:

```text
manager init
  -> cuDevicePrimaryCtxRetain()
  -> register and initialize GPU-bound task resources
  -> process frames
manager shutdown
  -> unload every task and release streams and allocations
  -> cuDevicePrimaryCtxRelease()
```

Other Runtime users and libraries in the process may also use or retain the
same primary context. Releasing GPUInfra's reference does not imply exclusive
control of the whole context lifetime. If GPUInfra releases the last retained
reference, the primary context may be reset.

The stored handle is not ownership of every CUDA allocation. GPUInfra must
still release its streams, pinned buffers, device buffers, and algorithm
resources deterministically.

### 3. Per-assignment worker binding and validation

CUDA context state is current per host thread. The stored handle allows a Graph
worker to establish and verify the selected task's primary context.

For a graph worker that is assigned a GPU-bound task and does not need to
preserve an incoming CUDA context, use Runtime binding followed by Driver API
identity validation:

```cpp
CUDA_CHECK(cudaSetDevice(gpuId), return false);
CUDA_CHECK(cudaFree(nullptr), return false);

CUcontext current = nullptr;
CUdevice currentDevice = -1;
CU_CHECK(cuCtxGetCurrent(&current), return false);
if (current != nullptr) {
    CU_CHECK(cuCtxGetDevice(&currentDevice), return false);
}
if (current != primaryCtx || currentDevice != expectedDevice) {
    return false;
}

// Create streams, buffers, and algorithm resources.
```

Do not push `primaryCtx` after this sequence merely to bind it: `cudaSetDevice()`
has already made that primary context current.

If a Graph worker may arrive with another context current and GPUInfra must
restore it afterward, use a true scoped push/pop instead:

```cpp
CU_CHECK(cuCtxPushCurrent(primaryCtx), return false);
CUDA_CHECK(cudaFree(nullptr), return false);

// Perform Runtime and Driver API work in primaryCtx.

CUcontext popped = nullptr;
CU_CHECK(cuCtxPopCurrent(&popped), return false);
if (popped != primaryCtx) {
    return false;
}
```

`cuCtxPopCurrent()` requires storage for the popped handle. Passing `nullptr`
is not part of its documented contract. A production scoped-context guard
should remember the expected context, verify the popped handle, and log rather
than throw if pop fails in its destructor.

The current `makeTaskCurrent()` implementation uses `cudaSetDevice()` for each
assignment. It retains `primaryCtx` for process lifetime but does not push,
compare, or log the handle. Exact identity validation would therefore be an
additional feature rather than a description of current behavior.

Context validation belongs at ownership and assignment boundaries, not before
every kernel launch. GPUInfra already calls `cudaSetDevice()` when a worker
acquires a task. If same-thread third-party code may install a different
context, exact identity may also be checked at that boundary. Because task
assignment is part of the measured frame path, the extra Driver API query
should be benchmarked and may instead be enabled for diagnostic builds or
performed after an error.

CUDA current-context state is per host thread. Another worker thread cannot
directly change it; the relevant risk is other code or a third-party library
running on the same worker thread.

### 4. Multi-GPU identity and invariant checking

Each `GpuContext` records an expected `(gpuId, primaryCtx)` pair. This can
detect a worker accidentally using another GPU or another CUDA context:

```cpp
CUcontext current = nullptr;
CUdevice currentDevice = -1;
CU_CHECK(cuCtxGetCurrent(&current), return false);
if (current != nullptr) {
    CU_CHECK(cuCtxGetDevice(&currentDevice), return false);
}

if (current != primaryCtx || currentDevice != expectedDevice) {
    // The worker is bound to the wrong context.
}
```

For the normal primary-context-only design, `cudaGetDevice()` is sufficient to
detect that a worker is using the wrong GPU. The additional `CUcontext`
comparison detects the narrower case where the worker is on the expected GPU
but a non-primary context is current, for example because same-thread
third-party code created or installed its own context.

NUMA locality is not a CUDA-context property. It must be diagnosed separately
from graph-worker CPU affinity, the graph copy's NUMA node, and the selected
task GPU's authoritative `GpuContext::numaNode` mapping.

### 5. Better attribution in custom logs

Error attribution should start with one unified, allocation-free diagnostic
record shared by Runtime and Driver API checks:

```cpp
struct CudaDiagnosticContext {
    int gpuId = -1;
    int numaNode = -1;
    int taskInstanceId = -1;
    int resourceId = -1;
    cudaStream_t stream = nullptr;
    std::uint64_t frameId = 0;
    const char* algorithm = nullptr;
    CUcontext expectedContext = nullptr;  // Optional.
};
```

`CudaCheck.h` should consume this metadata through the same path for all CUDA
failures. GPU, NUMA, task, worker, resource, stream, frame, and algorithm fields
remain useful in a Runtime-only build. When exact context validation is enabled,
`expectedContext` and a best-effort `cuCtxGetCurrent()` result can be appended
to the same record:

```text
frame=42 gpu=0 numa=0 task=2 resource=2 algorithm=CEL
expected_context=0x1234 current_context=0x1234 stream=0x5678
call=cudaStreamSynchronize(stream)
error=cudaErrorIllegalAddress
```

This can make a multi-GPU failure easier to attribute. The improvement comes
from GPUInfra carrying task and resource metadata into the error path, not from
retaining the context by itself. Context identity is an additional field only
when the application has a credible non-primary-context risk.

### 6. Supports future Driver API features

Keeping the handle is useful if the implementation later adds:

- `cuModuleLoad()` and `cuModuleGetFunction()`
- `cuLaunchKernel()`
- Driver API memory operations
- context-level configuration or inspection
- integration with a library that accepts a `CUcontext`

No context-model redesign is then required. However, hypothetical future use
alone is not enough to justify retaining unused push/pop and reference-lifetime
code today.

## What keeping `primaryCtx` does not help

### 1. It does not automatically improve CUDA error messages

Runtime errors still provide `cudaError_t`, `cudaGetErrorName()`, and
`cudaGetErrorString()`. Driver errors still provide `CUresult`,
`cuGetErrorName()`, and `cuGetErrorString()`.

The retained context does not add a stack trace, frame ID, algorithm name, or
kernel name. GPUInfra must add that metadata to its own logs.

The current `CudaCheck.h` helper reports the failed call, error name,
description, file, and line. It does not currently receive `primaryCtx` or a
host-thread identifier, so retaining the handle does not make its error text
richer yet.

### 2. It does not identify the exact asynchronous failure

CUDA kernel execution is asynchronous. A kernel launch can return success and
an illegal memory access may only surface later at `cudaStreamSynchronize()`.

The context can identify which GPU/context failed, but it cannot automatically
identify which earlier kernel caused the failure. Use the following when
debugging that class of problem:

- per-algorithm log or NVTX ranges
- launch-time `cudaPeekAtLastError()`/`cudaGetLastError()` checks
- `compute-sanitizer`
- CUDA line information in development builds
- `CUDA_LAUNCH_BLOCKING=1` for diagnosis only

### 3. It does not improve kernel or transfer performance

Retaining a primary-context handle does not increase SM utilization, PCIe
bandwidth, kernel concurrency, or copy-engine concurrency.

Performance comes from the stream and pipeline design: multiple task-owned
streams, best-effort cache hits that avoid repeated H2D, contiguous kernel/D2H
submission, preallocated cache and fallback buffers, and one synchronization
per `execute()` call.

### 4. It does not replace `cudaSetDevice()`

Each Graph worker still needs to establish the selected task's Runtime API
device for every assignment. Keeping a `CUcontext` member does not automatically
make that context current on another host thread.

### 5. It does not make shared state thread-safe

A context may be used by multiple host threads, but application data still
needs correct ownership. GPUInfra gives every `DummyTask` a distinct stream,
pinned input staging, persistent fallback `d_input`, scratch buffer, and
private algorithm objects. Graph-copy-scoped `StaticData` separately owns
an immutable registered frame-ID-to-metadata hash and a bounded `GpuCacheManager`;
every reusable `GpuCacheEntry` directly owns its persistent `GpuReplica`
allocations, while scoped `GpuDataAccess` objects coordinate cache hits, fills,
and task fallback. The
graph scheduler owns frame execution state, and its exclusive checkout prevents
concurrent use of one task instance while permitting sequential movement
between host threads. Cache
access counts independently prevent eviction of an active cache entry.

The context handle does not prevent races in task registration, future per-GPU
shared resources, or mutable task state. Those remain application-level
invariants.

### 6. It does not replace deterministic resource cleanup

Releasing GPUInfra's retained reference must not be used as its normal resource
cleanup mechanism. If the last retained reference is released, the primary
context may be reset and context-owned resources destroyed, but GPUInfra must
still explicitly release its own resources first:

- `cudaStreamDestroy()`
- `cudaFree()`
- `cudaFreeHost()`
- each algorithm's `close()`

### 7. It does not provide fault isolation

All task instances on one GPU intentionally share its primary context. A fatal
device error or illegal access can therefore affect other work using that
context. A stored context handle does not provide process isolation or
per-task fault containment.

## Shutdown requirements

`cuDevicePrimaryCtxRelease()` does not pop the primary context from any host
thread's context stack. Before GPUInfra releases its reference, it must ensure:

1. all Graph workers have stopped using tasks bound to the context;
2. every GPUInfra push has a matching pop;
3. all asynchronous work needed for teardown has completed;
4. streams, events, modules, device buffers, pinned buffers, and algorithm
   resources owned by GPUInfra have been released;
5. no GPUInfra-scoped context binding remains current on a worker thread.

The demo joins graph workers, releases cache-entry device data, and unloads every
task (including its fallback device buffer) before
`GpuContextManager::shutdown()`, which satisfies the first requirement. The
manager rejects shutdown while registered task resources remain active.

## Decision rule

This is not a decision about whether CUDA has a context. The Runtime API always
uses a context. The decision is whether GPUInfra's explicit Driver API context
code performs a real function that justifies its maintenance cost.

Use the following order:

1. First connect GPU, graph NUMA node, task, worker, resource, stream, frame,
   and algorithm metadata to the shared CUDA error path. This is valuable in
   every multi-GPU deployment and requires only Runtime API, graph topology,
   and existing task-resource state.
2. Add `CUcontext` identity to that same diagnostic record only when GPUInfra
   must detect or restore a non-primary context on the same worker thread.
3. Validate context identity at task registration or assignment boundaries,
   not before every kernel launch. Use diagnostic-build or error-path
   validation when an unconditional per-assignment query is too expensive.

Keep `primaryCtx` when at least one concrete requirement exists:

- GPUInfra performs Driver API operations that require the expected context;
- a same-thread third-party library may install a non-primary context;
- GPUInfra must preserve and restore an incoming context;
- an observed context-mismatch debugging problem justifies exact identity
  checks.

Remove `primaryCtx` and its retain/release pair when the implementation is
Runtime-API-only, workers use `cudaSetDevice()` per task assignment, and there
is no credible non-primary-context scenario. Wrong-GPU and NUMA diagnostics
remain available from Runtime API, `GraphConfig`, and the authoritative GPU
topology mapping.

The current intermediate state—retaining `primaryCtx` without reading it,
comparing it, logging it, or making decisions from it—is maintenance debt. It
suggests a context invariant that the implementation does not actually enforce.
GPUInfra should either implement context identity as the specific feature
described above or remove the unused Driver context layer.

## NVIDIA references

- [CUDA 10.0 Runtime and Driver API interactions](https://docs.nvidia.com/cuda/archive/10.0/cuda-runtime-api/group__CUDART__DRIVER.html)
- [CUDA 10.0 NVCC supported C++ dialects](https://docs.nvidia.com/cuda/archive/10.0/cuda-compiler-driver-nvcc/index.html)
- [CUDA 11.8 Runtime and Driver API interactions](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-runtime-api/group__CUDART__DRIVER.html)
- [CUDA 11.8 primary-context management](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-driver-api/group__CUDA__PRIMARY__CTX.html)
- [CUDA 11.8 context-stack management](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-driver-api/group__CUDA__CTX.html)
- [CUDA 12.0 Runtime and Driver API interactions](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-runtime-api/group__CUDART__DRIVER.html)
