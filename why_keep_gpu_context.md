# Why keep `CUcontext primaryContext_`?

## Short answer

Keep `primaryContext_` when GPUInfra intentionally mixes the CUDA Driver API
and CUDA Runtime API, needs explicit per-GPU context lifetime, or wants to
validate which context an external Graph worker is using.

Keeping the handle does **not** make kernels faster and does **not**
automatically produce more detailed CUDA error messages.

## Benefits

### 1. Runtime API and Driver API use the same context

The CUDA Runtime API normally operates on the device's primary context. By
calling `cuDevicePrimaryCtxRetain()`, GPUInfra obtains the Driver API handle for
that same context.

Consequently, resources created through the Runtime API can coexist with
future Driver API work in the same context:

- `cudaMalloc()` device allocations
- `cudaStreamCreateWithFlags()` streams
- Runtime API kernel launches
- Driver API modules loaded with `cuModuleLoad()`
- Driver API kernels launched with `cuLaunchKernel()`

Using the primary context avoids accidentally creating a separate context with
`cuCtxCreate()`, where resources and context state would be isolated from the
Runtime API's primary context.

### 2. Explicit per-GPU context lifetime

`cuDevicePrimaryCtxRetain()` obtains an explicit reference to the primary
context. `cuDevicePrimaryCtxRelease()` releases GPUInfra's reference during
shutdown.

This gives the infrastructure a clear lifetime boundary:

```text
manager init
  -> cuDevicePrimaryCtxRetain()
  -> configure algorithms and register Graph workers
  -> process frames
manager shutdown
  -> release streams and allocations
  -> cuDevicePrimaryCtxRelease()
```

The stored handle is not ownership of every CUDA allocation. Streams, pinned
buffers, device buffers, and algorithm resources must still be released
separately.

### 3. Explicit Graph-worker context binding

CUDA context state is current per host thread. The stored handle allows a Graph
worker to explicitly enter the selected GPU's context during cold-path setup:

```cpp
cudaSetDevice(gpuId);
cudaFree(nullptr);                 // prime Runtime state
cuCtxPushCurrent(primaryContext_);
// create stream and buffers
cuCtxPopCurrent(nullptr);
```

This is useful because Graph workers are created outside GPUInfra. The handle
provides an explicit boundary for registration and teardown without creating
an internal worker thread.

### 4. Multi-GPU identity and invariant checking

Each `GpuContext` records an expected `(gpuId, primaryContext_)` pair. This can
detect a worker accidentally using another GPU or another CUDA context:

```cpp
CUcontext current = nullptr;
cuCtxGetCurrent(&current);

if (current != primaryContext_) {
    // The worker is bound to the wrong context.
}
```

This is more precise than checking only `cudaGetDevice()`, particularly when
Driver API code or third-party CUDA libraries may manipulate the context
stack.

### 5. Better attribution in custom logs

The handle can be included in a structured error log alongside the expected
GPU, current context, stream, slot, frame, and algorithm:

```text
frame=42 gpu=0 slot=2 algorithm=CEL
expected_context=0x1234 current_context=0x1234 stream=0x5678
call=cudaStreamSynchronize(stream)
error=cudaErrorIllegalAddress
```

This can make a multi-GPU failure easier to attribute. The improvement comes
from GPUInfra explicitly collecting and logging the context information, not
from retaining the context by itself.

### 6. Supports future Driver API features

Keeping the handle is useful if the implementation later adds:

- `cuModuleLoad()` and `cuModuleGetFunction()`
- `cuLaunchKernel()`
- Driver API memory operations
- context-level configuration or inspection
- integration with a library that accepts a `CUcontext`

No context-model redesign is then required.

## What keeping `primaryContext_` does not help

### 1. It does not automatically improve CUDA error messages

Runtime errors still provide `cudaError_t`, `cudaGetErrorName()`, and
`cudaGetErrorString()`. Driver errors still provide `CUresult`,
`cuGetErrorName()`, and `cuGetErrorString()`.

The retained context does not add a stack trace, frame ID, algorithm name, or
kernel name. GPUInfra must add that metadata to its own logs.

The current `CudaUtils` error helper reports the failed call, error name,
description, file, line, and host thread. It does not currently receive
`primaryContext_`, so retaining the handle does not make its error text richer
yet.

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

Performance comes from the stream and pipeline design: multiple Graph-owned
streams, one shared H2D, a contiguous kernel batch, a contiguous D2H batch,
preallocated buffers, and one synchronization per frame.

### 4. It does not replace `cudaSetDevice()`

Each Graph worker still needs to establish the selected Runtime API device.
Keeping a `CUcontext` member does not automatically make that context current
on another host thread.

### 5. It does not make shared state thread-safe

A context may be used by multiple host threads, but application data still
needs correct ownership. GPUInfra relies on each thread having a distinct
slot, stream, scratch buffer, and per-algorithm output entry.

The context handle does not prevent races in `configureAndAlloc()`, slot
registration, or algorithm-global mutable state.

### 6. It does not clean up resources automatically

Releasing the primary-context reference is not a substitute for explicitly
destroying streams and freeing memory. GPUInfra must still call:

- `cudaStreamDestroy()`
- `cudaFree()`
- `cudaFreeHost()`
- each algorithm's `close()`

### 7. It does not provide fault isolation

All Graph workers on one GPU intentionally share its primary context. A fatal
device error or illegal access can therefore affect other work using that
context. A stored context handle does not provide process isolation or
per-thread fault containment.

## When it can be removed

`primaryContext_` can be removed if GPUInfra becomes Runtime-API-only and does
not need explicit context validation or any Driver API feature. In that design,
`cudaSetDevice()` is normally sufficient to select the Runtime API's primary
context.

For the current design, retaining it is reasonable because the infrastructure
explicitly uses both APIs and models one process-wide primary context per GPU.
Its strongest benefits are interoperability, explicit lifetime, worker-context
validation, and future Driver API integration—not performance or richer error
strings.
