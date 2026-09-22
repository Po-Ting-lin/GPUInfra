# CUDA error logging

CUDA Runtime/Driver error logs retain the original code, name, message and
source location, and append known resource identity: PID, bound GPU, NUMA,
task resource ID, context handle, Driver context ID, stream, frame/camera key,
and operation stage. Lexical thread-local scopes carry task identity through
H2D, algorithm calls and cache completion, and restore it on exit. They perform
no GPU queries, synchronization or allocations on the hot path.

Driver context ID is queried once during context initialization where supported;
an unavailable ID is labeled unknown in errors. It is not a profiler context ID.
Identity describes the bound resource, not a fresh observation of the driver's
current device. Outside an annotated scope, unavailable numbers are -1, pointers
are null, and frame/Driver IDs are unknown. Direct cache callers still supply
GPU, stream and frame identity without a task context. Cache rejection logs
explain the failure; normal timeout fallback does not emit an error.

Profiler-ID mapping is in [the backlog](open_issues.md#profiler-context-id-mapping-backlog).
This change enriches diagnostics without changing CacheStatus or bool completion
APIs; structured error-return API design remains separate.

## Unified error logging

`CUDA_CHECK(call, onError)` and
`GPUINFRA_REPORT_FAILURE(reason)` share the single `logGpuError()` output
implementation. CUDA macros retain their return/error handling and evaluate
the CUDA expression once; successful calls do not log. All failures include
file/line, operation, error source, name/code/message and known resource identity.
Application errors use `code=none`, never a fabricated CUDA error.

```cpp
const GpuDiagnosticScope scope(resources, "MyTask.execute", &key);
CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, resources.stream), return false);
GPUINFRA_REPORT_FAILURE("invalid algorithm input");
```

A caller without a scope can pass an explicit snapshot:

```cpp
GpuDiagnosticInfo info;
info.gpuId = gpuId;
info.stream = stream;
info.stage = "ExternalCaller";
GPUINFRA_REPORT_FAILURE_WITH_INFO("invalid input", info);
```

Explicit logging does not change thread-local scope state. The equivalent
`reportGpuFailure(reason, info, file, line)` function remains available.
`GpuErrorRecord` plus `logGpuError(error, info)` is the common lower-level
entrypoint. Output currently goes to stderr; a production logger adapter can
replace this one output implementation. Scope stage strings must outlive their
scope. Resource scopes are snapshots: establish a fresh scope after creating or
changing a stream, as done by `Task.load.buffers`.

## One include for new GPU algorithms

Include `CudaCheck.h` and link the CMake `gpuinfra` target. It exports the
CUDA Runtime, CUDA Driver and cuFFT dependencies. Diagnostic scopes, explicit
error records and application-error macros are all declared in that header;
their shared implementation stays in GpuDiagnostics.cpp. The old separate
diagnostics header and Driver-only check macro have been removed.

```cpp
#include "CudaCheck.h"

// Inside a function, with the appropriate resource lifetime/cleanup:
CUDA_CHECK(cudaSetDevice(gpuId), return false);
CUDA_CHECK(cuCtxGetCurrent(&context), return false);
CUDA_CHECK(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD), return false);
// Immediately after a custom kernel launch:
CUDA_CHECK(cudaGetLastError(), return false);
```

Overloads accept `cudaError_t`, `CUresult` and `cufftResult`; plain integers
and unrelated error types are not accepted. Each retains its own error source
and numeric code. cuFFT uses a status-name mapping, including an unknown-code
fallback, because no CUDA Runtime error-string conversion applies to it.

The legacy cudaHelper.h inspired typed dispatch and cuFFT status names.
Its process-exiting checks, automatic GPU selection, unrelated library helpers,
and helperString.h CLI/file utilities are deliberately not imported. Caller
error handling and graph NUMA/GPU selection stay unchanged.

## Output examples

The following are illustrative logs, not captured failures. Addresses, IDs,
file names and line numbers are examples; CUDA messages may vary by version.
All entries go to stderr as a single line.

### CUDA Runtime failure

Inside a diagnostic scope:

```cpp
const GpuDiagnosticScope scope(resources, "MyAlgo.allocate", &key);
CUDA_CHECK(cudaMalloc(&d_data, bytes), return false);
```

Example allocation failure:

```text
[CUDA Runtime] MyAlgo.cu:120 operation=cudaMalloc(&d_data, bytes) error=cudaErrorMemoryAllocation code=2 message=out of memory pid=12345 bound_gpu=0 numa=0 resource=2 context_handle=0x1234 driver_context_id=1 stream=0x5678 frame=100 camera=3 stage=MyAlgo.allocate
```

The macro calls the cudaError_t overload of reportCudaError(). On failure it
logs once and executes the supplied onError statement; it does not free earlier
allocations or terminate the process automatically. Caller cleanup remains
required. Successful calls do not print.

### CUDA Driver failure

```cpp
CUDA_CHECK(cuCtxGetCurrent(&context), return false);
```

If called before driver initialization, an illustrative output without an
established diagnostic scope is:

```text
[CUDA Driver] MyAlgo.cu:125 operation=cuCtxGetCurrent(&context) error=CUDA_ERROR_NOT_INITIALIZED code=3 message=initialization error pid=12345 bound_gpu=-1 numa=-1 resource=-1 context_handle=(nil) driver_context_id=unknown stream=(nil) frame=unknown camera=unknown stage=unknown
```

### cuFFT failure

```cpp
CUDA_CHECK(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD), return false);
```

Example inside a scope named MyAlgo.fft:

```text
[cuFFT] MyAlgo.cu:140 operation=cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD) error=CUFFT_INVALID_PLAN code=1 message=CUFFT_INVALID_PLAN pid=12345 bound_gpu=0 numa=0 resource=2 context_handle=0x1234 driver_context_id=1 stream=0x5678 frame=100 camera=3 stage=MyAlgo.fft
```

cuFFT message repeats the mapped status name; unknown values retain their numeric
code and use CUFFT_UNKNOWN_ERROR. Runtime, Driver and cuFFT numeric codes belong
to separate namespaces, even when their numbers happen to match.

### Application failure

```cpp
const GpuDiagnosticScope scope(resources, "MyAlgo.execute", &key);
if (bytes == 0) {
    GPUINFRA_REPORT_FAILURE("invalid input size");
    return false;
}
```

```text
[GPUInfra] MyAlgo.cu:150 operation=MyAlgo.execute error=Failure code=none message=invalid input size pid=12345 bound_gpu=0 numa=0 resource=2 context_handle=0x1234 driver_context_id=1 stream=0x5678 frame=100 camera=3 stage=MyAlgo.execute
```

This macro calls reportGpuFailure(). It always logs and does not return failure
or change control flow for the caller. Its operation is the diagnostic stage,
whereas reportCudaError() records the checked API expression.

Direct reportGpuFailure("reason") calls default to unknown:0 for source location.
Prefer the macro, or pass file/line explicitly. The explicit-info variant is:

```cpp
GpuDiagnosticInfo info;
info.gpuId = gpuId;
info.stream = stream;
info.stage = "ExternalCaller";
GPUINFRA_REPORT_FAILURE_WITH_INFO("invalid input", info);
```

## Diagnostic field reference

| Field | Meaning |
| --- | --- |
| Prefix | GPUInfra application error, CUDA Runtime, CUDA Driver, or cuFFT |
| file:line | Macro call site; unknown:0 when omitted from a direct application report |
| operation | Checked API expression, or application diagnostic stage |
| error / code / message | Error name, source-specific numeric code and explanation; application code is none |
| pid | Process emitting the log |
| bound_gpu | Known resource GPU logical ID, not a fresh current-device query |
| numa | Known NUMA node |
| resource | Registered task resource ID; not globally unique across GPUs |
| context_handle | Known CUcontext handle; null when unavailable |
| driver_context_id | Cached Driver ID; not guaranteed to equal profiler contextId |
| stream | Known CUDA stream handle; null when unavailable |
| frame / camera | Key attached to the scope/request; unknown when not supplied |
| stage | Caller-provided operation stage |

The scope is thread-local and does not propagate to another thread. It restores
previous identity on destruction, including early returns. Pass an explicit
snapshot or establish a new scope on another thread. Stage strings are borrowed
and must remain valid while used. Missing context information is not guessed.
Normal cache timeout fallback is a strategy outcome, not an error log.
