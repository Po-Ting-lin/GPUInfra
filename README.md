# GPUInfra CUDA demo

This repository is a corrected, runnable version of the OCR-derived GPU
infrastructure example in [plan.md](plan.md).

It demonstrates the register-thread/own-the-slot model:

- one `GpuContext` per discovered CUDA GPU;
- one nonblocking CUDA stream and one fixed slot per Graph worker;
- one shared H2D input transfer per frame;
- a compute batch for all algorithms, followed by a D2H batch;
- exactly one `cudaStreamSynchronize()` per frame;
- serialized algorithm configuration before workers start;
- no internal queue, dispatcher, or GPUInfra-owned worker threads.

The example currently registers two synthetic algorithms, CEL and SDD. Each
launches a real CUDA kernel that scans the full input and performs repeated
integer mixing plus a block reduction. These kernels create measurable GPU
load but do not implement OCR. Real CEL/SDD kernels can replace them without
changing the infrastructure or `IAlgo` lifecycle.

## Requirements

- Linux with NUMA topology exposed under `/sys/devices/system/node`
- CMake 3.24 or newer
- A C++17 compiler
- NVIDIA CUDA Toolkit
- A compatible NVIDIA driver and at least one CUDA GPU

`libnuma-dev` is not required. Thread affinity is implemented with pthreads
and Linux NUMA sysfs.

The default CUDA architecture is `sm_86` for RTX 3080. Override it for another
GPU with either:

```bash
CUDA_ARCHITECTURES=89 ./build.sh
```

or:

```bash
./build.sh -DCMAKE_CUDA_ARCHITECTURES=89
```

## Build

`build.sh` always creates a Release build:

```bash
./build.sh
```

Equivalent commands:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

The project also writes `build/compile_commands.json` for VS Code IntelliSense.

## Run and test

Run the example directly:

```bash
./build/gpuinfra_demo
```

Run it through CTest:

```bash
ctest --test-dir build --output-on-failure
```

The verified output on the current RTX 3080 system ends with:

```text
[GPUInfra] shutdown complete
delivered 24 frames across 1 CUDA GPU(s), failures=0
```

The demo discovers the GPU count dynamically. It does not assume the two-GPU
deployment described in the plan.

## Lifecycle

### 1. Manager initialization

`GpuContextManager::init()`:

1. calls `cuInit()` and discovers devices;
2. probes each GPU's PCI NUMA node through sysfs;
3. pins the startup thread before first-touch allocation;
4. calls `cudaSetDevice()` and primes the Runtime API with
   `cudaFree(nullptr)`;
5. retains the GPU's primary context with
   `cuDevicePrimaryCtxRetain()`;
6. creates one CEL and one SDD instance per GPU.

### 2. Serialized configuration

`GpuContextManager::configure()` runs before any worker registers. It allocates
all per-GPU, per-thread algorithm output buffers.

Configuration does not run from `DummyGraph::notifyParameters()`. The OCR
version configured the same algorithm instance concurrently from every Graph
worker, which could reallocate buffers while another worker was using them.

### 3. Worker registration

Each external Graph worker calls:

```cpp
ThreadSlot* slot =
    GpuContextManager::registerThread(numaNode, gpuId);
```

Registration creates:

- one `cudaStreamNonBlocking` stream;
- one `cudaHostAlloc()` input buffer;
- one `cudaMalloc()` device input buffer;
- optional per-slot shared device scratch.

Slot IDs come from a fixed table and are reused without renumbering active
workers. This keeps `slot.threadId` aligned with every algorithm's
`perThread[]` state.

### 4. Frame execution

`DummyGraph::execute()` records this chain:

```text
caller buffer
  -> pinned input memcpy
  -> one cudaMemcpyAsync H2D
  -> CEL synthetic CUDA kernel
  -> SDD synthetic CUDA kernel
  -> CEL cudaMemcpyAsync D2H
  -> SDD cudaMemcpyAsync D2H
  -> one cudaStreamSynchronize
  -> collect owned results
```

There is no per-frame CUDA allocation and no `cudaDeviceSynchronize()`.

### 5. Teardown

Workers unregister from their owning thread. GPUInfra synchronizes and destroys
the slot stream, frees slot memory, closes algorithm allocations while the
correct primary context is current, and finally calls
`cuDevicePrimaryCtxRelease()`.

## CUDA API split

| Responsibility | API |
| --- | --- |
| Driver initialization | `cuInit` |
| Primary-context ownership | `cuDevicePrimaryCtxRetain`, `cuCtxPushCurrent`, `cuCtxPopCurrent`, `cuDevicePrimaryCtxRelease` |
| Runtime device binding | `cudaSetDevice`, `cudaFree(nullptr)` |
| Slot resources | `cudaStreamCreateWithFlags`, `cudaHostAlloc`, `cudaMalloc` |
| Frame transfers | `cudaMemcpyAsync` |
| Per-frame wait | `cudaStreamSynchronize` |
| Cleanup | `cudaFree`, `cudaFreeHost`, `cudaStreamDestroy` |

CUDA errors report the API family, expression, error name, numeric code,
description, source file, and line.

## Source layout

```text
src/
  CudaCheck.h              CUDA Runtime/Driver error reporting
  GpuContext.h             per-GPU registry and fixed slot table
  GpuContextManager.*      discovery, configuration, registration, teardown
  ThreadSlot.h             per-Graph-thread CUDA resources
  IAlgo.h                  split compute/D2H algorithm interface
  Cel.*                    synthetic CEL CUDA kernel
  Sdd.*                    synthetic SDD CUDA kernel
  GraphStuff.h             external Graph framework stand-ins
  main.cpp                 multi-threaded runnable example
```
