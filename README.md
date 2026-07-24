# GPUInfra CUDA demo

This repository is a corrected, runnable version of the OCR-derived GPU
infrastructure example in [plan.md](plan.md).

It demonstrates the register-thread/own-the-slot model:

- one `GpuContext` per discovered CUDA GPU;
- one nonblocking CUDA stream and one fixed slot per Graph worker;
- one shared H2D input transfer per frame;
- one pinned host result matrix per algorithm and worker slot;
- a compute batch for all algorithms, followed by a D2H batch;
- exactly one `cudaStreamSynchronize()` per frame;
- serialized algorithm configuration before workers start;
- no internal queue, dispatcher, or GPUInfra-owned worker threads.

The example currently registers three synthetic algorithms: CEL, SDD, and MI.
Each launches a real CUDA matrix-multiplication kernel over a shared square byte
input frame. For size factor `F`, the input is `(8F)x(8F)`, CEL operates on the
top-left `(2F)x(2F)` region, and SDD and MI each operate on the top-left
`(3F)x(3F)` region. Results are returned as row-major `uint32_t` matrices in
`AlgoOutput::data`. These kernels create measurable GPU load but do not
implement OCR. Real CEL/SDD/MI kernels can replace them without changing the
infrastructure or `IAlgo` lifecycle.

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

The optional arguments select timed and warmup frames per GPU, followed by the
execution model and size factor:

```bash
./build/gpuinfra_demo 200 20 batched 128
./build/gpuinfra_demo 200 20 interleaved 128
```

The defaults are 200 timed frames, 20 warmup frames per GPU, and the `batched`
execution model. The size factor defaults to 128, must be a multiple of 16, and
must be between 16 and 256 inclusive. `batched` enqueues every algorithm kernel
before enqueueing all D2H transfers. `interleaved` enqueues each algorithm
kernel immediately followed by its D2H transfer on the same stream. All workers
finish setup and warmup before the timer starts. Timing stops after every worker
finishes its measured frames and before graph teardown. The report includes
aggregate frames per second and average milliseconds per frame. Benchmark
results are collected through the normal D2H and host-copy path but are not
retained by the demo sink, avoiding memory growth for large frame counts.

Run it through CTest:

```bash
ctest --test-dir build --output-on-failure
```

Sweep every valid size factor for both execution models with five repetitions:

```bash
python3 scripts/benchmark_model_factor.py
```

The script benchmarks factors 16 through 256 in steps of 16 using 200 timed
frames and 20 warmup frames. It alternates model order, randomizes factor order,
stores raw and median/IQR CSV data in a timestamped `benchmark_results/`
directory, saves linear and logarithmic PNG figures, and calls
`matplotlib.pyplot.show()` for both figures. Use `--no-show` in a headless
environment.

The output ends with:

```text
[GPUInfra] shutdown complete
execution_model=batched
size_factor=128 input=1024x1024 cel=256x256 sdd=384x384 mi=384x384
warmup 20 frames/GPU, timed 200 frames/GPU (200 total) in ... ms
throughput=... frames/s, average=... ms/frame
delivered 220 frames across 1 CUDA GPU(s), failures=0
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
6. creates one CEL, one SDD, and one MI instance per GPU.

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

During `DummyGraph::load()`, each worker also creates one reusable `JobResult`,
sizes its algorithm-output table, and allocates the CEL, SDD, and MI pageable
result buffers. The frame-execution path only copies into this prepared storage.
The demo sink consumes each result by reference and does not retain it.

### 4. Frame execution

`DummyGraph::execute()` records this chain:

```text
caller buffer
  -> pinned input memcpy
  -> one cudaMemcpyAsync H2D
  -> CEL (2F)x(2F) matrix-multiplication CUDA kernel
  -> SDD (3F)x(3F) matrix-multiplication CUDA kernel
  -> MI (3F)x(3F) matrix-multiplication CUDA kernel
  -> CEL cudaMemcpyAsync D2H into a pinned result matrix
  -> SDD cudaMemcpyAsync D2H into a pinned result matrix
  -> MI cudaMemcpyAsync D2H into a pinned result matrix
  -> one cudaStreamSynchronize
  -> collect owned results
```

There is no per-frame CUDA allocation and no `cudaDeviceSynchronize()`.
There is also no per-frame host allocation, vector growth, or result ownership
transfer.

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
  Cel.*                    synthetic CEL matrix-multiplication kernel
  Mi.*                     synthetic MI matrix-multiplication kernel
  Sdd.*                    synthetic SDD matrix-multiplication kernel
  GraphStuff.h             external Graph framework stand-ins
  main.cpp                 multi-threaded runnable example
```
