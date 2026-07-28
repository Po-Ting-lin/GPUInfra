# GPUInfra CUDA demo

This repository is a corrected, runnable version of the OCR-derived GPU
infrastructure example in [architecture.md](architecture.md).

It demonstrates the register-thread/own-the-slot model:

- one `GpuContext` per discovered CUDA GPU;
- one nonblocking CUDA stream and one fixed slot per Graph worker;
- one private CEL, SDD, and MI object per `DummyGraph`;
- one shared H2D input transfer per frame;
- one device output and one pinned-host output buffer owned by each algorithm;
- a compute batch for all algorithms, followed by a D2H batch;
- exactly one `cudaStreamSynchronize()` per frame;
- worker-local algorithm configuration before warmup and timing;
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
deployment illustrated by the architecture diagrams.

## Lifecycle

### 1. Manager initialization

`GpuContextManager::init()`:

1. calls `cuInit()` and discovers devices;
2. probes each GPU's PCI NUMA node through sysfs;
3. pins the startup thread before first-touch allocation;
4. calls `cudaSetDevice()` and primes the Runtime API with
   `cudaFree(nullptr)`;
5. retains the GPU's primary context with
   `cuDevicePrimaryCtxRetain()`.

The manager creates no algorithm objects. `GpuContext` owns the per-GPU
context identity and fixed slot table. It is also the intended owner for future
large immutable resources that must be shared by every worker on that GPU.

### 2. Runtime validation

`GpuContextManager::configure()` runs before workers register. It validates the
frame geometry and input-byte count and marks each context ready for
registration. It does not allocate algorithm resources.

### 3. Worker registration

Each external Graph worker calls:

```cpp
ThreadSlot* slot =
    GpuContextManager::registerThread(numaNode, gpuId);
```

Registration creates:

- one `cudaStreamNonBlocking` stream;
- one `cudaHostAlloc()` input buffer;
- one `cudaMalloc()` device input buffer.

Slot IDs come from a fixed table and are reused without renumbering active
workers. Each slot remains owned by its registering Graph thread.

During `DummyGraph::load()`, each worker:

1. creates its private CEL, SDD, and MI objects;
2. configures their output dimensions;
3. asks the manager to allocate optional shared scratch in the `ThreadSlot`;
4. asks each algorithm to allocate its own device and pinned-host output
   buffers;
5. creates one reusable `JobResult` and its pageable result matrices.

Output allocation is a cold-path operation after NUMA and GPU binding. Separate
algorithm-owned buffers keep every output valid until its Batched D2H copy.
Shared scratch may be reused because kernels on one slot execute in stream
order.

### 4. Frame execution

`DummyGraph::execute()` records this chain:

```text
caller buffer
  -> pinned input memcpy
  -> one cudaMemcpyAsync H2D
  -> CEL (2F)x(2F) matrix-multiplication CUDA kernel
  -> SDD (3F)x(3F) matrix-multiplication CUDA kernel
  -> MI (3F)x(3F) matrix-multiplication CUDA kernel
  -> CEL cudaMemcpyAsync D2H into its pinned output buffer
  -> SDD cudaMemcpyAsync D2H into its pinned output buffer
  -> MI cudaMemcpyAsync D2H into its pinned output buffer
  -> one cudaStreamSynchronize
  -> collect owned results
```

There is no per-frame CUDA allocation and no `cudaDeviceSynchronize()`.
There is also no per-frame host allocation, vector growth, or result ownership
transfer.

### 5. Teardown

Each `DummyGraph` synchronizes its stream, closes and destroys its private
algorithm objects and their output buffers, and then unregisters its slot.
GPUInfra frees the slot input, scratch, and stream while the correct primary
context is current. After all workers have joined, manager shutdown releases
the retained primary-context reference.

## Ownership summary

| Owner | Resources |
| --- | --- |
| `GpuContext` | GPU/NUMA identity, retained primary context, fixed slot table, future immutable per-GPU resources |
| `ThreadSlot` | Stream, pinned/device input, shared scratch |
| `DummyGraph` | Private algorithm objects, execution order, reusable `JobResult` |
| CEL/SDD/MI object | Configuration, device output, pinned-host output |

## CUDA API split

| Responsibility | API |
| --- | --- |
| Driver initialization | `cuInit` |
| Primary-context ownership | `cuDevicePrimaryCtxRetain`, `cuCtxPushCurrent`, `cuCtxPopCurrent`, `cuDevicePrimaryCtxRelease` |
| Runtime device binding | `cudaSetDevice`, `cudaFree(nullptr)` |
| Slot and algorithm resources | `cudaStreamCreateWithFlags`, `cudaHostAlloc`, `cudaMalloc` |
| Frame transfers | `cudaMemcpyAsync` |
| Per-frame wait | `cudaStreamSynchronize` |
| Cleanup | `cudaFree`, `cudaFreeHost`, `cudaStreamDestroy` |

CUDA errors report the API family, expression, error name, numeric code,
description, source file, and line.

## Source layout

```text
src/
  CudaCheck.h              CUDA Runtime/Driver error reporting
  GpuContext.h             per-GPU context identity and fixed slot table
  GpuContextManager.*      discovery, registration, scratch allocation, teardown
  ThreadSlot.h             per-Graph-thread input, stream, and shared scratch
  IAlgo.h                  algorithm allocation, compute, and D2H contracts
  Cel.*                    synthetic CEL matrix-multiplication kernel
  Mi.*                     synthetic MI matrix-multiplication kernel
  Sdd.*                    synthetic SDD matrix-multiplication kernel
  GraphStuff.h             external Graph framework stand-ins
  main.cpp                 multi-threaded runnable example
```
