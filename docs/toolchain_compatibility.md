# Toolchain compatibility

GPUInfra requires Linux, CMake 3.24+, and a C++17 host compiler. The CMake
`gpuinfra` target exports `cxx_std_17` to callers. `GpuDataAccess` is movable but not copyable. The supported host build and
caller contract remain C++17; this ownership change does not validate a
C++14 build.

## Core-only integration

The core contains only `.cpp` files and does not require CUDA-language
compilation. Use a separate build directory for each toolkit/compiler pair:

```sh
cmake -S . -B cmake-build-core \
    -DGPUINFRA_BUILD_DEMO=OFF \
    -DBUILD_TESTING=OFF \
    -DGPUINFRA_ENABLE_NVTX=OFF \
    -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build cmake-build-core --parallel 4
```

This mode links CUDA Runtime, Driver and cuFFT libraries but does not enable
nvcc or build demo-dependent integration tests. When using `add_subdirectory`,
set `GPUINFRA_BUILD_DEMO=OFF` before adding GPUInfra and link `gpuinfra`.
AOI's host compiler, standard-library ABI and CUDA library selection must remain
consistent with the surrounding application.

CMake accepts CUDA Toolkit 10.1 or newer for this core-only path. This is a
configuration floor, not a claim that every accepted combination is validated.
CUDA 10.1 nvcc only supports dialects through C++14
([NVIDIA compiler documentation](https://docs.nvidia.com/cuda/archive/10.1/cuda-compiler-driver-nvcc/index.html#options-for-specifying-behavior-of-compiler-linker)).
Keep cache lease acquisition in C++17 host `.cpp` translation units when using
that toolkit. Enabling C++17 on a host-only core does not make a CUDA 10.1 `.cu`
caller compatible with this API.

## Demo and tests

`GPUINFRA_BUILD_DEMO=ON` remains the default. The demo requires CUDA C++17 and
rejects nvcc versions older than 11.0 with an actionable configuration error.
Choose a CUDA architecture supported by both the toolkit and target GPU;
the repository's default `86` is not suitable for every older toolkit.
For the reported CUDA 11.0 / Turing build, explicitly use
`-DCMAKE_CUDA_ARCHITECTURES=75`. Do not copy this value to an unrelated GPU.
NVTX is optional and OFF by default; ON requires available `nvtx3/nvtx3.hpp`.

## Optional diagnostic enum detection

CMake compiles probes against the selected `cufft.h` for:

- `CUFFT_MISSING_DEPENDENCY`
- `CUFFT_NVRTC_FAILURE`
- `CUFFT_NVJITLINK_FAILURE`
- `CUFFT_NVSHMEM_FAILURE`

Only declared enum cases are compiled into `GpuDiagnostics.cpp`. cuFFT and
CUDA versions are not interchangeable, and enum constants are not preprocessor
macros. Unknown or unavailable codes still print `CUFFT_UNKNOWN_ERROR` and the
original numeric code. Public `CUDA_CHECK()` usage is unchanged.

Use the CMake target to preserve these private compile definitions. A separate
non-CMake build must perform equivalent header probes and define the four
`GPUINFRA_HAS_CUFFT_*` flags as 0 or 1 when compiling `GpuDiagnostics.cpp`;
without them optional names are omitted, but numeric error codes are retained.

## Validation status (2026-09-23)

- Local toolkit: CUDA 13.1.115, GNU C++ 13.3.0.
- Core-only build: passed without enabling the CUDA language.
- NVTX ON and OFF demo builds: all seven CTests passed in each; the hardware-dependent multi-NUMA case was skipped.
- Logging probes: optional codes 17–20 preserve their names when available,
  or report unknown with the original numeric code when omitted.
- Header-availability regression: built the core against a temporary copy of
  the local cuFFT header with all four optional enum declarations removed.
  All four probes rejected the missing enums and compilation succeeded.
  This checks feature detection, not old-toolkit compatibility as a whole.
- CUDA 10.1 / 11.0 / 11.3 with AOI's actual compilers: not installed locally;
  full build, link and execution validation remains required on those systems.

Changing the toolkit/compiler requires a fresh build directory so cached
feature results cannot be reused across different headers.
