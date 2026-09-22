#pragma once

// Profiling-only annotations. Use fixed names and a braced lexical scope.
// Disabled annotations do not evaluate (or require declarations for) arguments.
#if defined(GPUINFRA_ENABLE_NVTX) && GPUINFRA_ENABLE_NVTX
#include <nvtx3/nvtx3.hpp>
#define GPUINFRA_NVTX_JOIN_IMPL(left, right) left##right
#define GPUINFRA_NVTX_JOIN(left, right) GPUINFRA_NVTX_JOIN_IMPL(left, right)
#define GPUINFRA_NVTX_SCOPE(name) const nvtx3::scoped_range GPUINFRA_NVTX_JOIN(gpuinfraNvtxScope, __COUNTER__){name}
#else
#define GPUINFRA_NVTX_SCOPE(name) ((void)0)
#endif
