#pragma once

#include <cstddef>

#include <cuda_runtime.h>

// Borrowed execution resources for one payload. The caller owns the stream
// and fallback allocation and keeps both alive until the access finishes.
// Simultaneously used payloads must not overwrite each other's fallback data.
struct GpuCacheRequest {
    int gpuId = -1;
    cudaStream_t stream = nullptr;
    void* d_fallback = nullptr;
    std::size_t fallbackBytes = 0;
};
