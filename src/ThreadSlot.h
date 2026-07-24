#pragma once

#include <cstddef>
#include <thread>

#include <cuda_runtime.h>

class GpuContext;

struct ThreadSlot {
    int threadId = -1;
    std::thread::id ownerTid;
    int gpuId = -1;
    int numaNode = -1;

    cudaStream_t stream = nullptr;
    void* h_in = nullptr;
    void* d_in = nullptr;
    std::size_t inBytes = 0;

    void* d_scratch = nullptr;
    std::size_t scratchBytes = 0;

    GpuContext* ctx = nullptr;
};
