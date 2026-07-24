#pragma once

#include <cstddef>
#include <vector>

#include <cuda.h>

#include "IAlgo.h"
#include "ThreadSlot.h"

class GpuContext {
public:
    int gpuId = -1;
    int numaNode = -1;
    int maxThreadsPerGpu = 0;
    std::size_t inputBytes = 0;
    std::size_t scratchBytes = 0;
    bool configured = false;

    CUdevice device = 0;
    CUcontext primaryCtx = nullptr;

    // One algorithm instance per (GPU, algorithm).
    std::vector<IAlgo*> algos;

    // Fixed-size slot table. Empty entries are reused without renumbering
    // surviving workers, so threadId remains a stable algorithm-state index.
    std::vector<ThreadSlot*> threadSlots;

    GpuContext() = default;

    ~GpuContext() {
        for (ThreadSlot* slot : threadSlots) {
            delete slot;
        }
        for (IAlgo* algo : algos) {
            delete algo;
        }
    }

    std::size_t activeSlotCount() const {
        std::size_t count = 0;
        for (const ThreadSlot* slot : threadSlots) {
            if (slot != nullptr) {
                ++count;
            }
        }
        return count;
    }

    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;
};
