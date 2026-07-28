#pragma once

#include <cstddef>
#include <vector>

#include <cuda.h>

#include "ThreadSlot.h"

class GpuContext {
public:
    int gpuId = -1;
    int numaNode = -1;
    int maxThreadsPerGpu = 0;
    std::size_t inputBytes = 0;
    bool configured = false;

    CUdevice device = 0;
    CUcontext primaryCtx = nullptr;

    // Fixed-size slot table. Empty entries are reused without renumbering
    // surviving workers.
    std::vector<ThreadSlot*> threadSlots;

    GpuContext() = default;

    ~GpuContext() {
        for (ThreadSlot* slot : threadSlots) {
            delete slot;
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
