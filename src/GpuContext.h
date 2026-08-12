#pragma once

#include <cstddef>
#include <vector>

#include <cuda.h>

#include "TaskGpuResources.h"

class GpuContext {
public:
    int gpuId = -1;
    int numaNode = -1;
    CUdevice device = 0;
    CUcontext primaryCtx = nullptr;

    // Task resource IDs are stable while registered. Empty entries are reused
    // without renumbering surviving tasks.
    std::vector<TaskGpuResources*> taskResources;

    GpuContext() = default;

    std::size_t activeTaskCount() const {
        std::size_t count = 0;
        for (const TaskGpuResources* resources : taskResources) {
            if (resources != nullptr) {
                ++count;
            }
        }
        return count;
    }

    GpuContext(const GpuContext&) = delete;
    GpuContext& operator=(const GpuContext&) = delete;
};
