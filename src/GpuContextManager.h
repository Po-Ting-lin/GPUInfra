#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "GpuTopology.h"
#include "TaskGpuResources.h"

class GpuContext;

struct GpuInfraConfig {
    bool requireNuma = true;
};

class GpuContextManager {
public:
    // Serialized cold path: discover GPUs and retain their primary contexts.
    static bool init(const GpuInfraConfig& config);

    // Cold path, called under framework NUMA affinity. Reject zero/multiple GPUs.
    static bool gpuIdsForCurrentNumaNode(std::vector<int>& gpuIds);
    static bool registerTask(int gpuId, TaskGpuResources& resources);
    static bool makeTaskCurrent(const TaskGpuResources& resources);
    static bool unregisterTask(TaskGpuResources& resources);
    static void shutdown();

    static std::vector<GpuLocation> gpuLocations();

private:
    static std::vector<GpuContext*> contexts;
    static std::mutex lock;
    static std::atomic<bool> initialised;
};
