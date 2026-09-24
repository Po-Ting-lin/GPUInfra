#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "Context/GpuTopology.h"
#include "Types/TaskGpuResources.h"

class GpuContext;

struct GpuInfraConfig {
    // Unknown PCI placement is accepted only with one confirmed online node.
    // false explicitly allows node 0 fallback when topology is inconclusive.
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
