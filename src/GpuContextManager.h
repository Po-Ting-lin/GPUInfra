#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "TaskGpuResources.h"

class GpuContext;

struct GpuInfraConfig {
    bool requireNuma = true;
};

struct GpuLocation {
    int gpuId = -1;
    int numaNode = -1;
};

class GpuContextManager {
public:
    // Serialized cold path: discover GPUs and retain their primary contexts.
    static bool init(const GpuInfraConfig& config);

    static bool pinCurrentThreadToNumaNode(int numaNode);
    static bool validateGpuIdsForNumaNode(int numaNode, const std::vector<int>& gpuIds);
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
