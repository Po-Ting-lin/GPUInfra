#pragma once

#include <cstddef>
#include <vector>

#include "FrameMetadata.h"
#include "GpuCacheManager.h"
#include "IAlgo.h"

struct TaskGpuResources;

struct StaticDataConfig {
    std::vector<int> gpuIds;
    AlgoRuntimeInfo runtime;
    std::size_t gpuCacheEntries = 0;
};

// Graph-copy-scoped owner of fixed frame-layout validation and a bounded GPU
// cache. Frame execution state remains owned by the graph scheduler.
class StaticData {
public:
    StaticData() = default;
    ~StaticData();

    bool init(const StaticDataConfig& config);
    bool resetCache();
    bool execute() const;
    bool release();

    bool isInitialized() const;
    std::size_t gpuCacheEntryCount() const;
    bool validateFrame(const FrameMetadata& metadata) const;
    GpuDataAccess acquireGpuData(const FrameMetadata& metadata, const TaskGpuResources& resources);

    StaticData(const StaticData&) = delete;
    StaticData& operator=(const StaticData&) = delete;

private:
    GpuCacheManager gpuCacheManager;
    AlgoRuntimeInfo frameRuntime;
    bool initialized = false;
};
