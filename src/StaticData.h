#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "FrameMetadata.h"
#include "GpuCacheManager.h"
#include "IAlgo.h"

struct TaskGpuResources;

struct StaticDataConfig {
    std::vector<int> gpuIds;
    AlgoRuntimeInfo runtime;
    std::vector<FrameMetadata> frames;
    std::size_t gpuCacheEntries = 0;
};

// Graph-copy-scoped owner of an immutable frame registry and a bounded GPU
// cache. Frame execution state remains owned by the graph scheduler.
class StaticData {
public:
    StaticData() = default;
    ~StaticData();

    bool init(const StaticDataConfig& config);
    bool execute() const;
    bool release();

    std::size_t frameCount() const;
    std::size_t gpuCacheEntryCount() const;
    bool validateFrame(const FrameMetadata& metadata) const;
    GpuDataAccess acquireGpuData(const FrameMetadata& metadata, const TaskGpuResources& resources);

    StaticData(const StaticData&) = delete;
    StaticData& operator=(const StaticData&) = delete;

private:
    std::unordered_map<std::uint64_t, FrameMetadata> registeredFrames;
    GpuCacheManager gpuCacheManager;
    bool initialized = false;
};
