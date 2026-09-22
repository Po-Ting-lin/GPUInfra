#pragma once

#include <chrono>
#include <cstddef>

#include "Types/AlgoRuntimeInfo.h"
#include "DataCache/GpuCacheManager.h"
#include "Types/FrameMetadata.h"
#include "StaticData/StaticGpuData.h"

struct StaticDataConfig {
    AlgoRuntimeInfo runtime;
    std::size_t detectionZoneBytes = 0;
    std::size_t distortionBytes = 0;
    std::size_t gpuCacheEntries = 0;
    std::chrono::milliseconds gpuCacheWaitTimeout{50};
};

// Graph-copy-scoped owner of fixed frame-layout validation and a bounded GPU
// cache. Frame execution state remains owned by the graph scheduler.
class StaticData {
public:
    StaticData() = default;
    ~StaticData();

    bool init(const StaticDataConfig& config);
    // Cold path only; finalize all configured static data before execute.
    bool uploadStaticData(StaticGpuDataType type, const void* source, std::size_t bytes, std::size_t offset);
    bool finalizeStaticData();
    const void* staticGpuData(StaticGpuDataType type) const;
    std::size_t staticGpuDataBytes(StaticGpuDataType type) const;
    bool resetCache(CacheStatistics* completedStatistics = nullptr);
    CacheStatistics cacheStatisticsSnapshot() const;
    bool execute() const;
    bool release();

    bool isInitialized() const;
    std::size_t gpuCacheEntryCount() const;
    bool validateFrame(const FrameMetadata& metadata) const;
    GpuDataAccess getCacheData(const FrameMetadata& metadata, const GpuCacheRequest& request);

    StaticData(const StaticData&) = delete;
    StaticData& operator=(const StaticData&) = delete;

private:
    GpuCacheManager gpuCacheManager;
    StaticGpuData gpuStaticData;
    AlgoRuntimeInfo frameRuntime;
    bool initialized = false;
};
