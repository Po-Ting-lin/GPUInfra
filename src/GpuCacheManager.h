#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "FrameMetadata.h"
#include "GpuCacheEntry.h"
#include "GpuDataAccess.h"

struct TaskGpuResources;

// Graph-copy-scoped, fixed-capacity GPU data cache. The current implementation
// keys immutable frame inputs by FrameMetadata and may use task fallback.
class GpuCacheManager {
public:
    GpuCacheManager() = default;
    ~GpuCacheManager();

    bool initialize(const std::vector<int>& gpuIds, std::size_t bytes, std::size_t cacheEntryCount);
    GpuDataAccess acquire(const FrameMetadata& metadata, const TaskGpuResources& resources);
    bool release();

    bool isInitialized() const;
    std::size_t entryCount() const;
    std::size_t bytes() const;

    GpuCacheManager(const GpuCacheManager&) = delete;
    GpuCacheManager& operator=(const GpuCacheManager&) = delete;

private:
    friend class GpuDataAccess;

    bool completeAccess(GpuDataAccess& access, bool succeeded);
    void abortAccess(GpuDataAccess& access);
    GpuDataAccess makeFallbackAccess(const FrameMetadata& metadata, const TaskGpuResources& resources);
    void resetFillEntry(GpuCacheEntry& entry);
    std::uint64_t nextUse();

    mutable std::mutex lock;
    std::vector<std::unique_ptr<GpuCacheEntry>> entries;
    std::vector<int> eligibleGpuIds;
    std::size_t dataBytes = 0;
    std::size_t activeFallbackAccesses = 0;
    std::uint64_t useSequence = 0;
    bool initialized = false;
    bool releasing = false;
};
