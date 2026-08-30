#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "FrameMetadata.h"
#include "GpuCacheEntry.h"
#include "GpuDataAccess.h"
#include "GpuResidencyTable.h"

struct TaskGpuResources;

// Graph-copy-scoped, fixed-capacity GPU data cache. The current implementation
// indexes immutable frame inputs by GpuDataKey and may use task fallback.
class GpuCacheManager {
public:
    GpuCacheManager() = default;
    ~GpuCacheManager();

    bool initialize(const std::vector<int>& gpuIds, std::size_t bytes, std::size_t cacheEntryCount);
    bool resetCache();
    GpuDataAccess acquire(const FrameMetadata& metadata, const TaskGpuResources& resources);
    bool release();

    bool isInitialized() const;
    std::size_t entryCount() const;
    std::size_t bytes() const;

    GpuCacheManager(const GpuCacheManager&) = delete;
    GpuCacheManager& operator=(const GpuCacheManager&) = delete;

private:
    friend class GpuDataAccess;

    inline static constexpr std::size_t NO_ENTRY = std::numeric_limits<std::size_t>::max();

    bool canResetLocked() const;
    void resetEntriesLocked();
    std::size_t takeCandidate(bool& wasEmpty);
    void restoreCandidate(std::size_t index, bool wasEmpty);
    bool addEvictableEntry(std::size_t index);
    bool removeEvictableEntry(std::size_t index);
    bool completeAccess(GpuDataAccess& access, bool succeeded);
    void abortAccess(GpuDataAccess& access);
    GpuDataAccess makeFallbackAccess(const FrameMetadata& metadata, const TaskGpuResources& resources);
    void resetFillEntry(GpuCacheEntry& entry, std::size_t index);

    mutable std::mutex lock;
    std::vector<std::unique_ptr<GpuCacheEntry>> entries;
    std::vector<int> eligibleGpuIds;
    GpuResidencyTable residencyTable;
    std::vector<std::size_t> emptyEntries;
    std::size_t leastRecentlyUsed = NO_ENTRY;
    std::size_t mostRecentlyUsed = NO_ENTRY;
    std::size_t dataBytes = 0;
    std::size_t activeFallbackAccesses = 0;
    bool initialized = false;
    bool releasing = false;
};
