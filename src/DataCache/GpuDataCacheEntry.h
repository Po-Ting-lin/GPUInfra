#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "Types/FrameMetadata.h"

struct GpuReplica {
    int gpuId = -1;
    void* d_data = nullptr;
    bool valid = false;
};

enum class GpuCacheState {
    Empty,
    Loading,
    Valid,
};

// Reusable cache entry that owns its persistent per-GPU device allocations.
// It does not own scheduler state, CPU bytes, or task-private resources.
class GpuDataCacheEntry {
public:
    GpuDataCacheEntry() = default;
    ~GpuDataCacheEntry();

    bool initialize(const std::vector<int>& gpuIds, std::size_t bytes);
    bool release();
    bool isInitialized() const;
    std::size_t bytes() const;
    std::size_t replicaCount() const;

    GpuDataCacheEntry(const GpuDataCacheEntry&) = delete;
    GpuDataCacheEntry& operator=(const GpuDataCacheEntry&) = delete;
    GpuDataCacheEntry(GpuDataCacheEntry&&) = delete;
    GpuDataCacheEntry& operator=(GpuDataCacheEntry&&) = delete;

private:
    friend class GpuDataCache;

    void* dataForGpu(int gpuId);
    const void* dataForGpu(int gpuId) const;
    bool replicaValid(int gpuId) const;
    void invalidateReplicas();
    bool markReplicaValid(int gpuId);

    FrameMetadata metadata;
    GpuCacheState cacheState = GpuCacheState::Empty;
    std::size_t activeAccesses = 0;
    std::size_t previousEvictable = std::numeric_limits<std::size_t>::max();
    std::size_t nextEvictable = std::numeric_limits<std::size_t>::max();
    // FIFO order includes live readers so a Hit never changes fill age.
    std::size_t previousFilled = std::numeric_limits<std::size_t>::max();
    std::size_t nextFilled = std::numeric_limits<std::size_t>::max();
    std::vector<GpuReplica> replicas;
    std::size_t dataBytes = 0;
    bool inEvictableList = false;
    bool inFilledList = false;
    bool discardWhenUnused = false;
    bool initialized = false;
};
