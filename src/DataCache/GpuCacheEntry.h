#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "FrameMetadata.h"

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
class GpuCacheEntry {
public:
    GpuCacheEntry() = default;
    ~GpuCacheEntry();

    bool initialize(const std::vector<int>& gpuIds, std::size_t bytes);
    bool release();
    bool isInitialized() const;
    std::size_t bytes() const;
    std::size_t replicaCount() const;

    GpuCacheEntry(const GpuCacheEntry&) = delete;
    GpuCacheEntry& operator=(const GpuCacheEntry&) = delete;
    GpuCacheEntry(GpuCacheEntry&&) = delete;
    GpuCacheEntry& operator=(GpuCacheEntry&&) = delete;

private:
    friend class GpuCacheManager;

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
    std::vector<GpuReplica> replicas;
    std::size_t dataBytes = 0;
    bool inEvictableList = false;
    bool initialized = false;
};
