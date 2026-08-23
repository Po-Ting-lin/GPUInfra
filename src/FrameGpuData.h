#pragma once

#include <cstddef>
#include <vector>

struct FrameGpuReplica {
    int gpuId = -1;
    void* d_data = nullptr;
    bool valid = false;
};

///////////////////////////////////////////////////////////////////////////////////

// Device allocation owned by one reusable FrameSlot cache entry. Residency and
// lease state are coordinated by FrameGpuCache.
class FrameGpuData {
public:
    FrameGpuData() = default;
    ~FrameGpuData();

    bool initialize(const std::vector<int>& gpuIds, std::size_t bytes);
    bool release();

    bool isInitialized() const;
    std::size_t bytes() const;
    std::size_t replicaCount() const;

    FrameGpuData(const FrameGpuData&) = delete;
    FrameGpuData& operator=(const FrameGpuData&) = delete;
    FrameGpuData(FrameGpuData&&) = delete;
    FrameGpuData& operator=(FrameGpuData&&) = delete;

private:
    friend class FrameGpuCache;

    void* dataForGpu(int gpuId);
    const void* dataForGpu(int gpuId) const;
    bool replicaValid(int gpuId) const;
    void invalidateReplicas();
    bool markReplicaValid(int gpuId);

    std::vector<FrameGpuReplica> replicas;
    std::size_t dataBytes = 0;
    bool initialized = false;
};
