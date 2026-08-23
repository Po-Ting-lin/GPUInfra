#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "FrameGpuCache.h"
#include "FrameMetadata.h"
#include "IAlgo.h"

struct TaskGpuResources;

struct StaticDataConfig {
    int numaNode = -1;
    std::vector<int> gpuIds;
    AlgoRuntimeInfo runtime;
    std::vector<FrameMetadata> frames;
    std::size_t frameCacheSlots = 0;
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
    std::size_t frameCacheSlotCount() const;
    bool validateFrame(const FrameMetadata& metadata, int numaNode) const;
    FrameGpuAccess acquireFrameGpuAccess(const FrameMetadata& metadata, const TaskGpuResources& resources);

    bool isInitialized() const;

    StaticData(const StaticData&) = delete;
    StaticData& operator=(const StaticData&) = delete;

private:
    const FrameMetadata* findFrameMetadata(const FrameMetadata& metadata) const;

    std::vector<FrameMetadata> registeredFrames;
    std::unordered_map<std::uint64_t, std::size_t> frameIdToIndex;
    FrameGpuCache frameGpuCache;
    int graphNumaNode = -1;
    bool initialized = false;
};
