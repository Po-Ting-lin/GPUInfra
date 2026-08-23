#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "FrameGpuData.h"
#include "FrameMetadata.h"

enum class FrameCacheState {
    Empty,
    Loading,
    Valid,
};

// Reusable GPU cache entry. It does not own logical frame state or CPU bytes.
class FrameSlot {
public:
    FrameSlot() = default;

    bool initializeGpuData(const std::vector<int>& gpuIds, std::size_t bytes);
    bool releaseGpuData();
    bool isInitialized() const;

    FrameSlot(const FrameSlot&) = delete;
    FrameSlot& operator=(const FrameSlot&) = delete;

private:
    friend class FrameGpuCache;

    FrameMetadata metadata;
    FrameCacheState cacheState = FrameCacheState::Empty;
    std::size_t activeAccesses = 0;
    std::uint64_t lastUse = 0;
    FrameGpuData deviceData;
};
