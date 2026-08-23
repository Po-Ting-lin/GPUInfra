#include "FrameSlot.h"

bool FrameSlot::initializeGpuData(const std::vector<int>& gpuIds, std::size_t bytes) {
    if (cacheState != FrameCacheState::Empty || activeAccesses != 0 || lastUse != 0) {
        return false;
    }
    return deviceData.initialize(gpuIds, bytes);
}

bool FrameSlot::releaseGpuData() {
    if (activeAccesses != 0 || cacheState == FrameCacheState::Loading) {
        return false;
    }

    const bool released = deviceData.release();
    if (released && !deviceData.isInitialized()) {
        metadata = FrameMetadata();
        cacheState = FrameCacheState::Empty;
        lastUse = 0;
    }
    return released;
}

bool FrameSlot::isInitialized() const {
    return deviceData.isInitialized();
}
