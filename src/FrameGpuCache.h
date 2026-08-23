#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "FrameGpuAccess.h"
#include "FrameMetadata.h"
#include "FrameSlot.h"

struct TaskGpuResources;

// Graph-copy-scoped, fixed-capacity GPU frame cache. Cache misses may reserve an
// inactive slot for a fill or use the task-private fallback buffer immediately.
class FrameGpuCache {
public:
    FrameGpuCache() = default;
    ~FrameGpuCache();

    bool initialize(const std::vector<int>& gpuIds, std::size_t bytes, std::size_t slotCount);
    FrameGpuAccess acquire(const FrameMetadata& metadata, const TaskGpuResources& resources);
    bool release();

    bool isInitialized() const;
    std::size_t slotCount() const;
    std::size_t bytes() const;

    FrameGpuCache(const FrameGpuCache&) = delete;
    FrameGpuCache& operator=(const FrameGpuCache&) = delete;

private:
    friend class FrameGpuAccess;

    bool completeAccess(FrameGpuAccess& access, bool succeeded);
    void abortAccess(FrameGpuAccess& access);
    FrameGpuAccess makeFallbackAccess(const FrameMetadata& metadata, const TaskGpuResources& resources);
    void resetFillSlot(FrameSlot& slot);
    std::uint64_t nextUse();

    mutable std::mutex lock;
    std::vector<std::unique_ptr<FrameSlot>> slots;
    std::vector<int> eligibleGpuIds;
    std::size_t frameBytes = 0;
    std::size_t activeFallbackAccesses = 0;
    std::uint64_t useSequence = 0;
    bool initialized = false;
    bool releasing = false;
};
