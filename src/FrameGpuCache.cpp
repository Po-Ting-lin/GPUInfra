#include "FrameGpuCache.h"

#include <algorithm>
#include <limits>
#include <memory>

#include "TaskGpuResources.h"

namespace {

constexpr std::size_t NO_SLOT = std::numeric_limits<std::size_t>::max();

bool sameMetadata(const FrameMetadata& first, const FrameMetadata& second) {
    return first.id == second.id && first.bytes == second.bytes && first.width == second.width && first.height == second.height && first.dtype == second.dtype;
}

}  // namespace

FrameGpuCache::~FrameGpuCache() {
    release();
}

bool FrameGpuCache::initialize(const std::vector<int>& gpuIds, std::size_t bytes, std::size_t slotCount) {
    {
        std::lock_guard<std::mutex> guard(lock);
        if (initialized || releasing || !slots.empty() || !eligibleGpuIds.empty() || gpuIds.size() != 1 || gpuIds.front() < 0 || bytes == 0) {
            return false;
        }
    }

    std::vector<std::unique_ptr<FrameSlot>> newSlots;
    try {
        newSlots.reserve(slotCount);
        for (std::size_t index = 0; index < slotCount; ++index) {
            std::unique_ptr<FrameSlot> slot = std::make_unique<FrameSlot>();
            if (!slot->initializeGpuData(gpuIds, bytes)) {
                return false;
            }
            newSlots.push_back(std::move(slot));
        }
    } catch (...) {
        return false;
    }

    std::lock_guard<std::mutex> guard(lock);
    if (initialized || releasing || !slots.empty() || !eligibleGpuIds.empty()) {
        return false;
    }
    try {
        eligibleGpuIds = gpuIds;
    } catch (...) {
        eligibleGpuIds.clear();
        return false;
    }
    slots = std::move(newSlots);
    frameBytes = bytes;
    activeFallbackAccesses = 0;
    useSequence = 0;
    initialized = true;
    return true;
}

FrameGpuAccess FrameGpuCache::acquire(const FrameMetadata& metadata, const TaskGpuResources& resources) {
    std::lock_guard<std::mutex> guard(lock);
    if (!initialized || releasing || metadata.bytes != frameBytes || metadata.width <= 0 || metadata.height <= 0 || resources.gpuId < 0 || resources.stream == nullptr || resources.d_input == nullptr || resources.inBytes != frameBytes || std::find(eligibleGpuIds.begin(), eligibleGpuIds.end(), resources.gpuId) == eligibleGpuIds.end()) {
        return FrameGpuAccess();
    }

    for (std::size_t index = 0; index < slots.size(); ++index) {
        FrameSlot& slot = *slots[index];
        if (slot.cacheState == FrameCacheState::Empty || slot.metadata.id != metadata.id) {
            continue;
        }
        if (!sameMetadata(slot.metadata, metadata)) {
            return FrameGpuAccess();
        }
        if (slot.cacheState == FrameCacheState::Loading) {
            return makeFallbackAccess(metadata, resources);
        }
        if (slot.cacheState == FrameCacheState::Valid) {
            void* deviceData = slot.deviceData.dataForGpu(resources.gpuId);
            if (deviceData == nullptr || !slot.deviceData.replicaValid(resources.gpuId)) {
                // A future multi-GPU implementation can reserve a local replica
                // fill here. The current one-GPU scope safely re-uploads instead.
                return makeFallbackAccess(metadata, resources);
            }
            ++slot.activeAccesses;
            slot.lastUse = nextUse();
            return FrameGpuAccess(this, deviceData, frameBytes, index, metadata.id, resources.stream, resources.gpuId, FrameGpuAccessSource::CacheHit);
        }
    }

    std::size_t candidateIndex = NO_SLOT;
    std::uint64_t oldestUse = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < slots.size(); ++index) {
        const FrameSlot& slot = *slots[index];
        if (slot.cacheState == FrameCacheState::Empty) {
            candidateIndex = index;
            break;
        }
        if (slot.cacheState == FrameCacheState::Valid && slot.activeAccesses == 0 && slot.lastUse < oldestUse) {
            candidateIndex = index;
            oldestUse = slot.lastUse;
        }
    }
    if (candidateIndex == NO_SLOT) {
        return makeFallbackAccess(metadata, resources);
    }

    FrameSlot& candidate = *slots[candidateIndex];
    void* deviceData = candidate.deviceData.dataForGpu(resources.gpuId);
    if (deviceData == nullptr) {
        return FrameGpuAccess();
    }
    candidate.metadata = metadata;
    candidate.deviceData.invalidateReplicas();
    candidate.cacheState = FrameCacheState::Loading;
    candidate.activeAccesses = 1;
    candidate.lastUse = nextUse();
    return FrameGpuAccess(this, deviceData, frameBytes, candidateIndex, metadata.id, resources.stream, resources.gpuId, FrameGpuAccessSource::CacheFill);
}

bool FrameGpuCache::release() {
    {
        std::lock_guard<std::mutex> guard(lock);
        if (!initialized && slots.empty() && eligibleGpuIds.empty()) {
            return true;
        }
        if (releasing) {
            return false;
        }
        if (activeFallbackAccesses != 0) {
            return false;
        }
        for (const std::unique_ptr<FrameSlot>& slot : slots) {
            if (slot != nullptr && (slot->activeAccesses != 0 || slot->cacheState == FrameCacheState::Loading)) {
                return false;
            }
        }
        initialized = false;
        releasing = true;
    }

    bool ok = true;
    bool allReleased = true;
    for (const std::unique_ptr<FrameSlot>& slot : slots) {
        if (slot == nullptr) {
            continue;
        }
        if (!slot->releaseGpuData()) {
            ok = false;
        }
        if (slot->isInitialized()) {
            allReleased = false;
        }
    }

    std::lock_guard<std::mutex> guard(lock);
    if (allReleased) {
        slots.clear();
        eligibleGpuIds.clear();
        frameBytes = 0;
        activeFallbackAccesses = 0;
        useSequence = 0;
        initialized = false;
    }
    releasing = false;
    return ok;
}

bool FrameGpuCache::isInitialized() const {
    std::lock_guard<std::mutex> guard(lock);
    return initialized;
}

std::size_t FrameGpuCache::slotCount() const {
    std::lock_guard<std::mutex> guard(lock);
    return slots.size();
}

std::size_t FrameGpuCache::bytes() const {
    std::lock_guard<std::mutex> guard(lock);
    return frameBytes;
}

bool FrameGpuCache::completeAccess(FrameGpuAccess& access, bool succeeded) {
    if (access.owner != this) {
        return false;
    }
    if (access.accessSource == FrameGpuAccessSource::TaskFallback) {
        std::lock_guard<std::mutex> guard(lock);
        const bool valid = initialized && !releasing && activeFallbackAccesses > 0 && access.slotIndex == NO_SLOT && access.d_data != nullptr && access.dataBytes == frameBytes;
        if (valid) {
            --activeFallbackAccesses;
        }
        access.reset();
        return valid && succeeded;
    }

    std::lock_guard<std::mutex> guard(lock);
    bool valid = initialized && !releasing && access.slotIndex < slots.size();
    FrameSlot* slot = valid ? slots[access.slotIndex].get() : nullptr;
    if (valid) {
        valid = slot != nullptr && slot->metadata.id == access.frameId && slot->deviceData.dataForGpu(access.deviceId) == access.d_data && access.dataBytes == frameBytes;
    }

    if (access.accessSource == FrameGpuAccessSource::CacheHit) {
        valid = valid && slot->cacheState == FrameCacheState::Valid && slot->activeAccesses > 0 && slot->deviceData.replicaValid(access.deviceId);
        if (valid) {
            --slot->activeAccesses;
        }
    }
    else if (access.accessSource == FrameGpuAccessSource::CacheFill) {
        valid = valid && slot->cacheState == FrameCacheState::Loading && slot->activeAccesses == 1;
        if (valid && succeeded) {
            if (slot->deviceData.markReplicaValid(access.deviceId)) {
                slot->cacheState = FrameCacheState::Valid;
                slot->activeAccesses = 0;
            }
            else {
                resetFillSlot(*slot);
                succeeded = false;
            }
        }
        else if (valid) {
            resetFillSlot(*slot);
        }
        else {
            succeeded = false;
        }
    }
    else {
        valid = false;
    }

    access.reset();
    return valid && succeeded;
}

void FrameGpuCache::abortAccess(FrameGpuAccess& access) {
    if (access.owner != this) {
        return;
    }
    if (access.accessSource == FrameGpuAccessSource::TaskFallback) {
        std::lock_guard<std::mutex> guard(lock);
        if (activeFallbackAccesses > 0) {
            --activeFallbackAccesses;
        }
        access.reset();
        return;
    }

    std::lock_guard<std::mutex> guard(lock);
    if (access.slotIndex < slots.size()) {
        FrameSlot& slot = *slots[access.slotIndex];
        const bool matchingAccess = slot.metadata.id == access.frameId && slot.deviceData.dataForGpu(access.deviceId) == access.d_data && access.dataBytes == frameBytes;
        if (matchingAccess && access.accessSource == FrameGpuAccessSource::CacheHit && slot.cacheState == FrameCacheState::Valid && slot.activeAccesses > 0) {
            --slot.activeAccesses;
        }
        else if (matchingAccess && access.accessSource == FrameGpuAccessSource::CacheFill && slot.cacheState == FrameCacheState::Loading && slot.activeAccesses == 1) {
            resetFillSlot(slot);
        }
    }
    access.reset();
}

FrameGpuAccess FrameGpuCache::makeFallbackAccess(const FrameMetadata& metadata, const TaskGpuResources& resources) {
    ++activeFallbackAccesses;
    return FrameGpuAccess(this, resources.d_input, frameBytes, NO_SLOT, metadata.id, resources.stream, resources.gpuId, FrameGpuAccessSource::TaskFallback);
}

void FrameGpuCache::resetFillSlot(FrameSlot& slot) {
    slot.metadata = FrameMetadata();
    slot.deviceData.invalidateReplicas();
    slot.cacheState = FrameCacheState::Empty;
    slot.activeAccesses = 0;
    slot.lastUse = 0;
}

std::uint64_t FrameGpuCache::nextUse() {
    ++useSequence;
    if (useSequence != 0) {
        return useSequence;
    }

    std::uint64_t next = 1;
    for (const std::unique_ptr<FrameSlot>& slot : slots) {
        if (slot != nullptr && slot->cacheState != FrameCacheState::Empty) {
            slot->lastUse = next;
            ++next;
        }
    }
    useSequence = next;
    return useSequence;
}
