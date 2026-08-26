#include "GpuCacheManager.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#include "TaskGpuResources.h"

namespace {

constexpr std::size_t NO_ENTRY = std::numeric_limits<std::size_t>::max();

bool sameMetadata(const FrameMetadata& first, const FrameMetadata& second) {
    return first.id == second.id && first.bytes == second.bytes && first.width == second.width && first.height == second.height && first.dtype == second.dtype;
}

}  // namespace

GpuCacheManager::~GpuCacheManager() {
    release();
}

bool GpuCacheManager::initialize(const std::vector<int>& gpuIds, std::size_t bytes, std::size_t cacheEntryCount) {
    {
        std::lock_guard<std::mutex> guard(lock);
        if (initialized || releasing || !entries.empty() || !eligibleGpuIds.empty() || gpuIds.size() != 1 || gpuIds.front() < 0 || bytes == 0) {
            return false;
        }
    }

    std::vector<std::unique_ptr<GpuCacheEntry>> newEntries;
    try {
        newEntries.reserve(cacheEntryCount);
        for (std::size_t index = 0; index < cacheEntryCount; ++index) {
            std::unique_ptr<GpuCacheEntry> entry = std::make_unique<GpuCacheEntry>();
            if (!entry->initialize(gpuIds, bytes)) {
                return false;
            }
            newEntries.push_back(std::move(entry));
        }
    } catch (...) {
        return false;
    }

    std::lock_guard<std::mutex> guard(lock);
    if (initialized || releasing || !entries.empty() || !eligibleGpuIds.empty()) {
        return false;
    }
    try {
        eligibleGpuIds = gpuIds;
    } catch (...) {
        eligibleGpuIds.clear();
        return false;
    }
    entries = std::move(newEntries);
    dataBytes = bytes;
    activeFallbackAccesses = 0;
    useSequence = 0;
    initialized = true;
    return true;
}

GpuDataAccess GpuCacheManager::acquire(const FrameMetadata& metadata, const TaskGpuResources& resources) {
    std::lock_guard<std::mutex> guard(lock);
    if (!initialized || releasing || metadata.bytes != dataBytes || metadata.width <= 0 || metadata.height <= 0 || resources.gpuId < 0 || resources.stream == nullptr || resources.d_input == nullptr || resources.inBytes != dataBytes || std::find(eligibleGpuIds.begin(), eligibleGpuIds.end(), resources.gpuId) == eligibleGpuIds.end()) {
        return GpuDataAccess();
    }

    for (std::size_t index = 0; index < entries.size(); ++index) {
        GpuCacheEntry& entry = *entries[index];
        if (entry.cacheState == GpuCacheState::Empty || entry.metadata.id != metadata.id) {
            continue;
        }
        if (!sameMetadata(entry.metadata, metadata)) {
            return GpuDataAccess();
        }
        if (entry.cacheState == GpuCacheState::Loading) {
            return makeFallbackAccess(metadata, resources);
        }
        if (entry.cacheState == GpuCacheState::Valid) {
            void* deviceData = entry.dataForGpu(resources.gpuId);
            if (deviceData == nullptr || !entry.replicaValid(resources.gpuId)) {
                // A future multi-GPU implementation can reserve a local replica
                // fill here. The current one-GPU scope safely re-uploads instead.
                return makeFallbackAccess(metadata, resources);
            }
            ++entry.activeAccesses;
            entry.lastUse = nextUse();
            return GpuDataAccess(this, deviceData, dataBytes, index, metadata.id, resources.stream, resources.gpuId, GpuDataAccessSource::CacheHit);
        }
    }

    std::size_t candidateIndex = NO_ENTRY;
    std::uint64_t oldestUse = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const GpuCacheEntry& entry = *entries[index];
        if (entry.cacheState == GpuCacheState::Empty) {
            candidateIndex = index;
            break;
        }
        if (entry.cacheState == GpuCacheState::Valid && entry.activeAccesses == 0 && entry.lastUse < oldestUse) {
            candidateIndex = index;
            oldestUse = entry.lastUse;
        }
    }
    if (candidateIndex == NO_ENTRY) {
        return makeFallbackAccess(metadata, resources);
    }

    GpuCacheEntry& candidate = *entries[candidateIndex];
    void* deviceData = candidate.dataForGpu(resources.gpuId);
    if (deviceData == nullptr) {
        return GpuDataAccess();
    }
    candidate.metadata = metadata;
    candidate.invalidateReplicas();
    candidate.cacheState = GpuCacheState::Loading;
    candidate.activeAccesses = 1;
    candidate.lastUse = nextUse();
    return GpuDataAccess(this, deviceData, dataBytes, candidateIndex, metadata.id, resources.stream, resources.gpuId, GpuDataAccessSource::CacheFill);
}

bool GpuCacheManager::release() {
    {
        std::lock_guard<std::mutex> guard(lock);
        if (!initialized && entries.empty() && eligibleGpuIds.empty()) {
            return true;
        }
        if (releasing) {
            return false;
        }
        if (activeFallbackAccesses != 0) {
            return false;
        }
        for (const std::unique_ptr<GpuCacheEntry>& entry : entries) {
            if (entry != nullptr && (entry->activeAccesses != 0 || entry->cacheState == GpuCacheState::Loading)) {
                return false;
            }
        }
        initialized = false;
        releasing = true;
    }

    bool ok = true;
    bool allReleased = true;
    for (const std::unique_ptr<GpuCacheEntry>& entry : entries) {
        if (entry == nullptr) {
            continue;
        }
        if (!entry->release()) {
            ok = false;
        }
        if (entry->isInitialized()) {
            allReleased = false;
        }
    }

    std::lock_guard<std::mutex> guard(lock);
    if (allReleased) {
        entries.clear();
        eligibleGpuIds.clear();
        dataBytes = 0;
        activeFallbackAccesses = 0;
        useSequence = 0;
        initialized = false;
    }
    releasing = false;
    return ok;
}

bool GpuCacheManager::isInitialized() const {
    std::lock_guard<std::mutex> guard(lock);
    return initialized;
}

std::size_t GpuCacheManager::entryCount() const {
    std::lock_guard<std::mutex> guard(lock);
    return entries.size();
}

std::size_t GpuCacheManager::bytes() const {
    std::lock_guard<std::mutex> guard(lock);
    return dataBytes;
}

bool GpuCacheManager::completeAccess(GpuDataAccess& access, bool succeeded) {
    if (access.owner != this) {
        return false;
    }
    if (access.accessSource == GpuDataAccessSource::TaskFallback) {
        std::lock_guard<std::mutex> guard(lock);
        const bool valid = initialized && !releasing && activeFallbackAccesses > 0 && access.entryIndex == NO_ENTRY && access.d_data != nullptr && access.dataBytes == dataBytes;
        if (valid) {
            --activeFallbackAccesses;
        }
        access.reset();
        return valid && succeeded;
    }

    std::lock_guard<std::mutex> guard(lock);
    bool valid = initialized && !releasing && access.entryIndex < entries.size();
    GpuCacheEntry* entry = valid ? entries[access.entryIndex].get() : nullptr;
    if (valid) {
        valid = entry != nullptr && entry->metadata.id == access.dataId && entry->dataForGpu(access.deviceId) == access.d_data && access.dataBytes == dataBytes;
    }

    if (access.accessSource == GpuDataAccessSource::CacheHit) {
        valid = valid && entry->cacheState == GpuCacheState::Valid && entry->activeAccesses > 0 && entry->replicaValid(access.deviceId);
        if (valid) {
            --entry->activeAccesses;
        }
    }
    else if (access.accessSource == GpuDataAccessSource::CacheFill) {
        valid = valid && entry->cacheState == GpuCacheState::Loading && entry->activeAccesses == 1;
        if (valid && succeeded) {
            if (entry->markReplicaValid(access.deviceId)) {
                entry->cacheState = GpuCacheState::Valid;
                entry->activeAccesses = 0;
            }
            else {
                resetFillEntry(*entry);
                succeeded = false;
            }
        }
        else if (valid) {
            resetFillEntry(*entry);
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

void GpuCacheManager::abortAccess(GpuDataAccess& access) {
    if (access.owner != this) {
        return;
    }
    if (access.accessSource == GpuDataAccessSource::TaskFallback) {
        std::lock_guard<std::mutex> guard(lock);
        if (activeFallbackAccesses > 0) {
            --activeFallbackAccesses;
        }
        access.reset();
        return;
    }

    std::lock_guard<std::mutex> guard(lock);
    if (access.entryIndex < entries.size()) {
        GpuCacheEntry& entry = *entries[access.entryIndex];
        const bool matchingAccess = entry.metadata.id == access.dataId && entry.dataForGpu(access.deviceId) == access.d_data && access.dataBytes == dataBytes;
        if (matchingAccess && access.accessSource == GpuDataAccessSource::CacheHit && entry.cacheState == GpuCacheState::Valid && entry.activeAccesses > 0) {
            --entry.activeAccesses;
        }
        else if (matchingAccess && access.accessSource == GpuDataAccessSource::CacheFill && entry.cacheState == GpuCacheState::Loading && entry.activeAccesses == 1) {
            resetFillEntry(entry);
        }
    }
    access.reset();
}

GpuDataAccess GpuCacheManager::makeFallbackAccess(const FrameMetadata& metadata, const TaskGpuResources& resources) {
    ++activeFallbackAccesses;
    return GpuDataAccess(this, resources.d_input, dataBytes, NO_ENTRY, metadata.id, resources.stream, resources.gpuId, GpuDataAccessSource::TaskFallback);
}

void GpuCacheManager::resetFillEntry(GpuCacheEntry& entry) {
    entry.metadata = FrameMetadata();
    entry.invalidateReplicas();
    entry.cacheState = GpuCacheState::Empty;
    entry.activeAccesses = 0;
    entry.lastUse = 0;
}

std::uint64_t GpuCacheManager::nextUse() {
    ++useSequence;
    if (useSequence != 0) {
        return useSequence;
    }

    std::uint64_t next = 1;
    for (const std::unique_ptr<GpuCacheEntry>& entry : entries) {
        if (entry != nullptr && entry->cacheState != GpuCacheState::Empty) {
            entry->lastUse = next;
            ++next;
        }
    }
    useSequence = next;
    return useSequence;
}
