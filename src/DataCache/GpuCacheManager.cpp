#include "Nvtx.h"
#include "CudaCheck.h"

#include "DataCache/GpuCacheManager.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

GpuCacheManager::~GpuCacheManager() {
    release();
}

bool GpuCacheManager::initialize(const std::vector<int>& gpuIds, std::size_t bytes, std::size_t cacheEntryCount, std::chrono::milliseconds configuredWaitTimeout, CacheEvictionPolicy configuredEvictionPolicy) {
    {
        std::lock_guard<std::mutex> guard(lock);
        if (initialized || releasing || !entries.empty() || !eligibleGpuIds.empty() || residencyTable.isInitialized() || !emptyEntries.empty() || leastRecentlyUsed != NO_ENTRY || mostRecentlyUsed != NO_ENTRY || oldestFilled != NO_ENTRY || newestFilled != NO_ENTRY || gpuIds.size() != 1 || gpuIds.front() < 0 || bytes == 0 || configuredWaitTimeout.count() < 0 || (configuredEvictionPolicy != CacheEvictionPolicy::LRU && configuredEvictionPolicy != CacheEvictionPolicy::FIFO)) {
            return false;
        }
    }

    GpuResidencyTable newResidencyTable;
    if (!newResidencyTable.initialize(cacheEntryCount)) {
        return false;
    }

    std::vector<std::unique_ptr<GpuCacheEntry>> newEntries;
    std::vector<std::size_t> newEmptyEntries;
    try {
        newEntries.reserve(cacheEntryCount);
        newEmptyEntries.reserve(cacheEntryCount);
        for (std::size_t index = 0; index < cacheEntryCount; ++index) {
            std::unique_ptr<GpuCacheEntry> entry = std::make_unique<GpuCacheEntry>();
            if (!entry->initialize(gpuIds, bytes)) {
                return false;
            }
            newEntries.push_back(std::move(entry));
        }
        for (std::size_t index = cacheEntryCount; index > 0; --index) {
            newEmptyEntries.push_back(index - 1);
        }
    } catch (...) {
        return false;
    }

    std::lock_guard<std::mutex> guard(lock);
    if (initialized || releasing || !entries.empty() || !eligibleGpuIds.empty() || residencyTable.isInitialized() || !emptyEntries.empty() || leastRecentlyUsed != NO_ENTRY || mostRecentlyUsed != NO_ENTRY || oldestFilled != NO_ENTRY || newestFilled != NO_ENTRY) {
        return false;
    }
    try {
        eligibleGpuIds = gpuIds;
    } catch (...) {
        eligibleGpuIds.clear();
        return false;
    }
    entries = std::move(newEntries);
    residencyTable.swap(newResidencyTable);
    emptyEntries = std::move(newEmptyEntries);
    dataBytes = bytes;
    waitTimeout = configuredWaitTimeout;
    evictionPolicy = configuredEvictionPolicy;
    activeFallbackAccesses = 0;
    statistics = CacheStatistics();
    initialized = true;
    return true;
}

bool GpuCacheManager::resetCache(CacheStatistics* completedStatistics) {
    std::lock_guard<std::mutex> guard(lock);
    if (!initialized || releasing || !canResetLocked()) {
        return false;
    }
    if (completedStatistics != nullptr) {
        *completedStatistics = statistics;
    }
    statistics = CacheStatistics();
    resetEntriesLocked();
    return true;
}

CacheStatistics GpuCacheManager::statisticsSnapshot() const {
    std::lock_guard<std::mutex> guard(lock);
    return statistics;
}

GpuDataAccess GpuCacheManager::getCacheData(const FrameMetadata& metadata, const GpuCacheRequest& request) {
    GPUINFRA_NVTX_SCOPE("Cache.getCacheData");
    const GpuDiagnosticScope diagnosticScope(request.gpuId, request.stream, metadata.key, "Cache.getCacheData");
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> guard(lock);
    if (!initialized || releasing || metadata.bytes != dataBytes || metadata.width <= 0 || metadata.height <= 0 || request.gpuId < 0 || request.stream == nullptr || request.d_fallback == nullptr || request.fallbackBytes < dataBytes || std::find(eligibleGpuIds.begin(), eligibleGpuIds.end(), request.gpuId) == eligibleGpuIds.end()) {
        ++statistics.invalid;
        GPUINFRA_REPORT_FAILURE("invalid cache request or unavailable manager");
        return GpuDataAccess();
    }

    // Include mutex acquisition in the budget; never restart it after a wakeup.
    const std::chrono::milliseconds maximumWait = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::time_point::max() - started);
    const std::chrono::steady_clock::time_point deadline = started + std::min(waitTimeout, maximumWait);
    bool blocked = false;
    bool waited = false;
    std::chrono::nanoseconds requestWait{0};
    while (true) {
        std::size_t residentIndex = NO_ENTRY;
        if (residencyTable.find(metadata.key, residentIndex)) {
            const std::size_t index = residentIndex;
            if (index >= entries.size() || entries[index] == nullptr) {
                ++statistics.invalid;
                GPUINFRA_REPORT_FAILURE("invalid residency index");
                return GpuDataAccess();
            }

            GpuCacheEntry& entry = *entries[index];
            if (!(entry.metadata == metadata)) {
                ++statistics.invalid;
                GPUINFRA_REPORT_FAILURE("cache metadata mismatch");
                return GpuDataAccess();
            }
            if (entry.cacheState == GpuCacheState::Loading) {
                if (!blocked) {
                    ++statistics.firstBlockedLoading;
                    blocked = true;
                }
                if (waitForCacheChange(guard, deadline, waited, requestWait)) {
                    continue;
                }
                return makeFallbackAccess(metadata, request, waitTimeout.count() == 0 ? CacheFallbackReason::LoadingNoWait : CacheFallbackReason::LoadingTimeout);
            }
            if (entry.cacheState != GpuCacheState::Valid) {
                ++statistics.invalid;
                GPUINFRA_REPORT_FAILURE("invalid resident cache state");
                return GpuDataAccess();
            }

            void* deviceData = entry.dataForGpu(request.gpuId);
            if (deviceData == nullptr || !entry.replicaValid(request.gpuId)) {
                // A future multi-GPU implementation can reserve a local replica
                // fill here. In the current one-GPU scope, the caller handles fallback.
                return makeFallbackAccess(metadata, request, CacheFallbackReason::ReplicaUnavailable);
            }
            if ((evictionPolicy == CacheEvictionPolicy::FIFO && !entry.inFilledList) || (entry.activeAccesses == 0 && !removeEvictableEntry(index)) || (entry.activeAccesses != 0 && entry.inEvictableList)) {
                ++statistics.invalid;
                GPUINFRA_REPORT_FAILURE("inconsistent cache reader lease");
                return GpuDataAccess();
            }
            ++statistics.hit;
            ++entry.activeAccesses;
            return GpuDataAccess(this, deviceData, dataBytes, index, metadata.key, request.stream, request.gpuId, CacheStatus::CacheHit);
        }

        bool wasEmpty = false;
        const std::size_t candidateIndex = takeCandidate(wasEmpty);
        if (candidateIndex == NO_ENTRY) {
            if (entries.empty()) {
                return makeFallbackAccess(metadata, request, CacheFallbackReason::CapacityZero);
            }
            if (!blocked) {
                ++statistics.firstBlockedFull;
                blocked = true;
            }
            if (waitForCacheChange(guard, deadline, waited, requestWait)) {
                continue;
            }
            return makeFallbackAccess(metadata, request, waitTimeout.count() == 0 ? CacheFallbackReason::FullNoWait : CacheFallbackReason::FullTimeout);
        }

        GpuCacheEntry& candidate = *entries[candidateIndex];
        void* deviceData = candidate.dataForGpu(request.gpuId);
        if (deviceData == nullptr) {
            restoreCandidate(candidateIndex, wasEmpty);
            ++statistics.invalid;
            GPUINFRA_REPORT_FAILURE("missing candidate GPU allocation");
            return GpuDataAccess();
        }
        if (wasEmpty) {
            if (candidate.cacheState != GpuCacheState::Empty || candidate.activeAccesses != 0 || candidate.inEvictableList) {
                restoreCandidate(candidateIndex, true);
                ++statistics.invalid;
                GPUINFRA_REPORT_FAILURE("invalid empty cache candidate");
                return GpuDataAccess();
            }
        }
        else {
            if (candidate.cacheState != GpuCacheState::Valid || candidate.activeAccesses != 0 || candidate.inEvictableList) {
                restoreCandidate(candidateIndex, false);
                ++statistics.invalid;
                GPUINFRA_REPORT_FAILURE("invalid eviction candidate");
                return GpuDataAccess();
            }
            if (!residencyTable.erase(candidate.metadata.key, candidateIndex)) {
                restoreCandidate(candidateIndex, false);
                ++statistics.invalid;
                GPUINFRA_REPORT_FAILURE("failed residency erase");
                return GpuDataAccess();
            }
        }
        if (!residencyTable.insert(metadata.key, candidateIndex)) {
            if (!wasEmpty) {
                residencyTable.insert(candidate.metadata.key, candidateIndex);
            }
            restoreCandidate(candidateIndex, wasEmpty);
            ++statistics.invalid;
            GPUINFRA_REPORT_FAILURE("failed residency insert");
            return GpuDataAccess();
        }
        if (!wasEmpty && evictionPolicy == CacheEvictionPolicy::FIFO && !removeFilledEntry(candidateIndex)) {
            residencyTable.erase(metadata.key, candidateIndex);
            residencyTable.insert(candidate.metadata.key, candidateIndex);
            restoreCandidate(candidateIndex, false);
            ++statistics.invalid;
            GPUINFRA_REPORT_FAILURE("failed FIFO candidate removal");
            return GpuDataAccess();
        }

        ++statistics.fill;
        if (!wasEmpty) {
            ++statistics.eviction;
        }
        candidate.metadata = metadata;
        candidate.invalidateReplicas();
        candidate.cacheState = GpuCacheState::Loading;
        candidate.activeAccesses = 1;
        candidate.discardWhenUnused = false;
        return GpuDataAccess(this, deviceData, dataBytes, candidateIndex, metadata.key, request.stream, request.gpuId, CacheStatus::CacheFill);
    }
}


bool GpuCacheManager::release() {
    {
        std::lock_guard<std::mutex> guard(lock);
        if (!initialized && entries.empty() && eligibleGpuIds.empty() && !residencyTable.isInitialized() && emptyEntries.empty()) {
            return true;
        }
        if (releasing || !canResetLocked()) {
            return false;
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
        residencyTable.release();
        emptyEntries.clear();
        leastRecentlyUsed = NO_ENTRY;
        mostRecentlyUsed = NO_ENTRY;
        oldestFilled = NO_ENTRY;
        newestFilled = NO_ENTRY;
        dataBytes = 0;
        activeFallbackAccesses = 0;
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

bool GpuCacheManager::waitForCacheChange(std::unique_lock<std::mutex>& guard, std::chrono::steady_clock::time_point deadline, bool& waited, std::chrono::nanoseconds& requestWait) {
    if (entries.empty() || waitTimeout.count() == 0 || std::chrono::steady_clock::now() >= deadline) {
        return false;
    }

    // Count sleepers while the mutex is released so reset/release cannot pass.
    if (!waited) {
        ++statistics.waitedRequests;
        waited = true;
    }
    const std::chrono::steady_clock::time_point waitStarted = std::chrono::steady_clock::now();
    ++waitingAccesses;
    try {
        GPUINFRA_NVTX_SCOPE("Cache.wait");
        cacheChanged.wait_until(guard, deadline);
    } catch (...) {
        --waitingAccesses;
        throw;
    }
    --waitingAccesses;
    const std::chrono::nanoseconds elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStarted);
    requestWait += elapsed;
    statistics.totalWaitTime += elapsed;
    statistics.maxWaitTime = std::max(statistics.maxWaitTime, requestWait);
    // Always recheck residency, including on timeout and spurious wakeups.
    return true;
}

bool GpuCacheManager::canResetLocked() const {
    if (activeFallbackAccesses != 0 || waitingAccesses != 0) {
        return false;
    }
    for (const std::unique_ptr<GpuCacheEntry>& entry : entries) {
        if (entry == nullptr || entry->activeAccesses != 0 || entry->cacheState == GpuCacheState::Loading) {
            return false;
        }
    }
    return true;
}

void GpuCacheManager::resetEntriesLocked() {
    residencyTable.clear();
    emptyEntries.clear();
    leastRecentlyUsed = NO_ENTRY;
    mostRecentlyUsed = NO_ENTRY;
    oldestFilled = NO_ENTRY;
    newestFilled = NO_ENTRY;
    for (std::size_t index = entries.size(); index > 0; --index) {
        GpuCacheEntry& entry = *entries[index - 1];
        entry.metadata = FrameMetadata();
        entry.invalidateReplicas();
        entry.cacheState = GpuCacheState::Empty;
        entry.activeAccesses = 0;
        entry.previousEvictable = NO_ENTRY;
        entry.nextEvictable = NO_ENTRY;
        entry.inEvictableList = false;
        entry.previousFilled = NO_ENTRY;
        entry.nextFilled = NO_ENTRY;
        entry.inFilledList = false;
        entry.discardWhenUnused = false;
        emptyEntries.push_back(index - 1);
    }
}

std::size_t GpuCacheManager::takeCandidate(bool& wasEmpty) {
    if (!emptyEntries.empty()) {
        const std::size_t index = emptyEntries.back();
        emptyEntries.pop_back();
        wasEmpty = true;
        return index;
    }
    if (leastRecentlyUsed == NO_ENTRY) {
        return NO_ENTRY;
    }

    std::size_t index = leastRecentlyUsed;
    if (evictionPolicy == CacheEvictionPolicy::FIFO) {
        // Keep busy entries in FIFO order. Skip them only for this eviction;
        // after their readers finish they retain their original fill age.
        index = oldestFilled;
        while (index != NO_ENTRY && entries[index]->activeAccesses != 0) {
            index = entries[index]->nextFilled;
        }
        if (index == NO_ENTRY) {
            return NO_ENTRY;
        }
    }
    if (!removeEvictableEntry(index)) {
        return NO_ENTRY;
    }
    wasEmpty = false;
    return index;
}

void GpuCacheManager::restoreCandidate(std::size_t index, bool wasEmpty) {
    if (wasEmpty) {
        emptyEntries.push_back(index);
        return;
    }
    addEvictableEntry(index);
}

bool GpuCacheManager::addEvictableEntry(std::size_t index) {
    if (index >= entries.size() || entries[index] == nullptr) {
        return false;
    }

    GpuCacheEntry& entry = *entries[index];
    if (entry.cacheState != GpuCacheState::Valid || entry.activeAccesses != 0 || entry.inEvictableList) {
        return false;
    }

    entry.previousEvictable = mostRecentlyUsed;
    entry.nextEvictable = NO_ENTRY;
    entry.inEvictableList = true;
    if (mostRecentlyUsed != NO_ENTRY) {
        entries[mostRecentlyUsed]->nextEvictable = index;
    }
    else {
        leastRecentlyUsed = index;
    }
    mostRecentlyUsed = index;
    cacheChanged.notify_all();
    return true;
}

bool GpuCacheManager::removeEvictableEntry(std::size_t index) {
    if (index >= entries.size() || entries[index] == nullptr) {
        return false;
    }

    GpuCacheEntry& entry = *entries[index];
    if (!entry.inEvictableList) {
        return false;
    }
    if (entry.previousEvictable != NO_ENTRY) {
        entries[entry.previousEvictable]->nextEvictable = entry.nextEvictable;
    }
    else {
        leastRecentlyUsed = entry.nextEvictable;
    }
    if (entry.nextEvictable != NO_ENTRY) {
        entries[entry.nextEvictable]->previousEvictable = entry.previousEvictable;
    }
    else {
        mostRecentlyUsed = entry.previousEvictable;
    }
    entry.previousEvictable = NO_ENTRY;
    entry.nextEvictable = NO_ENTRY;
    entry.inEvictableList = false;
    return true;
}

bool GpuCacheManager::addFilledEntry(std::size_t index) {
    if (evictionPolicy != CacheEvictionPolicy::FIFO) {
        return true;
    }
    if (index >= entries.size() || entries[index] == nullptr) {
        return false;
    }

    GpuCacheEntry& entry = *entries[index];
    if (entry.cacheState != GpuCacheState::Valid || entry.activeAccesses != 0 || entry.inFilledList) {
        return false;
    }
    entry.previousFilled = newestFilled;
    entry.nextFilled = NO_ENTRY;
    entry.inFilledList = true;
    if (newestFilled != NO_ENTRY) {
        entries[newestFilled]->nextFilled = index;
    }
    else {
        oldestFilled = index;
    }
    newestFilled = index;
    return true;
}

bool GpuCacheManager::removeFilledEntry(std::size_t index) {
    if (index >= entries.size() || entries[index] == nullptr) {
        return false;
    }

    GpuCacheEntry& entry = *entries[index];
    if (!entry.inFilledList) {
        return false;
    }
    if (entry.previousFilled != NO_ENTRY) {
        entries[entry.previousFilled]->nextFilled = entry.nextFilled;
    }
    else {
        oldestFilled = entry.nextFilled;
    }
    if (entry.nextFilled != NO_ENTRY) {
        entries[entry.nextFilled]->previousFilled = entry.previousFilled;
    }
    else {
        newestFilled = entry.previousFilled;
    }
    entry.previousFilled = NO_ENTRY;
    entry.nextFilled = NO_ENTRY;
    entry.inFilledList = false;
    return true;
}

bool GpuCacheManager::releaseReaderEntry(GpuCacheEntry& entry, std::size_t index) {
    if (entry.activeAccesses != 0) {
        return true;
    }
    if (entry.discardWhenUnused) {
        resetEntry(entry, index);
        ++statistics.discard;
        return true;
    }
    return addEvictableEntry(index);
}

bool GpuCacheManager::completeAccess(GpuDataAccess& access, bool succeeded, CacheRetention retention) {
    GPUINFRA_NVTX_SCOPE("Cache.publish_release");
    if (access.owner != this) {
        return false;
    }
    if (access.accessStatus == CacheStatus::TaskFallback) {
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
        valid = entry != nullptr && entry->metadata.key == access.dataKey && entry->dataForGpu(access.deviceId) == access.d_data && access.dataBytes == dataBytes;
    }

    if (access.accessStatus == CacheStatus::CacheHit) {
        valid = valid && entry->cacheState == GpuCacheState::Valid && entry->activeAccesses > 0 && !entry->inEvictableList && entry->replicaValid(access.deviceId);
        if (valid) {
            entry->discardWhenUnused = entry->discardWhenUnused || retention == CacheRetention::Discard;
            --entry->activeAccesses;
            if (!releaseReaderEntry(*entry, access.entryIndex)) {
                valid = false;
            }
        }
    }
    else if (access.accessStatus == CacheStatus::CacheFill) {
        valid = valid && entry->cacheState == GpuCacheState::Loading && entry->activeAccesses == 1 && !entry->inEvictableList;
        if (valid && succeeded) {
            if (entry->markReplicaValid(access.deviceId)) {
                entry->cacheState = GpuCacheState::Valid;
                entry->activeAccesses = 0;
                if (retention == CacheRetention::Discard) {
                    resetEntry(*entry, access.entryIndex);
                    ++statistics.discard;
                }
                else if (!addFilledEntry(access.entryIndex) || !addEvictableEntry(access.entryIndex)) {
                    resetEntry(*entry, access.entryIndex);
                    succeeded = false;
                }
            }
            else {
                resetEntry(*entry, access.entryIndex);
                succeeded = false;
            }
        }
        else if (valid) {
            resetEntry(*entry, access.entryIndex);
        }
        else {
            succeeded = false;
        }
    }
    else {
        valid = false;
    }

    if (access.accessStatus == CacheStatus::CacheFill) {
        if (valid && succeeded) {
            ++statistics.fillSucceeded;
        }
        else {
            ++statistics.fillFailed;
        }
    }
    access.reset();
    return valid && succeeded;
}

void GpuCacheManager::abortAccess(GpuDataAccess& access) {
    if (access.owner != this) {
        return;
    }
    if (access.accessStatus == CacheStatus::TaskFallback) {
        std::lock_guard<std::mutex> guard(lock);
        if (activeFallbackAccesses > 0) {
            --activeFallbackAccesses;
        }
        access.reset();
        return;
    }

    std::lock_guard<std::mutex> guard(lock);
    if (access.accessStatus == CacheStatus::CacheFill) {
        ++statistics.fillFailed;
    }
    if (access.entryIndex < entries.size()) {
        GpuCacheEntry& entry = *entries[access.entryIndex];
        const bool matchingAccess = entry.metadata.key == access.dataKey && entry.dataForGpu(access.deviceId) == access.d_data && access.dataBytes == dataBytes;
        if (matchingAccess && access.accessStatus == CacheStatus::CacheHit && entry.cacheState == GpuCacheState::Valid && entry.activeAccesses > 0 && !entry.inEvictableList) {
            --entry.activeAccesses;
            releaseReaderEntry(entry, access.entryIndex);
        }
        else if (matchingAccess && access.accessStatus == CacheStatus::CacheFill && entry.cacheState == GpuCacheState::Loading && entry.activeAccesses == 1 && !entry.inEvictableList) {
            resetEntry(entry, access.entryIndex);
        }
    }
    access.reset();
}

GpuDataAccess GpuCacheManager::makeFallbackAccess(const FrameMetadata& metadata, const GpuCacheRequest& request, CacheFallbackReason reason) {
    ++statistics.fallback;
    switch (reason) {
        case CacheFallbackReason::CapacityZero: ++statistics.fallbackReasons.capacityZero; break;
        case CacheFallbackReason::LoadingTimeout: ++statistics.fallbackReasons.loadingTimeout; break;
        case CacheFallbackReason::FullTimeout: ++statistics.fallbackReasons.fullTimeout; break;
        case CacheFallbackReason::LoadingNoWait: ++statistics.fallbackReasons.loadingNoWait; break;
        case CacheFallbackReason::FullNoWait: ++statistics.fallbackReasons.fullNoWait; break;
        case CacheFallbackReason::ReplicaUnavailable: ++statistics.fallbackReasons.replicaUnavailable; break;
    }
    ++activeFallbackAccesses;
    return GpuDataAccess(this, request.d_fallback, dataBytes, NO_ENTRY, metadata.key, request.stream, request.gpuId, CacheStatus::TaskFallback);
}

void GpuCacheManager::resetEntry(GpuCacheEntry& entry, std::size_t index) {
    if (entry.inEvictableList) {
        removeEvictableEntry(index);
    }
    if (entry.inFilledList) {
        removeFilledEntry(index);
    }
    residencyTable.erase(entry.metadata.key, index);
    entry.metadata = FrameMetadata();
    entry.invalidateReplicas();
    entry.cacheState = GpuCacheState::Empty;
    entry.activeAccesses = 0;
    entry.previousEvictable = NO_ENTRY;
    entry.nextEvictable = NO_ENTRY;
    entry.inEvictableList = false;
    entry.previousFilled = NO_ENTRY;
    entry.nextFilled = NO_ENTRY;
    entry.inFilledList = false;
    entry.discardWhenUnused = false;
    emptyEntries.push_back(index);
    cacheChanged.notify_all();
}
