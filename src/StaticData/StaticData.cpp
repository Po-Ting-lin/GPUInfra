#include "StaticData/StaticData.h"

#include <cstdio>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "Context/GpuContextManager.h"

namespace {

bool matchesRuntime(const FrameMetadata& metadata, const AlgoRuntimeInfo& runtime) {
    return metadata.bytes != 0 && metadata.bytes == runtime.inBytes && metadata.width == runtime.frameW && metadata.height == runtime.frameH && metadata.dtype == runtime.frameDtype;
}

}  // namespace

StaticData::~StaticData() {
    release();
}

bool StaticData::init(const StaticDataConfig& config) {
    if (initialized || gpuStaticData.state() != StaticGpuDataState::Empty || gpuCacheManager.isInitialized() || frameRuntime.inBytes != 0 || config.runtime.inBytes == 0) {
        return false;
    }

    std::vector<int> gpuIds;
    if (!GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds)) {
        return false;
    }

    if (config.gpuCacheEntries != 0 && config.runtime.inBytes > std::numeric_limits<std::size_t>::max() / config.gpuCacheEntries) {
        return false;
    }
    const std::size_t cacheGpuBytes = config.gpuCacheEntries * config.runtime.inBytes;
    const std::size_t maximumBytes = std::numeric_limits<std::size_t>::max();
    if (config.detectionZoneBytes > maximumBytes - cacheGpuBytes || config.distortionBytes > maximumBytes - cacheGpuBytes - config.detectionZoneBytes) {
        return false;
    }
    const std::size_t totalGpuBytes = cacheGpuBytes + config.detectionZoneBytes + config.distortionBytes;
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (totalGpuBytes != 0) {
        CUDA_CHECK(cudaSetDevice(gpuIds.front()), return false);
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes), return false);
        if (totalGpuBytes > freeBytes) {
            std::fprintf(stderr, "[GPUInfra] insufficient StaticData GPU memory gpu=%d required=%zu free=%zu total=%zu\n", gpuIds.front(), totalGpuBytes, freeBytes, totalBytes);
            return false;
        }
    }
    if (!gpuCacheManager.initialize(gpuIds, config.runtime.inBytes, config.gpuCacheEntries, config.gpuCacheWaitTimeout)) {
        return false;
    }

    if (!gpuStaticData.initialize(gpuIds.front(), config.detectionZoneBytes, config.distortionBytes)) {
        release();
        return false;
    }
    // Preserve the frame-only lifecycle when both static regions are disabled.
    if (config.detectionZoneBytes == 0 && config.distortionBytes == 0 && !gpuStaticData.finalize()) {
        release();
        return false;
    }

    frameRuntime = config.runtime;
    initialized = true;
    std::fprintf(stderr, "[GPUInfra] StaticData GPU cache plan gpu=%d cache_entries=%zu bytes_per_entry=%zu allocated_bytes=%zu\n", gpuIds.front(), config.gpuCacheEntries, config.runtime.inBytes, cacheGpuBytes);
    return true;
}

bool StaticData::uploadStaticData(StaticGpuDataType type, const void* source, std::size_t bytes, std::size_t offset) {
    return initialized && gpuStaticData.upload(type, source, bytes, offset);
}

bool StaticData::finalizeStaticData() {
    return initialized && gpuStaticData.finalize();
}

const void* StaticData::staticGpuData(StaticGpuDataType type) const {
    return initialized ? gpuStaticData.data(type) : nullptr;
}

std::size_t StaticData::staticGpuDataBytes(StaticGpuDataType type) const {
    return gpuStaticData.bytes(type);
}

bool StaticData::resetCache(CacheStatistics* completedStatistics) {
    return initialized && frameRuntime.inBytes != 0 && gpuCacheManager.resetCache(completedStatistics);
}

CacheStatistics StaticData::cacheStatisticsSnapshot() const {
    return gpuCacheManager.statisticsSnapshot();
}

bool StaticData::execute() const {
    return initialized && gpuStaticData.state() == StaticGpuDataState::Ready;
}

bool StaticData::release() {
    // Reject live cache users before touching static buffers. The framework must
    // also drain readers that only borrow static pointers (there are no leases).
    if (!gpuCacheManager.release()) {
        return false;
    }
    initialized = false;
    const bool ok = gpuStaticData.release();
    if (gpuStaticData.state() == StaticGpuDataState::Empty) {
        frameRuntime = AlgoRuntimeInfo();
    }
    return ok;
}

bool StaticData::isInitialized() const {
    return initialized;
}

std::size_t StaticData::gpuCacheEntryCount() const {
    return gpuCacheManager.entryCount();
}

bool StaticData::validateFrame(const FrameMetadata& metadata) const {
    return execute() && matchesRuntime(metadata, frameRuntime);
}

GpuDataAccess StaticData::getCacheData(const FrameMetadata& metadata, const GpuCacheRequest& request) {
    if (!validateFrame(metadata)) {
        return GpuDataAccess();
    }
    return gpuCacheManager.getCacheData(metadata, request);
}
