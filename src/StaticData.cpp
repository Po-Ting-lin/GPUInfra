#include "StaticData.h"

#include <cstdio>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "GpuContextManager.h"

namespace {

bool matchesRuntime(const FrameMetadata& metadata, const AlgoRuntimeInfo& runtime) {
    return metadata.bytes != 0 && metadata.bytes == runtime.inBytes && metadata.width == runtime.frameW && metadata.height == runtime.frameH && metadata.dtype == runtime.frameDtype;
}

}  // namespace

StaticData::~StaticData() {
    release();
}

bool StaticData::init(const StaticDataConfig& config) {
    if (initialized || gpuCacheManager.isInitialized() || frameRuntime.inBytes != 0 || config.runtime.inBytes == 0) {
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
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (cacheGpuBytes != 0) {
        CUDA_CHECK(cudaSetDevice(gpuIds.front()), return false);
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes), return false);
        if (cacheGpuBytes > freeBytes) {
            std::fprintf(stderr, "[GPUInfra] insufficient StaticData GPU cache memory gpu=%d required=%zu free=%zu total=%zu\n", gpuIds.front(), cacheGpuBytes, freeBytes, totalBytes);
            return false;
        }
    }
    if (!gpuCacheManager.initialize(gpuIds, config.runtime.inBytes, config.gpuCacheEntries)) {
        return false;
    }

    frameRuntime = config.runtime;
    initialized = true;
    std::fprintf(stderr, "[GPUInfra] StaticData GPU cache plan gpu=%d cache_entries=%zu bytes_per_entry=%zu allocated_bytes=%zu\n", gpuIds.front(), config.gpuCacheEntries, config.runtime.inBytes, cacheGpuBytes);
    return true;
}

bool StaticData::resetCache() {
    return initialized && frameRuntime.inBytes != 0 && gpuCacheManager.resetCache();
}

bool StaticData::execute() const {
    return initialized;
}

bool StaticData::release() {
    initialized = false;

    const bool ok = gpuCacheManager.release();
    if (ok && !gpuCacheManager.isInitialized()) {
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
    return initialized && matchesRuntime(metadata, frameRuntime);
}

GpuDataAccess StaticData::getCacheData(const FrameMetadata& metadata, const GpuCacheRequest& request) {
    if (!validateFrame(metadata)) {
        return GpuDataAccess();
    }
    return gpuCacheManager.getCacheData(metadata, request);
}
