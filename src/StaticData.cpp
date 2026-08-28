#include "StaticData.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <unordered_map>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "TaskGpuResources.h"

namespace {

bool matchesRuntime(const FrameMetadata& metadata, const AlgoRuntimeInfo& runtime) {
    return metadata.bytes != 0 && metadata.bytes == runtime.inBytes && metadata.width == runtime.frameW && metadata.height == runtime.frameH && metadata.dtype == runtime.frameDtype;
}

}  // namespace

StaticData::~StaticData() {
    release();
}

bool StaticData::init(const StaticDataConfig& config) {
    if (initialized || !registeredFrames.empty() || gpuCacheManager.isInitialized() || config.gpuIds.size() != 1 || config.gpuIds.front() < 0 || config.runtime.inBytes == 0) {
        return false;
    }

    std::unordered_map<std::uint64_t, FrameMetadata> newRegisteredFrames;
    try {
        newRegisteredFrames.reserve(config.frames.size());
        for (std::size_t index = 0; index < config.frames.size(); ++index) {
            const FrameMetadata& metadata = config.frames[index];
            if (!matchesRuntime(metadata, config.runtime)) {
                std::fprintf(stderr, "[GPUInfra] invalid StaticData frame definition index=%zu frame_id=%llu\n", index, static_cast<unsigned long long>(metadata.id));
                return false;
            }
            if (!newRegisteredFrames.emplace(metadata.id, metadata).second) {
                std::fprintf(stderr, "[GPUInfra] duplicate StaticData frame ID frame_id=%llu\n", static_cast<unsigned long long>(metadata.id));
                return false;
            }
        }
    } catch (...) {
        return false;
    }

    const std::size_t effectiveCacheEntries = std::min(config.gpuCacheEntries, config.frames.size());
    if (effectiveCacheEntries != 0 && config.runtime.inBytes > std::numeric_limits<std::size_t>::max() / effectiveCacheEntries) {
        return false;
    }
    const std::size_t cacheGpuBytes = effectiveCacheEntries * config.runtime.inBytes;
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (cacheGpuBytes != 0) {
        CUDA_CHECK(cudaSetDevice(config.gpuIds.front()), return false);
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes), return false);
        if (cacheGpuBytes > freeBytes) {
            std::fprintf(stderr, "[GPUInfra] insufficient StaticData GPU cache memory gpu=%d required=%zu free=%zu total=%zu\n", config.gpuIds.front(), cacheGpuBytes, freeBytes, totalBytes);
            return false;
        }
    }
    if (!gpuCacheManager.initialize(config.gpuIds, config.runtime.inBytes, effectiveCacheEntries)) {
        return false;
    }

    registeredFrames.swap(newRegisteredFrames);
    initialized = true;
    std::fprintf(stderr, "[GPUInfra] StaticData GPU cache plan gpu=%d registered_frames=%zu cache_entries=%zu bytes_per_entry=%zu allocated_bytes=%zu\n", config.gpuIds.front(), registeredFrames.size(), effectiveCacheEntries, config.runtime.inBytes, cacheGpuBytes);
    return true;
}

bool StaticData::execute() const {
    return initialized;
}

bool StaticData::release() {
    initialized = false;

    const bool ok = gpuCacheManager.release();
    if (ok && !gpuCacheManager.isInitialized()) {
        registeredFrames.clear();
    }
    return ok;
}

std::size_t StaticData::frameCount() const {
    return registeredFrames.size();
}

std::size_t StaticData::gpuCacheEntryCount() const {
    return gpuCacheManager.entryCount();
}

bool StaticData::validateFrame(const FrameMetadata& metadata) const {
    if (!initialized) {
        return false;
    }

    const auto match = registeredFrames.find(metadata.id);
    return match != registeredFrames.end() && match->second == metadata;
}

GpuDataAccess StaticData::acquireGpuData(const FrameMetadata& metadata, const TaskGpuResources& resources) {
    if (!validateFrame(metadata)) {
        return GpuDataAccess();
    }
    return gpuCacheManager.acquire(metadata, resources);
}
