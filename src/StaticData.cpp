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

bool sameMetadata(const FrameMetadata& first, const FrameMetadata& second) {
    return first.id == second.id && first.bytes == second.bytes && first.width == second.width && first.height == second.height && first.dtype == second.dtype;
}

}  // namespace

StaticData::~StaticData() {
    release();
}

bool StaticData::init(const StaticDataConfig& config) {
    if (initialized || !registeredFrames.empty() || !frameIdToIndex.empty() || gpuCacheManager.isInitialized() || graphNumaNode >= 0 || config.numaNode < 0 || config.gpuIds.size() != 1 || config.gpuIds.front() < 0 || config.runtime.inBytes == 0) {
        return false;
    }

    std::unordered_map<std::uint64_t, std::size_t> newFrameIdToIndex;
    std::vector<FrameMetadata> newRegisteredFrames;
    try {
        newFrameIdToIndex.reserve(config.frames.size());
        newRegisteredFrames.reserve(config.frames.size());
        for (std::size_t index = 0; index < config.frames.size(); ++index) {
            const FrameMetadata& metadata = config.frames[index];
            if (!matchesRuntime(metadata, config.runtime)) {
                std::fprintf(stderr, "[GPUInfra] invalid StaticData frame definition index=%zu frame_id=%llu\n", index, static_cast<unsigned long long>(metadata.id));
                return false;
            }
            if (!newFrameIdToIndex.emplace(metadata.id, index).second) {
                std::fprintf(stderr, "[GPUInfra] duplicate StaticData frame ID frame_id=%llu\n", static_cast<unsigned long long>(metadata.id));
                return false;
            }
            newRegisteredFrames.push_back(metadata);
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
    frameIdToIndex.swap(newFrameIdToIndex);
    graphNumaNode = config.numaNode;
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
        frameIdToIndex.clear();
        graphNumaNode = -1;
    }
    return ok;
}

std::size_t StaticData::frameCount() const {
    return registeredFrames.size();
}

std::size_t StaticData::gpuCacheEntryCount() const {
    return gpuCacheManager.entryCount();
}

bool StaticData::validateFrame(const FrameMetadata& metadata, int numaNode) const {
    return initialized && graphNumaNode == numaNode && findFrameMetadata(metadata) != nullptr;
}

GpuDataAccess StaticData::acquireGpuData(const FrameMetadata& metadata, const TaskGpuResources& resources) {
    if (!validateFrame(metadata, resources.numaNode)) {
        return GpuDataAccess();
    }
    return gpuCacheManager.acquire(metadata, resources);
}

bool StaticData::isInitialized() const {
    return initialized;
}

const FrameMetadata* StaticData::findFrameMetadata(const FrameMetadata& metadata) const {
    if (!initialized) {
        return nullptr;
    }

    const auto match = frameIdToIndex.find(metadata.id);
    if (match == frameIdToIndex.end() || match->second >= registeredFrames.size()) {
        return nullptr;
    }

    const FrameMetadata& registeredMetadata = registeredFrames[match->second];
    return sameMetadata(registeredMetadata, metadata) ? &registeredMetadata : nullptr;
}
