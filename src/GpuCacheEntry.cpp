#include "GpuCacheEntry.h"

#include <cuda_runtime.h>

#include "CudaCheck.h"

GpuCacheEntry::~GpuCacheEntry() {
    release();
}

bool GpuCacheEntry::initialize(const std::vector<int>& gpuIds, std::size_t bytes) {
    if (initialized || !replicas.empty() || cacheState != GpuCacheState::Empty || activeAccesses != 0 || lastUse != 0 || gpuIds.size() != 1 || gpuIds.front() < 0 || bytes == 0) {
        return false;
    }

    try {
        replicas.reserve(gpuIds.size());
        for (int gpuId : gpuIds) {
            replicas.push_back({gpuId, nullptr, false});
        }
    } catch (...) {
        replicas.clear();
        return false;
    }
    dataBytes = bytes;

    bool ok = true;
    for (GpuReplica& replica : replicas) {
        CUDA_CHECK(cudaSetDevice(replica.gpuId), ok = false);
        if (ok) {
            CUDA_CHECK(cudaMalloc(&replica.d_data, dataBytes), ok = false);
        }
        if (!ok) {
            release();
            return false;
        }
    }

    initialized = true;
    return true;
}

bool GpuCacheEntry::release() {
    if (activeAccesses != 0 || cacheState == GpuCacheState::Loading) {
        return false;
    }

    bool ok = true;
    for (GpuReplica& replica : replicas) {
        if (replica.d_data == nullptr) {
            continue;
        }

        bool deviceReady = true;
        CUDA_CHECK(cudaSetDevice(replica.gpuId), deviceReady = false);
        if (!deviceReady) {
            ok = false;
            continue;
        }

        bool freed = true;
        CUDA_CHECK(cudaFree(replica.d_data), freed = false);
        if (freed) {
            replica.d_data = nullptr;
            replica.valid = false;
        }
        else {
            ok = false;
        }
    }

    bool allReleased = true;
    for (const GpuReplica& replica : replicas) {
        if (replica.d_data != nullptr) {
            allReleased = false;
            break;
        }
    }
    if (allReleased) {
        metadata = FrameMetadata();
        cacheState = GpuCacheState::Empty;
        activeAccesses = 0;
        lastUse = 0;
        replicas.clear();
        dataBytes = 0;
        initialized = false;
    }
    return ok;
}

bool GpuCacheEntry::isInitialized() const {
    return initialized;
}

std::size_t GpuCacheEntry::bytes() const {
    return dataBytes;
}

std::size_t GpuCacheEntry::replicaCount() const {
    return replicas.size();
}

void* GpuCacheEntry::dataForGpu(int gpuId) {
    const GpuCacheEntry* entry = this;
    return const_cast<void*>(entry->dataForGpu(gpuId));
}

const void* GpuCacheEntry::dataForGpu(int gpuId) const {
    for (const GpuReplica& replica : replicas) {
        if (replica.gpuId == gpuId) {
            return replica.d_data;
        }
    }
    return nullptr;
}

bool GpuCacheEntry::replicaValid(int gpuId) const {
    for (const GpuReplica& replica : replicas) {
        if (replica.gpuId == gpuId) {
            return replica.valid;
        }
    }
    return false;
}

void GpuCacheEntry::invalidateReplicas() {
    for (GpuReplica& replica : replicas) {
        replica.valid = false;
    }
}

bool GpuCacheEntry::markReplicaValid(int gpuId) {
    for (GpuReplica& replica : replicas) {
        if (replica.gpuId == gpuId) {
            replica.valid = true;
            return true;
        }
    }
    return false;
}
