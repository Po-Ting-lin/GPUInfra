#include "FrameGpuData.h"

#include <cuda_runtime.h>

#include "CudaCheck.h"

FrameGpuData::~FrameGpuData() {
    release();
}

bool FrameGpuData::initialize(const std::vector<int>& gpuIds, std::size_t bytes) {
    if (initialized || !replicas.empty() || gpuIds.size() != 1 || gpuIds.front() < 0 || bytes == 0) {
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
    for (FrameGpuReplica& replica : replicas) {
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

bool FrameGpuData::release() {
    bool ok = true;
    for (FrameGpuReplica& replica : replicas) {
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
    for (const FrameGpuReplica& replica : replicas) {
        if (replica.d_data != nullptr) {
            allReleased = false;
            break;
        }
    }
    if (allReleased) {
        replicas.clear();
        dataBytes = 0;
        initialized = false;
    }
    return ok;
}

bool FrameGpuData::isInitialized() const {
    return initialized;
}

std::size_t FrameGpuData::bytes() const {
    return dataBytes;
}

std::size_t FrameGpuData::replicaCount() const {
    return replicas.size();
}

void* FrameGpuData::dataForGpu(int gpuId) {
    const FrameGpuData* data = this;
    return const_cast<void*>(data->dataForGpu(gpuId));
}

const void* FrameGpuData::dataForGpu(int gpuId) const {
    for (const FrameGpuReplica& replica : replicas) {
        if (replica.gpuId == gpuId) {
            return replica.d_data;
        }
    }
    return nullptr;
}

bool FrameGpuData::replicaValid(int gpuId) const {
    for (const FrameGpuReplica& replica : replicas) {
        if (replica.gpuId == gpuId) {
            return replica.valid;
        }
    }
    return false;
}

void FrameGpuData::invalidateReplicas() {
    for (FrameGpuReplica& replica : replicas) {
        replica.valid = false;
    }
}

bool FrameGpuData::markReplicaValid(int gpuId) {
    for (FrameGpuReplica& replica : replicas) {
        if (replica.gpuId == gpuId) {
            replica.valid = true;
            return true;
        }
    }
    return false;
}
