#include "FrameGpuData.h"

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "TaskGpuResources.h"

namespace {

bool validAccessMode(FrameGpuAccessMode mode) {
    return mode == FrameGpuAccessMode::Read || mode == FrameGpuAccessMode::Upload;
}

}  // namespace

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
            replicas.push_back({gpuId, nullptr, 0, false});
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

    residentFrameId = 0;
    payloadValid = false;
    initialized = true;
    return true;
}

FrameGpuAccess FrameGpuData::acquire(FrameGpuAccessMode mode, std::uint64_t frameId, const TaskGpuResources& resources) {
    if (!initialized || !validAccessMode(mode) || resources.gpuId < 0 || resources.stream == nullptr) {
        return FrameGpuAccess();
    }

    std::size_t replicaIndex = replicas.size();
    for (std::size_t index = 0; index < replicas.size(); ++index) {
        if (replicas[index].gpuId == resources.gpuId) {
            replicaIndex = index;
            break;
        }
    }
    if (replicaIndex == replicas.size()) {
        return FrameGpuAccess();
    }

    const FrameGpuReplica& replica = replicas[replicaIndex];
    if (mode == FrameGpuAccessMode::Read && (!payloadValid || residentFrameId != frameId || !replica.valid || replica.frameId != frameId)) {
        // A future two-GPU implementation will enqueue replica migration here.
        return FrameGpuAccess();
    }
    if (mode == FrameGpuAccessMode::Upload) {
        payloadValid = false;
        for (FrameGpuReplica& currentReplica : replicas) {
            currentReplica.valid = false;
        }
    }

    return FrameGpuAccess(this, replica.d_data, dataBytes, replicaIndex, frameId, resources.stream, replica.gpuId, mode);
}

bool FrameGpuData::completeAccess(FrameGpuAccess& access, bool succeeded) {
    if (access.owner != this) {
        return false;
    }

    bool valid = initialized && access.replicaIndex < replicas.size();
    if (valid) {
        const FrameGpuReplica& replica = replicas[access.replicaIndex];
        valid = access.d_data == replica.d_data && access.dataBytes == dataBytes && access.deviceId == replica.gpuId;
    }

    if (valid && succeeded) {
        FrameGpuReplica& replica = replicas[access.replicaIndex];
        if (access.mode == FrameGpuAccessMode::Read) {
            valid = payloadValid && residentFrameId == access.frameId && replica.valid && replica.frameId == access.frameId;
        }
        else if (access.mode == FrameGpuAccessMode::Upload) {
            residentFrameId = access.frameId;
            payloadValid = true;
            replica.frameId = access.frameId;
            replica.valid = true;
        }
        else {
            valid = false;
        }
    }

    access.reset();
    return valid && succeeded;
}

void FrameGpuData::abortAccess(FrameGpuAccess& access) {
    if (access.owner == this) {
        access.reset();
    }
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
        residentFrameId = 0;
        payloadValid = false;
        initialized = false;
    }
    return ok;
}

bool FrameGpuData::isInitialized() const {
    return initialized;
}

bool FrameGpuData::hasData(std::uint64_t frameId) const {
    return initialized && payloadValid && residentFrameId == frameId;
}

std::size_t FrameGpuData::bytes() const {
    return dataBytes;
}

std::size_t FrameGpuData::replicaCount() const {
    return replicas.size();
}

std::uint64_t FrameGpuData::frameId() const {
    return residentFrameId;
}
