#include "StaticData/StaticGpuData.h"

#include "CudaCheck.h"

StaticGpuData::~StaticGpuData() {
    release();
}

bool StaticGpuData::initialize(int gpuId, std::size_t detectionZoneBytes, std::size_t distortionBytes) {
    if (currentState != StaticGpuDataState::Empty) {
        return false;
    }
    currentState = StaticGpuDataState::Failed;
    if (gpuId < 0) {
        return false;
    }
    CUDA_CHECK(cudaSetDevice(gpuId), return false);
    deviceId = gpuId;
    buffers[0].capacity = detectionZoneBytes;
    buffers[1].capacity = distortionBytes;
    if (detectionZoneBytes != 0 || distortionBytes != 0) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking), return failInitialization());
        for (Buffer& buffer : buffers) {
            if (buffer.capacity == 0) {
                continue;
            }
            CUDA_CHECK(cudaMalloc(&buffer.d_data, buffer.capacity), return failInitialization());
            CUDA_CHECK(cudaMemsetAsync(buffer.d_data, 0, buffer.capacity, uploadStream), return failInitialization());
        }
    }
    currentState = StaticGpuDataState::Uploading;
    return true;
}

bool StaticGpuData::failInitialization() {
    release();
    currentState = StaticGpuDataState::Failed;
    return false;
}

bool StaticGpuData::upload(StaticGpuDataType type, const void* source, std::size_t count, std::size_t offset) {
    if (currentState != StaticGpuDataState::Uploading) {
        return false;
    }
    const Buffer* buffer = bufferFor(type);
    if (buffer == nullptr || buffer->capacity == 0 || source == nullptr || count == 0 || offset > buffer->capacity || count > buffer->capacity - offset) {
        currentState = StaticGpuDataState::Failed;
        return false;
    }
    CUDA_CHECK(cudaSetDevice(deviceId), { currentState = StaticGpuDataState::Failed; return false; });
    bool succeeded = true;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<unsigned char*>(buffer->d_data) + offset, source, count, cudaMemcpyHostToDevice, uploadStream), succeeded = false);
    // Drain even after a submission error so earlier work cannot borrow source
    // storage past this call. CUDA execution errors still fail initialization.
    CUDA_CHECK(cudaStreamSynchronize(uploadStream), succeeded = false);
    if (!succeeded) {
        currentState = StaticGpuDataState::Failed;
    }
    return succeeded;
}

bool StaticGpuData::finalize() {
    if (currentState != StaticGpuDataState::Uploading) {
        return false;
    }
    CUDA_CHECK(cudaSetDevice(deviceId), { currentState = StaticGpuDataState::Failed; return false; });
    if (uploadStream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(uploadStream), { currentState = StaticGpuDataState::Failed; return false; });
    }
    currentState = StaticGpuDataState::Ready;
    return true;
}

const StaticGpuData::Buffer* StaticGpuData::bufferFor(StaticGpuDataType type) const {
    switch (type) {
        case StaticGpuDataType::DetectionZones:
            return &buffers[0];
        case StaticGpuDataType::DistortionTables:
            return &buffers[1];
    }
    return nullptr;
}

const void* StaticGpuData::data(StaticGpuDataType type) const {
    const Buffer* buffer = bufferFor(type);
    return currentState == StaticGpuDataState::Ready && buffer != nullptr ? buffer->d_data : nullptr;
}

std::size_t StaticGpuData::bytes(StaticGpuDataType type) const {
    const Buffer* buffer = bufferFor(type);
    return buffer == nullptr ? 0 : buffer->capacity;
}

StaticGpuDataState StaticGpuData::state() const {
    return currentState;
}

bool StaticGpuData::release() {
    bool hasResources = uploadStream != nullptr;
    for (const Buffer& buffer : buffers) {
        hasResources = hasResources || buffer.d_data != nullptr;
    }
    if (hasResources) {
        currentState = StaticGpuDataState::Failed;
        CUDA_CHECK(cudaSetDevice(deviceId), return false);
    }

    bool succeeded = true;
    if (uploadStream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(uploadStream), succeeded = false);
    }
    for (Buffer& buffer : buffers) {
        if (buffer.d_data != nullptr) {
            bool freed = true;
            CUDA_CHECK(cudaFree(buffer.d_data), freed = false);
            if (freed) {
                buffer.d_data = nullptr;
            }
            succeeded = freed && succeeded;
        }
    }
    if (uploadStream != nullptr) {
        bool destroyed = true;
        CUDA_CHECK(cudaStreamDestroy(uploadStream), destroyed = false);
        if (destroyed) {
            uploadStream = nullptr;
        }
        succeeded = destroyed && succeeded;
    }

    bool allReleased = uploadStream == nullptr;
    for (const Buffer& buffer : buffers) {
        allReleased = allReleased && buffer.d_data == nullptr;
    }
    if (allReleased) {
        buffers = {};
        deviceId = -1;
        currentState = StaticGpuDataState::Empty;
    }
    return succeeded;
}
