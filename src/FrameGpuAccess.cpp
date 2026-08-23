#include "FrameGpuAccess.h"

#include "CudaCheck.h"
#include "FrameGpuCache.h"

FrameGpuAccess::FrameGpuAccess(FrameGpuCache* accessOwner, void* deviceData, std::size_t bytes, std::size_t index, std::uint64_t targetFrameId, cudaStream_t accessStream, int gpuId, FrameGpuAccessSource source)
    : owner(accessOwner),
      d_data(deviceData),
      dataBytes(bytes),
      slotIndex(index),
      frameId(targetFrameId),
      stream(accessStream),
      deviceId(gpuId),
      accessSource(source) {}

FrameGpuAccess::~FrameGpuAccess() {
    if (owner == nullptr) {
        return;
    }

    // Already-submitted CUDA work must not outlive the lease on an early exit.
    CUDA_CHECK(cudaStreamSynchronize(stream), );
    owner->abortAccess(*this);
}

FrameGpuAccess::operator bool() const {
    return owner != nullptr;
}

const void* FrameGpuAccess::data() const {
    return d_data;
}

void* FrameGpuAccess::writableData() const {
    return needsUpload() ? d_data : nullptr;
}

std::size_t FrameGpuAccess::bytes() const {
    return dataBytes;
}

int FrameGpuAccess::gpuId() const {
    return deviceId;
}

bool FrameGpuAccess::needsUpload() const {
    return accessSource == FrameGpuAccessSource::CacheFill || accessSource == FrameGpuAccessSource::TaskFallback;
}

FrameGpuAccessSource FrameGpuAccess::source() const {
    return accessSource;
}

bool FrameGpuAccess::complete(bool submittedSuccessfully) {
    if (owner == nullptr || stream == nullptr) {
        return false;
    }

    bool succeeded = submittedSuccessfully;
    CUDA_CHECK(cudaStreamSynchronize(stream), succeeded = false);
    return owner->completeAccess(*this, succeeded);
}

void FrameGpuAccess::reset() {
    owner = nullptr;
    d_data = nullptr;
    dataBytes = 0;
    slotIndex = 0;
    frameId = 0;
    stream = nullptr;
    deviceId = -1;
    accessSource = FrameGpuAccessSource::Invalid;
}
