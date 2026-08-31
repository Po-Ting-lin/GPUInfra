#include "DataCache/GpuDataAccess.h"

#include "CudaCheck.h"
#include "DataCache/GpuCacheManager.h"

GpuDataAccess::GpuDataAccess(GpuCacheManager* accessOwner, void* deviceData, std::size_t bytes, std::size_t index, const GpuDataKey& targetDataKey, cudaStream_t accessStream, int gpuId, GpuDataAccessSource source)
    : owner(accessOwner),
      d_data(deviceData),
      dataBytes(bytes),
      entryIndex(index),
      dataKey(targetDataKey),
      stream(accessStream),
      deviceId(gpuId),
      accessSource(source) {}

GpuDataAccess::~GpuDataAccess() {
    if (owner == nullptr) {
        return;
    }

    // Already-submitted CUDA work must not outlive the access on an early exit.
    CUDA_CHECK(cudaStreamSynchronize(stream), );
    owner->abortAccess(*this);
}

GpuDataAccess::operator bool() const {
    return owner != nullptr;
}

const void* GpuDataAccess::data() const {
    return d_data;
}

void* GpuDataAccess::writableData() const {
    return needsUpload() ? d_data : nullptr;
}

std::size_t GpuDataAccess::bytes() const {
    return dataBytes;
}

int GpuDataAccess::gpuId() const {
    return deviceId;
}

bool GpuDataAccess::needsUpload() const {
    return accessSource == GpuDataAccessSource::CacheFill || accessSource == GpuDataAccessSource::TaskFallback;
}

GpuDataAccessSource GpuDataAccess::source() const {
    return accessSource;
}

bool GpuDataAccess::complete(bool submittedSuccessfully) {
    if (owner == nullptr || stream == nullptr) {
        return false;
    }

    bool succeeded = submittedSuccessfully;
    CUDA_CHECK(cudaStreamSynchronize(stream), succeeded = false);
    return owner->completeAccess(*this, succeeded);
}

void GpuDataAccess::reset() {
    owner = nullptr;
    d_data = nullptr;
    dataBytes = 0;
    entryIndex = 0;
    dataKey = GpuDataKey();
    stream = nullptr;
    deviceId = -1;
    accessSource = GpuDataAccessSource::Invalid;
}
