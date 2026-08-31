#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "DataCache/GpuDataKey.h"

enum class GpuDataAccessSource {
    Invalid,
    CacheHit,
    CacheFill,
    TaskFallback,
};

class GpuCacheManager;

// Scoped, non-owning device-data access. complete() synchronizes the task
// stream before a cache fill is published or a cache reader is released.
class GpuDataAccess {
public:
    GpuDataAccess() = default;
    ~GpuDataAccess();

    explicit operator bool() const;
    const void* data() const;
    void* writableData() const;
    std::size_t bytes() const;
    int gpuId() const;
    bool needsUpload() const;
    GpuDataAccessSource source() const;
    bool complete(bool submittedSuccessfully);

    GpuDataAccess(const GpuDataAccess&) = delete;
    GpuDataAccess& operator=(const GpuDataAccess&) = delete;
    GpuDataAccess(GpuDataAccess&&) = delete;
    GpuDataAccess& operator=(GpuDataAccess&&) = delete;

private:
    friend class GpuCacheManager;

    GpuDataAccess(GpuCacheManager* accessOwner, void* deviceData, std::size_t bytes, std::size_t index, const GpuDataKey& targetDataKey, cudaStream_t accessStream, int gpuId, GpuDataAccessSource accessSource);
    void reset();

    GpuCacheManager* owner = nullptr;
    void* d_data = nullptr;
    std::size_t dataBytes = 0;
    std::size_t entryIndex = 0;
    GpuDataKey dataKey;
    cudaStream_t stream = nullptr;
    int deviceId = -1;
    GpuDataAccessSource accessSource = GpuDataAccessSource::Invalid;
};
