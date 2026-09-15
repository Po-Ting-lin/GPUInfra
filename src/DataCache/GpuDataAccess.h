#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "DataCache/GpuDataKey.h"

enum class CacheStatus {
    Invalid,
    CacheHit,
    CacheFill,
    TaskFallback,
};

class GpuCacheManager;

// Scoped, non-owning access to GPU data selected by GpuCacheManager. Keeping
// this object alive prevents a cache-backed entry from being evicted while
// queued CUDA work may still read the returned pointer; task fallback use is
// tracked for the same lifetime.
//
// Typical caller-owned miss handling:
//
//   GpuDataAccess access = cache.getCacheData(metadata, request);
//   const CacheStatus status = access.status();
//   if (status == CacheStatus::Invalid) {
//       return false;
//   }
//   const cudaStream_t stream = access.getStream();
//   bool submitted = true;
//   if (status == CacheStatus::CacheFill || status == CacheStatus::TaskFallback) {
//       submitted = enqueueData(access.writableData(), stream);
//   }
//   if (submitted) {
//       submitted = enqueueComputeAndD2H(access.data(), stream);
//   }
//   if (!access.freeCacheData(submitted)) {
//       return false;
//   }
//
// CacheHit is read-only. CacheFill and TaskFallback expose writableData();
// the caller uploads or computes the entire payload before reading it.
// The cache does not upload, compute, or inspect the payload itself.
//
// Submit all work using this pointer to getStream(), the borrowed request
// stream. Call freeCacheData() once even if submission fails. It synchronizes
// that stream, publishes a successful fill or rolls back a failed fill, and
// ends this lease. No fill is published early. The access and its pointers
// are invalid afterwards; getStream() returns nullptr and status() is Invalid.
//
// The destructor is an error-path safety net: an unfinished access attempts
// synchronization and aborts. It never publishes an unfinished CacheFill.
// Keep the manager, stream, and fallback allocation alive through completion.
class GpuDataAccess {
public:
    GpuDataAccess() = default;
    ~GpuDataAccess();

    explicit operator bool() const;
    const void* data() const;
    void* writableData() const;
    std::size_t bytes() const;
    int gpuId() const;
    CacheStatus status() const;
    cudaStream_t getStream() const;

    // Required normal-path finish operation; invalidates this access.
    // Releases the reservation while retaining the underlying GPU allocation.
    bool freeCacheData(bool submittedSuccessfully);

    GpuDataAccess(const GpuDataAccess&) = delete;
    GpuDataAccess& operator=(const GpuDataAccess&) = delete;
    GpuDataAccess(GpuDataAccess&&) = delete;
    GpuDataAccess& operator=(GpuDataAccess&&) = delete;

private:
    friend class GpuCacheManager;

    GpuDataAccess(GpuCacheManager* accessOwner, void* deviceData, std::size_t bytes, std::size_t index, const GpuDataKey& targetDataKey, cudaStream_t accessStream, int gpuId, CacheStatus accessStatus);
    void reset();

    GpuCacheManager* owner = nullptr;
    void* d_data = nullptr;
    std::size_t dataBytes = 0;
    std::size_t entryIndex = 0;
    GpuDataKey dataKey;
    cudaStream_t stream = nullptr;
    int deviceId = -1;
    CacheStatus accessStatus = CacheStatus::Invalid;
};
