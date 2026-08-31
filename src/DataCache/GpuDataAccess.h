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

// Scoped, non-owning access to GPU data selected by GpuCacheManager. Keeping
// this object alive prevents a cache-backed entry from being evicted while
// queued CUDA work may still read the returned pointer; task fallback use is
// tracked for the same lifetime.
//
// Typical call sequence:
//
//   GpuDataAccess access = staticData.acquireGpuData(metadata, resources);
//   if (!access) {
//       return false;
//   }
//
//   bool submitted = true;
//   if (access.needsUpload()) {
//       submitted = enqueueH2D(access.writableData(), resources.stream);
//   }
//   if (submitted) {
//       submitted = enqueueComputeAndD2H(access.data(), resources.stream);
//   }
//   if (!access.complete(submitted)) {
//       return false;
//   }
//
// data() is the read-only pointer consumed by kernels for every valid source.
// writableData() is non-null only for CacheFill and TaskFallback, where the
// caller must enqueue H2D before compute. CacheHit requires no upload.
//
// Submit H2D, compute, and D2H to the same TaskGpuResources stream used during
// acquire. Call complete() exactly once on the normal path, even when enqueueing
// failed. It synchronizes that stream, publishes a successful CacheFill, or
// releases/rolls back the cache or fallback reservation. After complete(), the
// access and its pointers are invalid.
//
// The destructor is an error-path safety net: if complete() was not called, it
// synchronizes already-submitted work and aborts the access. Do not rely on the
// destructor for successful completion because an unfinished CacheFill will
// not be published.
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

    // Required normal-path finish operation; invalidates this access.
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
