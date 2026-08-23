#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

enum class FrameGpuAccessSource {
    Invalid,
    CacheHit,
    CacheFill,
    TaskFallback,
};

class FrameGpuCache;

// Scoped, non-owning device-data lease. complete() synchronizes the task stream
// before a cache fill is published or a cache reader is released.
class FrameGpuAccess {
public:
    FrameGpuAccess() = default;
    ~FrameGpuAccess();

    explicit operator bool() const;
    const void* data() const;
    void* writableData() const;
    std::size_t bytes() const;
    int gpuId() const;
    bool needsUpload() const;
    FrameGpuAccessSource source() const;
    bool complete(bool submittedSuccessfully);

    FrameGpuAccess(const FrameGpuAccess&) = delete;
    FrameGpuAccess& operator=(const FrameGpuAccess&) = delete;
    FrameGpuAccess(FrameGpuAccess&&) = delete;
    FrameGpuAccess& operator=(FrameGpuAccess&&) = delete;

private:
    friend class FrameGpuCache;

    FrameGpuAccess(FrameGpuCache* accessOwner, void* deviceData, std::size_t bytes, std::size_t index, std::uint64_t targetFrameId, cudaStream_t accessStream, int gpuId, FrameGpuAccessSource accessSource);
    void reset();

    FrameGpuCache* owner = nullptr;
    void* d_data = nullptr;
    std::size_t dataBytes = 0;
    std::size_t slotIndex = 0;
    std::uint64_t frameId = 0;
    cudaStream_t stream = nullptr;
    int deviceId = -1;
    FrameGpuAccessSource accessSource = FrameGpuAccessSource::Invalid;
};
