#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

enum class FrameGpuAccessMode {
    Read,
    Upload,
};

class FrameGpuData;

// Scoped, non-owning replica lease. complete() synchronizes and publishes an
// Upload frame ID or validates a Read; destruction aborts incomplete access.
class FrameGpuAccess {
public:
    FrameGpuAccess() = default;
    ~FrameGpuAccess();

    explicit operator bool() const;
    const void* data() const;
    void* writableData() const;
    std::size_t bytes() const;
    int gpuId() const;
    bool complete(bool submittedSuccessfully);

    FrameGpuAccess(const FrameGpuAccess&) = delete;
    FrameGpuAccess& operator=(const FrameGpuAccess&) = delete;
    FrameGpuAccess(FrameGpuAccess&&) = delete;
    FrameGpuAccess& operator=(FrameGpuAccess&&) = delete;

private:
    friend class FrameGpuData;

    FrameGpuAccess(FrameGpuData* accessOwner, void* deviceData, std::size_t bytes, std::size_t index, std::uint64_t targetFrameId, cudaStream_t accessStream, int gpuId, FrameGpuAccessMode accessMode);
    void reset();

    FrameGpuData* owner = nullptr;
    void* d_data = nullptr;
    std::size_t dataBytes = 0;
    std::size_t replicaIndex = 0;
    std::uint64_t frameId = 0;
    cudaStream_t stream = nullptr;
    int deviceId = -1;
    FrameGpuAccessMode mode = FrameGpuAccessMode::Read;
};
