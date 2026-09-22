#pragma once

#include <array>
#include <cstddef>

#include <cuda_runtime.h>

enum class StaticGpuDataType {
    DetectionZones,
    DistortionTables
};

enum class StaticGpuDataState {
    Empty,
    Uploading,
    Ready,
    Failed
};

// Graph-copy-owned buffers. The framework serializes all cold-path operations
// and drains every GPU reader before release/reinitialization. There are no
// reader leases. Only Ready data may be shared across task execution streams.
class StaticGpuData {
public:
    StaticGpuData() = default;
    ~StaticGpuData();

    bool initialize(int gpuId, std::size_t detectionZoneBytes, std::size_t distortionBytes);
    // Synchronous with respect to source lifetime: source may be reused on return.
    // Invalid uploads during Uploading poison this initialization attempt.
    bool upload(StaticGpuDataType type, const void* source, std::size_t bytes, std::size_t offset);
    bool finalize();
    const void* data(StaticGpuDataType type) const;
    std::size_t bytes(StaticGpuDataType type) const;
    StaticGpuDataState state() const;
    bool release();

    StaticGpuData(const StaticGpuData&) = delete;
    StaticGpuData& operator=(const StaticGpuData&) = delete;

private:
    struct Buffer {
        void* d_data = nullptr;
        std::size_t capacity = 0;
    };

    const Buffer* bufferFor(StaticGpuDataType type) const;
    bool failInitialization();

    std::array<Buffer, 2> buffers{};
    cudaStream_t uploadStream = nullptr;
    int deviceId = -1;
    StaticGpuDataState currentState = StaticGpuDataState::Empty;
};
