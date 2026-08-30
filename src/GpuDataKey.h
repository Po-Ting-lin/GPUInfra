#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

struct GpuDataKey {
    std::uint64_t frameId = 0;
    std::uint32_t cameraId = 0;

    bool operator==(const GpuDataKey& other) const noexcept {
        return frameId == other.frameId && cameraId == other.cameraId;
    }
};

struct GpuDataKeyHash {
    std::size_t operator()(const GpuDataKey& key) const noexcept {
        std::size_t value = std::hash<std::uint64_t>{}(key.frameId);
        value ^= std::hash<std::uint32_t>{}(key.cameraId) + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) + (value << 6U) + (value >> 2U);
        return value;
    }
};
