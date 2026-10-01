#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

struct GpuDataKey {
    std::uint64_t frameId = 0;
    std::uint32_t cameraId = 0;
    // Caller-defined payload identity, shared by all callers of this manager.
    // 0 preserves the original single-variant/raw-frame behavior. Distinct
    // preprocessing results need distinct IDs, not unchecked tuple hashes.
    // Never reuse an ID for different bytes while old entries remain cached.
    std::uint64_t variantId = 0;

    bool operator==(const GpuDataKey& other) const noexcept {
        return frameId == other.frameId && cameraId == other.cameraId && variantId == other.variantId;
    }
};

struct GpuDataKeyHash {
    std::size_t operator()(const GpuDataKey& key) const noexcept {
        std::size_t value = std::hash<std::uint64_t>{}(key.frameId);
        value ^= std::hash<std::uint32_t>{}(key.cameraId) + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) + (value << 6U) + (value >> 2U);
        value ^= std::hash<std::uint64_t>{}(key.variantId) + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL) + (value << 6U) + (value >> 2U);
        return value;
    }
};
