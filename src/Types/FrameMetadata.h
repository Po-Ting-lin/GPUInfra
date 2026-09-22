#pragma once

#include <cstddef>

#include "DataCache/GpuDataKey.h"

struct FrameMetadata {
    GpuDataKey key;
    std::size_t bytes = 0;
    int width = 0;
    int height = 0;
    int dtype = 0;

    bool operator==(const FrameMetadata& other) const noexcept {
        return key == other.key && bytes == other.bytes && width == other.width && height == other.height && dtype == other.dtype;
    }
};
