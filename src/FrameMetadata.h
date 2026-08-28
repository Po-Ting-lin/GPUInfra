#pragma once

#include <cstddef>
#include <cstdint>

struct FrameMetadata {
    std::uint64_t id = 0;
    std::size_t bytes = 0;
    int width = 0;
    int height = 0;
    int dtype = 0;

    bool operator==(const FrameMetadata& other) const {
        return id == other.id && bytes == other.bytes && width == other.width && height == other.height && dtype == other.dtype;
    }
};
