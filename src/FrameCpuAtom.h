#pragma once

#include <cstdint>
#include <vector>

#include "FrameMetadata.h"

struct FrameCpuAtom {
    explicit FrameCpuAtom(const FrameMetadata& frameMetadata);

    bool matchesMetadata(const FrameMetadata& expected) const;

    FrameMetadata metadata;
    std::vector<std::uint8_t> data;

    FrameCpuAtom(const FrameCpuAtom&) = delete;
    FrameCpuAtom& operator=(const FrameCpuAtom&) = delete;
};
