#pragma once

#include <cstdint>
#include <vector>

#include "FrameMetadata.h"
#include "Algo/IAlgo.h"

struct FrameCpuAtom {
    FrameCpuAtom(const FrameMetadata& frameMetadata, const AlgoRuntimeInfo& runtime);

    FrameMetadata metadata;
    std::vector<std::uint8_t> data;
    JobResult result;

    FrameCpuAtom(const FrameCpuAtom&) = delete;
    FrameCpuAtom& operator=(const FrameCpuAtom&) = delete;
};
