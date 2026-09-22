#pragma once

#include <cstddef>

struct AlgoRuntimeInfo {
    std::size_t inBytes = 0;
    int sizeFactor = 0;
    int frameW = 0;
    int frameH = 0;
    int frameDtype = 0;
};
