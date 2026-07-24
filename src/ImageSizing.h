#pragma once

#include <cstddef>

namespace ImageSizing {

constexpr int DEFAULT_FACTOR = 128;
constexpr int MIN_FACTOR = 16;
constexpr int MAX_FACTOR = 256;
constexpr int FACTOR_ALIGNMENT = 16;
constexpr int INPUT_MULTIPLIER = 8;
constexpr int CEL_MULTIPLIER = 2;
constexpr int SDD_MULTIPLIER = 3;
constexpr int MI_MULTIPLIER = 3;

inline bool isValidFactor(int factor) {
    return factor >= MIN_FACTOR && factor <= MAX_FACTOR && factor % FACTOR_ALIGNMENT == 0;
}

inline int scaledDimension(int factor, int multiplier) {
    return factor * multiplier;
}

inline std::size_t squareBytes(int dimension, std::size_t elementBytes) {
    return static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension) * elementBytes;
}

}  // namespace ImageSizing
