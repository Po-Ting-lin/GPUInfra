#include "FrameCpuAtom.h"

#include <cstddef>

FrameCpuAtom::FrameCpuAtom(const FrameMetadata& frameMetadata)
    : metadata(frameMetadata),
      data(frameMetadata.bytes) {
    for (std::size_t index = 0; index < data.size(); ++index) {
        data[index] = static_cast<std::uint8_t>((index + metadata.id) % 251U);
    }
}

bool FrameCpuAtom::matchesMetadata(const FrameMetadata& expected) const {
    return metadata.id == expected.id && metadata.bytes == expected.bytes && metadata.width == expected.width && metadata.height == expected.height && metadata.dtype == expected.dtype;
}
