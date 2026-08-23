#include "FrameCpuAtom.h"

#include <cstddef>

#include "ImageSizing.h"

namespace {

AlgoOutput makeOutput(const char* name, int dimension) {
    AlgoOutput output;
    output.algoName = name;
    output.width = dimension;
    output.height = dimension;
    output.data.resize(ImageSizing::squareBytes(dimension, sizeof(std::uint32_t)));
    return output;
}

}  // namespace

FrameCpuAtom::FrameCpuAtom(const FrameMetadata& frameMetadata, const AlgoRuntimeInfo& runtime)
    : metadata(frameMetadata),
      data(frameMetadata.bytes) {
    result.id = metadata.id;
    result.ok = true;
    result.outputs.reserve(3);
    result.outputs.push_back(makeOutput("cel", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::CEL_MULTIPLIER)));
    result.outputs.push_back(makeOutput("sdd", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::SDD_MULTIPLIER)));
    result.outputs.push_back(makeOutput("mi", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::MI_MULTIPLIER)));

    for (std::size_t index = 0; index < data.size(); ++index) {
        data[index] = static_cast<std::uint8_t>((index + metadata.id) % 251U);
    }
}
