#include "GraphTypes.h"

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

FrameSlot::FrameSlot(std::uint64_t frameId, int graphNumaNode, FramePhase framePhase, const AlgoRuntimeInfo& runtime)
    : id(frameId),
      numaNode(graphNumaNode),
      phase(framePhase),
      input(runtime.inBytes) {
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>((index + frameId) % 251U);
    }

    result.id = id;
    result.ok = true;
    result.outputs.reserve(3);
    result.outputs.push_back(makeOutput("cel", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::CEL_MULTIPLIER)));
    result.outputs.push_back(makeOutput("sdd", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::SDD_MULTIPLIER)));
    result.outputs.push_back(makeOutput("mi", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::MI_MULTIPLIER)));
}
