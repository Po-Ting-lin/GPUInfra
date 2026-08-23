#include "FrameSlot.h"

#include <utility>

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

FrameSlot::FrameSlot(const FrameMetadata& frameMetadata, int graphNumaNode, FramePhase framePhase, const AlgoRuntimeInfo& runtime) {
    bind(frameMetadata, graphNumaNode, framePhase, runtime);
}

bool FrameSlot::bind(const FrameMetadata& frameMetadata, int graphNumaNode, FramePhase framePhase, const AlgoRuntimeInfo& runtime) {
    if (bound || graphNumaNode < 0 || frameMetadata.bytes == 0 || frameMetadata.bytes != runtime.inBytes || frameMetadata.width != runtime.frameW || frameMetadata.height != runtime.frameH || frameMetadata.dtype != runtime.frameDtype) {
        return false;
    }

    JobResult frameResult;
    try {
        frameResult.id = frameMetadata.id;
        frameResult.ok = true;
        frameResult.outputs.reserve(3);
        frameResult.outputs.push_back(makeOutput("cel", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::CEL_MULTIPLIER)));
        frameResult.outputs.push_back(makeOutput("sdd", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::SDD_MULTIPLIER)));
        frameResult.outputs.push_back(makeOutput("mi", ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::MI_MULTIPLIER)));
    } catch (...) {
        return false;
    }

    metadata = frameMetadata;
    numaNode = graphNumaNode;
    phase = framePhase;
    state = FrameState::Prepared;
    result = std::move(frameResult);
    bound = true;
    return true;
}

bool FrameSlot::initializeGpuData(const std::vector<int>& gpuIds) {
    return bound && deviceData.initialize(gpuIds, metadata.bytes);
}

bool FrameSlot::releaseGpuData() {
    return deviceData.release();
}

bool FrameSlot::isBound() const {
    return bound;
}
