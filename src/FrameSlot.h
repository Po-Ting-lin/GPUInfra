#pragma once

#include <vector>

#include "FrameGpuData.h"
#include "FrameMetadata.h"
#include "GraphTypes.h"
#include "IAlgo.h"

struct FrameSlot {
    FrameSlot() = default;
    FrameSlot(const FrameMetadata& frameMetadata, int graphNumaNode, FramePhase framePhase, const AlgoRuntimeInfo& runtime);

    bool bind(const FrameMetadata& frameMetadata, int graphNumaNode, FramePhase framePhase, const AlgoRuntimeInfo& runtime);
    bool initializeGpuData(const std::vector<int>& gpuIds);
    bool releaseGpuData();
    bool isBound() const;

    FrameMetadata metadata;
    int numaNode = -1;
    FramePhase phase = FramePhase::Warmup;
    FrameState state = FrameState::Prepared;
    FrameGpuData deviceData;
    JobResult result;

    FrameSlot(const FrameSlot&) = delete;
    FrameSlot& operator=(const FrameSlot&) = delete;

private:
    bool bound = false;
};
