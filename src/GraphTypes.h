#pragma once

#include <cstdint>
#include <vector>

#include "IAlgo.h"

enum class ExecutionModel {
    Batched,
    Interleaved,
};

enum class FramePhase {
    Warmup,
    Timed,
};

enum class FrameState {
    Prepared,
    Ready,
    Executing,
    Completed,
    Failed,
    Cancelled,
};

struct FrameSlot {
    FrameSlot(std::uint64_t frameId, int graphNumaNode, FramePhase framePhase, const AlgoRuntimeInfo& runtime);

    std::uint64_t id = 0;
    int numaNode = -1;
    FramePhase phase = FramePhase::Warmup;
    FrameState state = FrameState::Prepared;
    std::vector<std::uint8_t> input;
    JobResult result;
};
