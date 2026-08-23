#pragma once

enum class ExecutionModel {
    Batched,
    Interleaved,
};

enum class FramePhase {
    Warmup,
    Timed,
};
