#pragma once

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
