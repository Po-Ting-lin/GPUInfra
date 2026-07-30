#pragma once

#include "IAlgo.h"

class Cel final : public IAlgo {
public:
    Cel() = default;
    ~Cel() override;

    bool init(const AlgoRuntimeInfo& info, const ThreadSlot& slot, AlgoOutput& output, std::size_t& scratchBytes) override;
    bool notifyParameter(const AlgoParams& params) override;
    bool launchKernels(const ThreadSlot& slot, cudaStream_t stream) override;
    bool launchD2H(const ThreadSlot& slot, cudaStream_t stream) override;
    bool collectResult(const ThreadSlot& slot, AlgoOutput& output) const override;
    bool close() override;

    Cel(const Cel&) = delete;
    Cel& operator=(const Cel&) = delete;

private:
    int gpuId = -1;
    int numaNode = -1;
    int threadId = -1;
    int frameW = 0;
    int frameH = 0;
    int matrixSize = 0;
    std::size_t inBytes = 0;
    std::size_t matrixBytes = 0;
    std::uint32_t* d_outputMatrix = nullptr;
    std::uint32_t* h_outputMatrix = nullptr;
    bool initialized = false;
    AlgoParams algoParams;
};
