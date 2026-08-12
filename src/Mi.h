#pragma once

#include "IAlgo.h"

class Mi final : public IAlgo {
public:
    Mi() = default;
    ~Mi() override;

    bool init(const AlgoRuntimeInfo& info, const TaskGpuResources& resources, std::size_t& scratchBytes) override;
    bool notifyParameter(const AlgoParams& params) override;
    bool launchKernels(const TaskGpuResources& resources, cudaStream_t stream) override;
    bool launchD2H(const TaskGpuResources& resources, cudaStream_t stream) override;
    bool collectResult(const TaskGpuResources& resources, AlgoOutput& output) const override;
    bool close() override;

    Mi(const Mi&) = delete;
    Mi& operator=(const Mi&) = delete;

private:
    int gpuId = -1;
    int numaNode = -1;
    int resourceId = -1;
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
