#pragma once

#include "IAlgo.h"

class Mi final : public IAlgo {
public:
    Mi();
    ~Mi() override;

    bool initStatic(const AlgoStaticInfo& info) override;
    bool configureAndAlloc(const AlgoRuntimeInfo& info) override;
    std::size_t scratchBytesNeeded() const override;
    bool launchKernels(const ThreadSlot& slot, cudaStream_t stream) override;
    bool launchD2H(const ThreadSlot& slot, cudaStream_t stream) override;
    bool prepareOutput(AlgoOutput& output) const override;
    bool collectResult(const ThreadSlot& slot, AlgoOutput& output) const override;
    bool close() override;

    Mi(const Mi&) = delete;
    Mi& operator=(const Mi&) = delete;

private:
    struct Impl;
    Impl* impl;
};
