#pragma once

#include "IAlgo.h"

class Cel final : public IAlgo {
public:
    Cel();
    ~Cel() override;

    bool initStatic(const AlgoStaticInfo& info) override;
    bool configure(const AlgoRuntimeInfo& info) override;
    bool allocateOutputBuffers(const ThreadSlot& slot) override;
    std::size_t scratchBytesNeeded() const override;
    bool launchKernels(const ThreadSlot& slot, cudaStream_t stream) override;
    bool launchD2H(const ThreadSlot& slot, cudaStream_t stream) override;
    bool prepareOutput(AlgoOutput& output) const override;
    bool collectResult(const ThreadSlot& slot, AlgoOutput& output) const override;
    bool close() override;

    Cel(const Cel&) = delete;
    Cel& operator=(const Cel&) = delete;

private:
    struct Impl;
    Impl* impl;
};
