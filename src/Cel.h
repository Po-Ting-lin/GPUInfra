#pragma once

#include "IAlgo.h"

class Cel final : public IAlgo {
public:
    Cel();
    ~Cel() override;

    bool initStatic(const AlgoStaticInfo& info) override;
    bool configureAndAlloc(const AlgoRuntimeInfo& info) override;
    std::size_t scratchBytesNeeded() const override;
    bool launchKernels(const ThreadSlot& slot, cudaStream_t stream) override;
    bool launchD2H(const ThreadSlot& slot, cudaStream_t stream) override;
    AlgoOutput collectResult(const ThreadSlot& slot) override;
    bool close() override;

    Cel(const Cel&) = delete;
    Cel& operator=(const Cel&) = delete;

private:
    struct Impl;
    Impl* impl;
};
