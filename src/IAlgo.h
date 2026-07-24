#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

struct ThreadSlot;

struct AlgoStaticInfo {
    int gpuId = -1;
    int numaNode = -1;
};

struct AlgoParams {
    std::string name;
    std::vector<std::uint8_t> blob;
};

struct AlgoRuntimeInfo {
    int numThreads = 0;
    std::size_t inBytes = 0;
    int sizeFactor = 0;
    int frameW = 0;
    int frameH = 0;
    int frameDtype = 0;
    AlgoParams params;
};

struct AlgoOutput {
    std::string algoName;
    std::vector<std::uint8_t> data;
    int width = 0;
    int height = 0;
};

struct JobResult {
    std::uint64_t id = 0;
    bool ok = true;
    std::vector<AlgoOutput> outputs;
};

class IAlgo {
public:
    virtual ~IAlgo() = default;

    virtual bool initStatic(const AlgoStaticInfo& info) = 0;
    virtual bool configureAndAlloc(const AlgoRuntimeInfo& info) = 0;
    virtual std::size_t scratchBytesNeeded() const = 0;

    // Enqueue compute only. D2H is deliberately kept out of this method so
    // the infrastructure can enqueue every algorithm's compute first.
    virtual bool launchKernels(const ThreadSlot& slot, cudaStream_t stream) = 0;

    // Enqueue this algorithm's output transfer after the whole kernel batch.
    virtual bool launchD2H(const ThreadSlot& slot, cudaStream_t stream) = 0;

    // Allocate and describe one reusable output during worker setup.
    virtual bool prepareOutput(AlgoOutput& output) const = 0;

    // Copy into the prepared output without resizing or allocating.
    virtual bool collectResult(const ThreadSlot& slot, AlgoOutput& output) const = 0;
    virtual bool close() = 0;
};
