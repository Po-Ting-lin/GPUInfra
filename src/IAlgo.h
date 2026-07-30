#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

struct ThreadSlot;

struct AlgoParams {
    std::string name;
    std::vector<std::uint8_t> blob;
};

struct AlgoRuntimeInfo {
    std::size_t inBytes = 0;
    int sizeFactor = 0;
    int frameW = 0;
    int frameH = 0;
    int frameDtype = 0;
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

    virtual bool init(const AlgoRuntimeInfo& info, const ThreadSlot& slot, AlgoOutput& output, std::size_t& scratchBytes) = 0;
    virtual bool notifyParameter(const AlgoParams& params) = 0;

    // Enqueue compute only. D2H is deliberately kept out of this method so
    // the infrastructure can enqueue every algorithm's compute first.
    virtual bool launchKernels(const ThreadSlot& slot, cudaStream_t stream) = 0;

    // Enqueue this algorithm's output transfer after the whole kernel batch.
    virtual bool launchD2H(const ThreadSlot& slot, cudaStream_t stream) = 0;

    // Copy into the prepared output without resizing or allocating.
    virtual bool collectResult(const ThreadSlot& slot, AlgoOutput& output) const = 0;
    virtual bool close() = 0;
};
