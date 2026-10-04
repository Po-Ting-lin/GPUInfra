#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "Context/GpuContextManager.h"
#include "DataCache/GpuCacheManager.h"

namespace result_pipeline {

// Demo-only atom: dataId maps to frameId in a separate result cache.
struct ResultAtom {
    FrameMetadata metadata;
    std::vector<unsigned char> frame;
    std::vector<unsigned char> result;
    std::size_t cpuStages = 0;
    std::uint64_t checksum = 0;
    bool ready = false;
    std::chrono::steady_clock::time_point started;
};

ResultAtom makeAtom(std::uint64_t id, std::size_t bytes);
bool runResultKernel(const void* input, void* output, std::size_t bytes, cudaStream_t stream);

// One stream and independent private/fallback buffers per task instance.
class ResultTaskResources {
public:
    ResultTaskResources() = default;
    ~ResultTaskResources();
    bool load(std::size_t bytes);
    bool unload();
    GpuCacheRequest request() const;
    ResultTaskResources(const ResultTaskResources&) = delete;
    ResultTaskResources& operator=(const ResultTaskResources&) = delete;

    TaskGpuResources gpu;
};

class TaskA {
public:
    bool load(std::size_t bytes);
    bool execute(GpuCacheManager& cache, ResultAtom& atom);
    bool unload();

private:
    ResultTaskResources resources;
};

class DummyCPUTask {
public:
    explicit DummyCPUTask(std::chrono::microseconds delay = std::chrono::microseconds(0));
    bool execute(ResultAtom& atom) const;

private:
    std::chrono::microseconds delay;
};

class TaskB {
public:
    bool load(std::size_t bytes);
    bool execute(GpuCacheManager& cache, const ResultAtom& atom, std::size_t requiredCpuStages, CacheRetention retention);
    bool unload();
    std::uint64_t hits = 0;
    std::uint64_t uploadedBytes = 0;

private:
    ResultTaskResources resources;
};

}  // namespace result_pipeline
