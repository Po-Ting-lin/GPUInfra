#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "ThreadSlot.h"

class GpuContext;

struct GpuInfraConfig {
    int threadsPerGpu = 3;
    bool requireNuma = true;
    std::size_t inputBytes = 1024U * 1024U;
};

class GpuContextManager {
public:
    // Serialized cold path: discover GPUs, retain primary contexts, and prepare registration.
    static bool init(const GpuInfraConfig& config);

    static ThreadSlot* registerThread(int numaHint = -1, int gpuHint = -1);
    static bool prepareThreadScratch(ThreadSlot* slot, std::size_t scratchBytes);
    static void unregisterThread(ThreadSlot* slot);
    static void shutdown();

    static std::size_t gpuCount();
    static int numaNodeForGpu(int gpuId);

private:
    static GpuInfraConfig config;
    static std::vector<GpuContext*> contexts;
    static std::unordered_map<int, std::vector<GpuContext*>> numaToCtxs;
    static std::mutex lock;
    static std::atomic<bool> initialised;
};
