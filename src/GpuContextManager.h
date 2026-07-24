#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "IAlgo.h"
#include "ThreadSlot.h"

class GpuContext;

struct GpuInfraConfig {
    int threadsPerGpu = 3;
    bool requireNuma = true;
    std::size_t inputBytes = 1024U * 1024U;
};

struct AlgoFactory {
    std::string name;
    std::function<IAlgo*()> make;
};

class GpuContextManager {
public:
    // Serialized cold path: discover GPUs, retain primary contexts, and
    // create one instance of every algorithm per GPU.
    static bool init(const GpuInfraConfig& config, const std::vector<AlgoFactory>& algos);

    // Serialized cold path. Must finish before Graph workers register.
    static bool configure(const AlgoRuntimeInfo& runtime);

    static ThreadSlot* registerThread(int numaHint = -1, int gpuHint = -1);
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
    static bool configured;
};
