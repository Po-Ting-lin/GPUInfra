#include "GpuContextManager.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include "CudaCheck.h"
#include "GpuContext.h"

GpuInfraConfig GpuContextManager::config;
std::vector<GpuContext*> GpuContextManager::contexts;
std::mutex GpuContextManager::lock;
std::atomic<bool> GpuContextManager::initialised{false};

namespace {

bool addCpuRange(cpu_set_t& set, const std::string& token) {
    const std::size_t separator = token.find('-');
    try {
        const int first = std::stoi(token.substr(0, separator));
        const int last = separator == std::string::npos ? first : std::stoi(token.substr(separator + 1));
        if (first < 0 || last < first) {
            return false;
        }
        for (int cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu) {
            CPU_SET(cpu, &set);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool nodeCpuSet(int node, cpu_set_t& set) {
    std::ifstream input("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string cpuList;
    if (!(input >> cpuList)) {
        return false;
    }

    CPU_ZERO(&set);
    std::istringstream tokens(cpuList);
    std::string token;
    while (std::getline(tokens, token, ',')) {
        if (!addCpuRange(set, token)) {
            return false;
        }
    }
    return true;
}

bool pinToNumaNode(int node) {
    if (node < 0) {
        return false;
    }

    cpu_set_t nodeSet;
    cpu_set_t allowedSet;
    if (!nodeCpuSet(node, nodeSet) || pthread_getaffinity_np(pthread_self(), sizeof(allowedSet), &allowedSet) != 0) {
        return false;
    }

    // Respect a container or cgroup's existing CPU allowance.
    CPU_AND(&nodeSet, &nodeSet, &allowedSet);
    if (CPU_COUNT(&nodeSet) == 0) {
        return false;
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(nodeSet), &nodeSet) == 0;
}

int probeNumaNodeOfGpu(int gpuId) {
    char busId[32]{};
    CUDA_CHECK(cudaDeviceGetPCIBusId(busId, static_cast<int>(sizeof(busId)), gpuId), return -1);

    for (char& character : busId) {
        if (character == '\0') {
            break;
        }
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }

    char path[256]{};
    std::snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", busId);
    std::FILE* file = std::fopen(path, "r");
    if (file == nullptr) {
        return -1;
    }

    int node = -1;
    const bool readOk = std::fscanf(file, "%d", &node) == 1;
    std::fclose(file);
    if (!readOk) {
        return -1;
    }
    // A sysfs value of -1 is normal on a single-node workstation.
    return node < 0 ? 0 : node;
}

GpuContext* findGpu(const std::vector<GpuContext*>& availableContexts, int gpuId) {
    for (GpuContext* context : availableContexts) {
        if (context != nullptr && context->gpuId == gpuId) {
            return context;
        }
    }
    return nullptr;
}

void releaseContexts(std::vector<GpuContext*>& availableContexts) {
    for (GpuContext* context : availableContexts) {
        if (context == nullptr) {
            continue;
        }
        if (context->activeTaskCount() != 0) {
            std::fprintf(stderr, "[GPUInfra] refusing to release gpu=%d with %zu registered task resource(s)\n", context->gpuId, context->activeTaskCount());
            continue;
        }
        if (context->primaryCtx != nullptr) {
            CU_CHECK(cuDevicePrimaryCtxRelease(context->device), );
            context->primaryCtx = nullptr;
        }
        delete context;
    }
    availableContexts.clear();
}

}  // namespace

bool GpuContextManager::init(const GpuInfraConfig& requestedConfig) {
    std::lock_guard<std::mutex> guard(lock);
    if (initialised.load(std::memory_order_acquire)) {
        return false;
    }

    CU_CHECK(cuInit(0), return false);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), return false);
    if (deviceCount <= 0) {
        return false;
    }

    config = requestedConfig;
    contexts.clear();

    for (int gpu = 0; gpu < deviceCount; ++gpu) {
        int node = probeNumaNodeOfGpu(gpu);
        if (node < 0 && !config.requireNuma) {
            node = 0;
        }
        if (node < 0) {
            releaseContexts(contexts);
            return false;
        }

        CUDA_CHECK(cudaSetDevice(gpu), {
            releaseContexts(contexts);
            return false;
        });
        CUDA_CHECK(cudaFree(nullptr), {
            releaseContexts(contexts);
            return false;
        });

        GpuContext* context = new (std::nothrow) GpuContext();
        if (context == nullptr) {
            releaseContexts(contexts);
            return false;
        }
        context->gpuId = gpu;
        context->numaNode = node;

        CU_CHECK(cuDeviceGet(&context->device, gpu), {
            delete context;
            releaseContexts(contexts);
            return false;
        });
        CU_CHECK(cuDevicePrimaryCtxRetain(&context->primaryCtx, context->device), {
            delete context;
            releaseContexts(contexts);
            return false;
        });

        contexts.push_back(context);

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, gpu), {
            releaseContexts(contexts);
            return false;
        });
        std::fprintf(stderr, "[GPUInfra] discovered gpu=%d name=%s numa=%d compute=%d.%d async_engines=%d\n", gpu, properties.name, node, properties.major, properties.minor, properties.asyncEngineCount);
    }

    initialised.store(true, std::memory_order_release);
    return true;
}

bool GpuContextManager::pinCurrentThreadToNumaNode(int numaNode) {
    return pinToNumaNode(numaNode);
}

bool GpuContextManager::registerTask(int numaNode, int gpuId, TaskGpuResources& resources) {
    if (!initialised.load(std::memory_order_acquire) || resources.ctx != nullptr || numaNode < 0 || gpuId < 0 || !pinToNumaNode(numaNode)) {
        return false;
    }

    std::lock_guard<std::mutex> guard(lock);
    GpuContext* context = findGpu(contexts, gpuId);
    if (context == nullptr || context->numaNode != numaNode) {
        return false;
    }

    std::size_t resourceId = context->taskResources.size();
    for (std::size_t index = 0; index < context->taskResources.size(); ++index) {
        if (context->taskResources[index] == nullptr) {
            resourceId = index;
            break;
        }
    }
    if (resourceId == context->taskResources.size()) {
        context->taskResources.push_back(&resources);
    }
    else {
        context->taskResources[resourceId] = &resources;
    }

    resources.resourceId = static_cast<int>(resourceId);
    resources.gpuId = gpuId;
    resources.numaNode = numaNode;
    resources.ctx = context;
    std::fprintf(stderr, "[GPUInfra] registered task gpu=%d numa=%d resource=%d\n", gpuId, numaNode, resources.resourceId);
    return true;
}

bool GpuContextManager::makeTaskCurrent(const TaskGpuResources& resources) {
    if (!initialised.load(std::memory_order_acquire) || resources.ctx == nullptr || resources.gpuId < 0 || resources.ctx->gpuId != resources.gpuId || resources.ctx->numaNode != resources.numaNode) {
        return false;
    }
    CUDA_CHECK(cudaSetDevice(resources.gpuId), return false);
    return true;
}

bool GpuContextManager::unregisterTask(TaskGpuResources& resources) {
    if (resources.ctx == nullptr || resources.resourceId < 0) {
        return true;
    }

    std::lock_guard<std::mutex> guard(lock);
    GpuContext* context = resources.ctx;
    const std::size_t resourceId = static_cast<std::size_t>(resources.resourceId);
    if (!initialised.load(std::memory_order_acquire) || resourceId >= context->taskResources.size() || context->taskResources[resourceId] != &resources) {
        return false;
    }

    context->taskResources[resourceId] = nullptr;
    std::fprintf(stderr, "[GPUInfra] unregistered task gpu=%d resource=%d\n", context->gpuId, resources.resourceId);
    resources.resourceId = -1;
    resources.gpuId = -1;
    resources.numaNode = -1;
    resources.ctx = nullptr;
    return true;
}

void GpuContextManager::shutdown() {
    std::lock_guard<std::mutex> guard(lock);
    if (!initialised.load(std::memory_order_acquire)) {
        return;
    }
    for (const GpuContext* context : contexts) {
        if (context != nullptr && context->activeTaskCount() != 0) {
            std::fprintf(stderr, "[GPUInfra] shutdown blocked by gpu=%d active_tasks=%zu\n", context->gpuId, context->activeTaskCount());
            return;
        }
    }

    releaseContexts(contexts);
    initialised.store(false, std::memory_order_release);
    std::fprintf(stderr, "[GPUInfra] shutdown complete\n");
}

std::vector<GpuLocation> GpuContextManager::gpuLocations() {
    std::lock_guard<std::mutex> guard(lock);
    std::vector<GpuLocation> locations;
    locations.reserve(contexts.size());
    for (const GpuContext* context : contexts) {
        if (context != nullptr) {
            locations.push_back({context->gpuId, context->numaNode});
        }
    }
    return locations;
}
