#include "GpuContextManager.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include "CudaCheck.h"
#include "GpuContext.h"

GpuInfraConfig GpuContextManager::config;
std::vector<GpuContext*> GpuContextManager::contexts;
std::unordered_map<int, std::vector<GpuContext*>> GpuContextManager::numaToCtxs;
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

    // Respect a container/cgroup's existing CPU allowance.
    CPU_AND(&nodeSet, &nodeSet, &allowedSet);
    if (CPU_COUNT(&nodeSet) == 0) {
        return false;
    }
    // Pin before cudaHostAlloc and host first-touch. Avoid process-wide
    // membind because GPUInfra does not own unrelated application memory.
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

int detectNumaAffinity() {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity) != 0) {
        return -1;
    }

    int detectedNode = -1;
    for (int node = 0; node < 256; ++node) {
        cpu_set_t nodeSet;
        if (!nodeCpuSet(node, nodeSet)) {
            continue;
        }
        CPU_AND(&nodeSet, &nodeSet, &affinity);
        if (CPU_COUNT(&nodeSet) > 0) {
            if (detectedNode < 0) {
                detectedNode = node;
            }
            else if (detectedNode != node) {
                return -1;
            }
        }
    }
    return detectedNode;
}

GpuContext* findGpu(const std::vector<GpuContext*>& contexts, int gpuId) {
    for (GpuContext* context : contexts) {
        if (context != nullptr && context->gpuId == gpuId) {
            return context;
        }
    }
    return nullptr;
}

GpuContext* pickContextOnNode(const std::unordered_map<int, std::vector<GpuContext*>>& contextsByNode, int node, int gpuHint) {
    const auto found = contextsByNode.find(node);
    if (found == contextsByNode.end() || found->second.empty()) {
        return nullptr;
    }

    if (gpuHint >= 0) {
        for (GpuContext* context : found->second) {
            if (context->gpuId == gpuHint) {
                return context;
            }
        }
        return nullptr;
    }

    GpuContext* best = nullptr;
    for (GpuContext* context : found->second) {
        if (context->activeSlotCount() >= static_cast<std::size_t>(context->maxThreadsPerGpu)) {
            continue;
        }
        if (best == nullptr || context->activeSlotCount() < best->activeSlotCount()) {
            best = context;
        }
    }
    return best;
}

bool activateContext(GpuContext* context, bool primeRuntime) {
    if (context == nullptr || context->primaryCtx == nullptr) {
        return false;
    }
    CUDA_CHECK(cudaSetDevice(context->gpuId), return false);
    if (primeRuntime) {
        CUDA_CHECK(cudaFree(nullptr), return false);
    }
    CU_CHECK(cuCtxPushCurrent(context->primaryCtx), return false);
    return true;
}

bool popContext() {
    CUcontext popped = nullptr;
    CU_CHECK(cuCtxPopCurrent(&popped), return false);
    return true;
}

bool firstTouchDevice(void* d_pointer, std::size_t bytes, cudaStream_t stream) {
    if (d_pointer == nullptr || bytes == 0 || stream == nullptr) {
        return false;
    }
    CUDA_CHECK(cudaMemsetAsync(d_pointer, 0, bytes, stream), return false);
    CUDA_CHECK(cudaStreamSynchronize(stream), return false);
    return true;
}

void destroyThreadScratch(ThreadSlot* slot) {
    if (slot == nullptr) {
        return;
    }
    if (slot->d_scratch != nullptr) {
        CUDA_CHECK(cudaFree(slot->d_scratch), );
        slot->d_scratch = nullptr;
    }
    slot->scratchBytes = 0;
}

void destroySlot(ThreadSlot* slot) {
    if (slot == nullptr) {
        return;
    }
    if (slot->stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(slot->stream), );
    }
    destroyThreadScratch(slot);
    if (slot->d_in != nullptr) {
        CUDA_CHECK(cudaFree(slot->d_in), );
        slot->d_in = nullptr;
    }
    if (slot->h_in != nullptr) {
        CUDA_CHECK(cudaFreeHost(slot->h_in), );
        slot->h_in = nullptr;
    }
    if (slot->stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(slot->stream), );
        slot->stream = nullptr;
    }
    delete slot;
}

void releaseContexts(std::vector<GpuContext*>& contexts) {
    for (GpuContext* context : contexts) {
        if (context == nullptr) {
            continue;
        }

        const bool active = activateContext(context, true);
        if (active) {
            for (ThreadSlot*& slot : context->threadSlots) {
                destroySlot(slot);
                slot = nullptr;
            }
            popContext();
        }

        if (context->primaryCtx != nullptr) {
            CU_CHECK(cuDevicePrimaryCtxRelease(context->device), );
            context->primaryCtx = nullptr;
        }
        delete context;
    }
    contexts.clear();
}

}  // namespace

bool GpuContextManager::init(const GpuInfraConfig& requestedConfig) {
    std::lock_guard<std::mutex> guard(lock);
    if (initialised.load(std::memory_order_acquire) || requestedConfig.threadsPerGpu <= 0 || requestedConfig.inputBytes == 0) {
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
    numaToCtxs.clear();

    for (int gpu = 0; gpu < deviceCount; ++gpu) {
        const int node = probeNumaNodeOfGpu(gpu);
        if (node < 0 || !pinToNumaNode(node)) {
            releaseContexts(contexts);
            numaToCtxs.clear();
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
        context->maxThreadsPerGpu = config.threadsPerGpu;
        context->inputBytes = config.inputBytes;
        context->threadSlots.assign(static_cast<std::size_t>(config.threadsPerGpu), nullptr);

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
        numaToCtxs[node].push_back(context);

        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, gpu), {
            releaseContexts(contexts);
            numaToCtxs.clear();
            return false;
        });
        std::fprintf(stderr, "[GPUInfra] discovered gpu=%d name=%s numa=%d compute=%d.%d async_engines=%d\n", gpu, properties.name, node, properties.major, properties.minor, properties.asyncEngineCount);
    }

    initialised.store(true, std::memory_order_release);
    return true;
}

ThreadSlot* GpuContextManager::registerThread(int numaHint, int gpuHint) {
    if (!initialised.load(std::memory_order_acquire)) {
        return nullptr;
    }

    int targetNode = numaHint;
    if (targetNode < 0) {
        targetNode = detectNumaAffinity();
    }
    else if (!pinToNumaNode(targetNode)) {
        return nullptr;
    }

    std::lock_guard<std::mutex> guard(lock);
    GpuContext* context = nullptr;
    if (gpuHint >= 0) {
        context = findGpu(contexts, gpuHint);
        if (context != nullptr && targetNode < 0) {
            targetNode = context->numaNode;
        }
        if (context == nullptr || context->numaNode != targetNode) {
            return nullptr;
        }
    }
    else if (targetNode >= 0) {
        context = pickContextOnNode(numaToCtxs, targetNode, -1);
    }

    if (context == nullptr && !config.requireNuma) {
        for (GpuContext* candidate : contexts) {
            if (candidate->activeSlotCount() < static_cast<std::size_t>(candidate->maxThreadsPerGpu) && (context == nullptr || candidate->activeSlotCount() < context->activeSlotCount())) {
                context = candidate;
            }
        }
    }
    if (context == nullptr || context->activeSlotCount() >= static_cast<std::size_t>(context->maxThreadsPerGpu) || !pinToNumaNode(context->numaNode) || !activateContext(context, true)) {
        return nullptr;
    }

    int slotId = -1;
    for (std::size_t index = 0; index < context->threadSlots.size(); ++index) {
        if (context->threadSlots[index] == nullptr) {
            slotId = static_cast<int>(index);
            break;
        }
    }
    if (slotId < 0) {
        popContext();
        return nullptr;
    }

    ThreadSlot* slot = new (std::nothrow) ThreadSlot();
    if (slot == nullptr) {
        popContext();
        return nullptr;
    }
    slot->threadId = slotId;
    slot->ownerTid = std::this_thread::get_id();
    slot->gpuId = context->gpuId;
    slot->numaNode = context->numaNode;
    slot->ctx = context;
    slot->inBytes = context->inputBytes;

    bool ok = true;
    CUDA_CHECK(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking), ok = false);
    if (ok) {
        CUDA_CHECK(cudaHostAlloc(&slot->h_in, slot->inBytes, cudaHostAllocPortable), ok = false);
    }
    if (ok) {
        std::memset(slot->h_in, 0, slot->inBytes);
        CUDA_CHECK(cudaMalloc(&slot->d_in, slot->inBytes), ok = false);
    }
    if (ok) {
        ok = firstTouchDevice(slot->d_in, slot->inBytes, slot->stream);
    }

    if (!ok) {
        destroySlot(slot);
        popContext();
        return nullptr;
    }

    context->threadSlots[static_cast<std::size_t>(slotId)] = slot;
    if (!popContext()) {
        context->threadSlots[static_cast<std::size_t>(slotId)] = nullptr;
        destroySlot(slot);
        return nullptr;
    }

    std::fprintf(stderr, "[GPUInfra] registered thread gpu=%d numa=%d slot=%d stream=%p\n", slot->gpuId, slot->numaNode, slot->threadId, static_cast<void*>(slot->stream));
    return slot;
}

bool GpuContextManager::prepareThreadScratch(ThreadSlot* slot, std::size_t scratchBytes) {
    if (slot == nullptr || slot->ctx == nullptr || slot->ownerTid != std::this_thread::get_id() || scratchBytes == 0) {
        return false;
    }

    std::lock_guard<std::mutex> guard(lock);
    GpuContext* context = slot->ctx;
    if (!initialised.load(std::memory_order_acquire) || slot->threadId < 0 || static_cast<std::size_t>(slot->threadId) >= context->threadSlots.size() || context->threadSlots[static_cast<std::size_t>(slot->threadId)] != slot || slot->d_scratch != nullptr || slot->scratchBytes != 0) {
        return false;
    }

    if (!activateContext(context, false)) {
        return false;
    }

    bool ok = true;
    CUDA_CHECK(cudaMalloc(&slot->d_scratch, scratchBytes), ok = false);
    if (ok) {
        ok = firstTouchDevice(slot->d_scratch, scratchBytes, slot->stream);
    }

    if (ok) {
        slot->scratchBytes = scratchBytes;
    }
    else {
        destroyThreadScratch(slot);
    }

    const bool popped = popContext();
    if (!ok || !popped) {
        return false;
    }

    std::fprintf(stderr, "[GPUInfra] prepared scratch gpu=%d slot=%d bytes=%zu\n", slot->gpuId, slot->threadId, scratchBytes);
    return true;
}

void GpuContextManager::unregisterThread(ThreadSlot* slot) {
    if (slot == nullptr || slot->ctx == nullptr || slot->ownerTid != std::this_thread::get_id()) {
        return;
    }

    std::lock_guard<std::mutex> guard(lock);
    GpuContext* context = slot->ctx;
    if (slot->threadId < 0 || static_cast<std::size_t>(slot->threadId) >= context->threadSlots.size() || context->threadSlots[static_cast<std::size_t>(slot->threadId)] != slot || !activateContext(context, false)) {
        return;
    }

    const int releasedSlot = slot->threadId;
    context->threadSlots[static_cast<std::size_t>(releasedSlot)] = nullptr;
    destroySlot(slot);
    popContext();
    std::fprintf(stderr, "[GPUInfra] unregistered thread gpu=%d slot=%d\n", context->gpuId, releasedSlot);
}

void GpuContextManager::shutdown() {
    std::lock_guard<std::mutex> guard(lock);
    if (!initialised.load(std::memory_order_acquire)) {
        return;
    }

    releaseContexts(contexts);
    numaToCtxs.clear();
    initialised.store(false, std::memory_order_release);
    std::fprintf(stderr, "[GPUInfra] shutdown complete\n");
}

std::size_t GpuContextManager::gpuCount() {
    std::lock_guard<std::mutex> guard(lock);
    return contexts.size();
}

int GpuContextManager::numaNodeForGpu(int gpuId) {
    std::lock_guard<std::mutex> guard(lock);
    GpuContext* context = findGpu(contexts, gpuId);
    return context == nullptr ? -1 : context->numaNode;
}
