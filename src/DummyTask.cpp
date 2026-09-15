#include "DummyTask.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "Algo/Cel.h"
#include "Algo/IAlgo.h"
#include "Algo/Mi.h"
#include "Algo/Sdd.h"
#include "CudaCheck.h"
#include "FrameCpuAtom.h"
#include "GpuContextManager.h"
#include "ImageSizing.h"
#include "StaticData.h"
#include "WorkloadSizing.h"

namespace {

bool validRuntime(const AlgoRuntimeInfo& runtime) {
    if (!ImageSizing::isValidFactor(runtime.sizeFactor)) {
        return false;
    }
    const int expectedFrameSize = ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::INPUT_MULTIPLIER);
    const std::size_t expectedInputBytes = ImageSizing::squareBytes(expectedFrameSize, sizeof(std::uint8_t));

    if (runtime.frameW != expectedFrameSize) {
        return false;
    }
    if (runtime.frameH != expectedFrameSize) {
        return false;
    }
    if (runtime.inBytes != expectedInputBytes) {
        return false;
    }
    return true;
}

}  // namespace

DummyTask::DummyTask(int instanceId, ExecutionModel model, const AlgoRuntimeInfo& runtime)
    : id(instanceId),
      executionModel(model),
      algoRuntime(runtime) {}

DummyTask::~DummyTask() {
    if (state == TaskLifecycle::Started) {
        stop();
    }
    unload();
}

bool DummyTask::registerParameters(ParameterRegistry& registry) {
    if (state != TaskLifecycle::Constructed || !registry.registerParameter(NAME_PARAMETER, ParameterType::String) || !registry.registerParameter(BLOB_PARAMETER, ParameterType::Bytes)) {
        return false;
    }
    parameterRegistry = &registry;
    state = TaskLifecycle::Registered;
    return true;
}

bool DummyTask::load() {
    ParameterSnapshot initialParameters;
    if (state != TaskLifecycle::Registered || parameterRegistry == nullptr || !parameterRegistry->snapshot(initialParameters) || id < 0 || !validRuntime(algoRuntime)) {
        return false;
    }

    // The framework establishes graph-local NUMA affinity before load().
    std::vector<int> gpuIds;
    if (!GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds) || !GpuContextManager::registerTask(gpuIds.front(), resources)) {
        state = TaskLifecycle::Failed;
        releaseResources();
        return false;
    }
    // Check TaskGpuResources ready and cudaSetDevice
    if (!GpuContextManager::makeTaskCurrent(resources)) {
        state = TaskLifecycle::Failed;
        releaseResources();
        return false;
    }

    resources.inBytes = algoRuntime.inBytes;

    // Create CUDA Stream
    bool ok = true;
    CUDA_CHECK(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), ok = false);
    if (ok) {
        CUDA_CHECK(cudaHostAlloc(&resources.h_in, resources.inBytes, cudaHostAllocPortable), ok = false);
    }
    if (ok) {
        std::memset(resources.h_in, 0, resources.inBytes);
    }
    if (ok) {
        CUDA_CHECK(cudaMalloc(&resources.d_input, resources.inBytes), ok = false);
    }
    if (ok) {
        CUDA_CHECK(cudaMemsetAsync(resources.d_input, 0, resources.inBytes, resources.stream), ok = false);
    }

    // Prepare Algo
    if (ok) {
        algorithms.reserve(3);
        algorithms.push_back(std::make_unique<Cel>());
        algorithms.push_back(std::make_unique<Sdd>());
        algorithms.push_back(std::make_unique<Mi>());
    }

    // Get the max common buffer size
    std::size_t scratchBytes = 0;
    if (ok) {
        for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
            std::size_t algorithmScratchBytes = 0;
            if (!algorithm->init(algoRuntime, resources, algorithmScratchBytes)) {
                ok = false;
                break;
            }
            scratchBytes = std::max(scratchBytes, algorithmScratchBytes);
        }
    }

    // Allocate the max common buffer size
    if (ok && scratchBytes > 0) {
        CUDA_CHECK(cudaMalloc(&resources.d_scratch, scratchBytes), ok = false);
        if (ok) {
            CUDA_CHECK(cudaMemsetAsync(resources.d_scratch, 0, scratchBytes, resources.stream), ok = false);
            resources.scratchBytes = scratchBytes;
        }
    }
    if (ok) {
        CUDA_CHECK(cudaStreamSynchronize(resources.stream), ok = false);
    }
    if (ok && !applyParameters(initialParameters)) {
        ok = false;
    }
    if (!ok) {
        state = TaskLifecycle::Failed;
        releaseResources();
        return false;
    }

    state = TaskLifecycle::Loaded;
    return true;
}

bool DummyTask::notifyParameters(const ParameterSnapshot& parameters) {
    if (state != TaskLifecycle::Loaded && state != TaskLifecycle::Started && state != TaskLifecycle::Stopped) {
        return false;
    }
    if (parameters.revision() < appliedParameterRevision) {
        return false;
    }
    if (parameters.revision() == appliedParameterRevision) {
        return true;
    }
    if (!GpuContextManager::makeTaskCurrent(resources) || !applyParameters(parameters)) {
        state = TaskLifecycle::Failed;
        return false;
    }

    return true;
}

bool DummyTask::start() {
    if (state != TaskLifecycle::Loaded) {
        return false;
    }

    // The simulation models only the lifecycle boundary. CUDA resources keep
    // their existing load-to-unload lifetime.
    state = TaskLifecycle::Started;
    return true;
}

bool DummyTask::applyParameters(const ParameterSnapshot& parameters) {
    AlgoParams values;
    if (!parameters.getString(NAME_PARAMETER, values.name) || !parameters.getBytes(BLOB_PARAMETER, values.blob)) {
        return false;
    }

    // Apply one complete parameter revision to every private algorithm.
    for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
        if (!algorithm->notifyParameter(values)) {
            return false;
        }
    }

    appliedParameterRevision = parameters.revision();
    return true;
}

bool DummyTask::execute(FrameCpuAtom& atom, StaticData& staticData) { // TODO: not sure where is the correct place for static data
    atom.result.id = atom.metadata.key.frameId;
    if (state != TaskLifecycle::Started) {
        atom.result.ok = false;
        return false;
    }

    // The input layout must match the task's cold-path allocation layout.
    if (atom.data.size() != atom.metadata.bytes || atom.metadata.bytes != resources.inBytes || atom.metadata.width != algoRuntime.frameW || atom.metadata.height != algoRuntime.frameH || atom.metadata.dtype != algoRuntime.frameDtype) {
        atom.result.ok = false;
        return false;
    }
    if (resources.d_input == nullptr || atom.result.outputs.size() != algorithms.size()) {
        atom.result.ok = false;
        return false;
    }

    atom.result.ok = GpuContextManager::makeTaskCurrent(resources);
    if (!atom.result.ok) {
        return false;
    }

    GpuCacheRequest request;
    request.gpuId = resources.gpuId;
    request.stream = resources.stream;
    request.d_fallback = resources.d_input;
    request.fallbackBytes = resources.inBytes;
    GpuDataAccess access = staticData.getCacheData(atom.metadata, request);
    const CacheStatus status = access.status();
    if (status == CacheStatus::Invalid) {
        atom.result.ok = false;
        return false;
    }

    const cudaStream_t stream = access.getStream();

    // Caller handles a frame miss: pageable to pinned staging, then H2D.
    if (status == CacheStatus::CacheFill || status == CacheStatus::TaskFallback) {
        std::memcpy(resources.h_in, atom.data.data(), atom.data.size());
        for (int repeat = 0; repeat < GPUINFRA_H2D_SIZE_MULTIPLIER && atom.result.ok; ++repeat) {
            CUDA_CHECK(cudaMemcpyAsync(access.writableData(), resources.h_in, resources.inBytes, cudaMemcpyHostToDevice, stream), atom.result.ok = false);
        }
    }

    // Compute: all kernels -> all D2H
    if (executionModel == ExecutionModel::Batched) {
        if (atom.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchKernels(resources, access.data(), stream)) {
                    atom.result.ok = false;
                    break;
                }
            }
        }
        if (atom.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchD2H(resources, stream)) {
                    atom.result.ok = false;
                    break;
                }
            }
        }
    }
    // Compute: kernel A -> D2H A -> kernel B -> D2H B ...
    else if (executionModel == ExecutionModel::Interleaved) {
        if (atom.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchKernels(resources, access.data(), stream) || !algorithm->launchD2H(resources, stream)) {
                    atom.result.ok = false;
                    break;
                }
            }
        }
    }
    else {
        atom.result.ok = false;
    }

    if (!access.freeCacheData(atom.result.ok)) {  // Tells everyone the data is free now. Only 1 sync here
        atom.result.ok = false;
    }

    // Pinned staging to pageable results.
    if (atom.result.ok) {
        for (std::size_t index = 0; index < algorithms.size(); ++index) {
            if (!algorithms[index]->collectResult(resources, atom.result.outputs[index])) {
                atom.result.ok = false;
                break;
            }
        }
    }
    return atom.result.ok;
}

bool DummyTask::stop() {
    if (state != TaskLifecycle::Started) {
        return false;
    }

    // DummyGraph guarantees that no execute call is active at this boundary.
    state = TaskLifecycle::Stopped;
    return true;
}

bool DummyTask::unload() {
    if (state == TaskLifecycle::Unloaded) {
        return true;
    }
    if (state == TaskLifecycle::Started) {
        return false;
    }

    const bool ok = releaseResources();
    parameterRegistry = nullptr;
    appliedParameterRevision = 0;
    state = TaskLifecycle::Unloaded;
    return ok;
}

int DummyTask::instanceId() const {
    return id;
}

int DummyTask::gpuId() const {
    return resources.gpuId;
}

TaskLifecycle DummyTask::lifecycle() const {
    return state;
}

bool DummyTask::releaseResources() {
    bool ok = true;
    if (resources.ctx != nullptr && !GpuContextManager::makeTaskCurrent(resources)) {
        ok = false;
    }
    if (resources.stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(resources.stream), ok = false);
    }
    for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
        if (algorithm != nullptr && !algorithm->close()) {
            ok = false;
        }
    }
    algorithms.clear();

    if (resources.d_scratch != nullptr) {
        CUDA_CHECK(cudaFree(resources.d_scratch), ok = false);
        resources.d_scratch = nullptr;
    }
    resources.scratchBytes = 0;
    if (resources.d_input != nullptr) {
        CUDA_CHECK(cudaFree(resources.d_input), ok = false);
        resources.d_input = nullptr;
    }
    if (resources.h_in != nullptr) {
        CUDA_CHECK(cudaFreeHost(resources.h_in), ok = false);
        resources.h_in = nullptr;
    }
    if (resources.stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(resources.stream), ok = false);
        resources.stream = nullptr;
    }
    resources.inBytes = 0;
    if (!GpuContextManager::unregisterTask(resources)) {
        ok = false;
    }
    return ok;
}
