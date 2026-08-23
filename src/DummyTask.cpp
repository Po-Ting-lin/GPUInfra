#include "DummyTask.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "Cel.h"
#include "CudaCheck.h"
#include "FrameCpuAtom.h"
#include "GpuContextManager.h"
#include "IAlgo.h"
#include "ImageSizing.h"
#include "Mi.h"
#include "Sdd.h"
#include "StaticData.h"

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

DummyTask::DummyTask(int instanceId, int numaNode, int gpuId, ExecutionModel model, const AlgoRuntimeInfo& runtime)
    : id(instanceId),
      node(numaNode),
      gpu(gpuId),
      executionModel(model),
      algoRuntime(runtime) {}

DummyTask::~DummyTask() {
    unload();
}

bool DummyTask::load() {
    if (state != TaskLifecycle::Constructed || id < 0 || node < 0 || gpu < 0 || !validRuntime(algoRuntime)) {
        return false;
    }

    // GpuContextManager register
    if (!GpuContextManager::registerTask(node, gpu, resources)) {
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
    if (!ok) {
        state = TaskLifecycle::Failed;
        releaseResources();
        return false;
    }

    state = TaskLifecycle::Loaded;
    return true;
}

bool DummyTask::registerParameters(ParameterRegistry& registry) {
    if (state != TaskLifecycle::Loaded || !registry.registerParameter(NAME_PARAMETER, ParameterType::String) || !registry.registerParameter(BLOB_PARAMETER, ParameterType::Bytes)) {
        return false;
    }
    state = TaskLifecycle::Registered;
    return true;
}

bool DummyTask::notifyParameters(const ParameterSnapshot& parameters) {
    if (state != TaskLifecycle::Registered) {
        return false;
    }

    AlgoParams values;
    if (!parameters.getString(NAME_PARAMETER, values.name) || !parameters.getBytes(BLOB_PARAMETER, values.blob)) {
        return false;
    }

    // notify each parameter
    for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
        if (!algorithm->notifyParameter(values)) {
            return false;
        }
    }

    state = TaskLifecycle::Notified;
    return true;
}

bool DummyTask::execute(FrameCpuAtom& atom, StaticData& staticData) {
    atom.result.id = atom.metadata.id;
    if (state != TaskLifecycle::Notified || !staticData.isInitialized()) {
        atom.result.ok = false;
        return false;
    }

    // The atom must be registered by the same NUMA-local StaticData instance.
    if (!staticData.validateFrame(atom.metadata, node)) {
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

    FrameGpuAccess access = staticData.acquireFrameGpuAccess(atom.metadata, resources);
    if (!access) {
        atom.result.ok = false;
        return false;
    }

    // Pageable to pinned staging.
    // H2D
    if (access.needsUpload()) {
        std::memcpy(resources.h_in, atom.data.data(), atom.data.size());
        CUDA_CHECK(cudaMemcpyAsync(access.writableData(), resources.h_in, resources.inBytes, cudaMemcpyHostToDevice, resources.stream), atom.result.ok = false);
    }

    // Compute: all kernels -> all D2H
    if (executionModel == ExecutionModel::Batched) {
        if (atom.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchKernels(resources, access.data(), resources.stream)) {
                    atom.result.ok = false;
                    break;
                }
            }
        }
        if (atom.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchD2H(resources, resources.stream)) {
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
                if (!algorithm->launchKernels(resources, access.data(), resources.stream) || !algorithm->launchD2H(resources, resources.stream)) {
                    atom.result.ok = false;
                    break;
                }
            }
        }
    }
    else {
        atom.result.ok = false;
    }

    if (!access.complete(atom.result.ok)) {
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

bool DummyTask::unload() {
    if (state == TaskLifecycle::Unloaded) {
        return true;
    }

    const bool ok = releaseResources();
    state = TaskLifecycle::Unloaded;
    return ok;
}

int DummyTask::instanceId() const {
    return id;
}

int DummyTask::gpuId() const {
    return gpu;
}

int DummyTask::numaNode() const {
    return node;
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
