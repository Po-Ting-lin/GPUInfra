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
#include "GpuContextManager.h"
#include "IAlgo.h"
#include "ImageSizing.h"
#include "Mi.h"
#include "Sdd.h"

namespace {

class ExecutionGuard {
public:
    explicit ExecutionGuard(std::atomic<bool>& target)
        : executing(target) {}

    ~ExecutionGuard() {
        executing.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool>& executing;
};

bool validRuntime(const AlgoRuntimeInfo& runtime) {
    if (!ImageSizing::isValidFactor(runtime.sizeFactor)) {
        return false;
    }
    const int expectedFrameSize = ImageSizing::scaledDimension(runtime.sizeFactor, ImageSizing::INPUT_MULTIPLIER);
    const std::size_t expectedInputBytes = ImageSizing::squareBytes(expectedFrameSize, sizeof(std::uint8_t));
    return runtime.frameW == expectedFrameSize && runtime.frameH == expectedFrameSize && runtime.inBytes == expectedInputBytes;
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
    if (!GpuContextManager::registerTask(node, gpu, resources) || !GpuContextManager::makeTaskCurrent(resources)) {
        state = TaskLifecycle::Failed;
        releaseResources();
        return false;
    }

    resources.inBytes = algoRuntime.inBytes;
    bool ok = true;
    CUDA_CHECK(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), ok = false);
    if (ok) {
        CUDA_CHECK(cudaHostAlloc(&resources.h_in, resources.inBytes, cudaHostAllocPortable), ok = false);
    }
    if (ok) {
        std::memset(resources.h_in, 0, resources.inBytes);
        CUDA_CHECK(cudaMalloc(&resources.d_in, resources.inBytes), ok = false);
    }
    if (ok) {
        CUDA_CHECK(cudaMemsetAsync(resources.d_in, 0, resources.inBytes, resources.stream), ok = false);
    }

    if (ok) {
        algorithms.reserve(3);
        algorithms.push_back(std::make_unique<Cel>());
        algorithms.push_back(std::make_unique<Sdd>());
        algorithms.push_back(std::make_unique<Mi>());
    }

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
    for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
        if (!algorithm->notifyParameter(values)) {
            return false;
        }
    }

    state = TaskLifecycle::Notified;
    return true;
}

bool DummyTask::execute(FrameSlot& frame) {
    frame.result.id = frame.id;
    if (state != TaskLifecycle::Notified || frame.numaNode != node || frame.input.size() != resources.inBytes || frame.result.outputs.size() != algorithms.size()) {
        frame.result.ok = false;
        return false;
    }

    bool expected = false;
    if (!executing.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        frame.result.ok = false;
        return false;
    }
    ExecutionGuard executionGuard(executing);

    frame.result.ok = GpuContextManager::makeTaskCurrent(resources);
    if (!frame.result.ok) {
        return false;
    }

    std::memcpy(resources.h_in, frame.input.data(), frame.input.size());
    CUDA_CHECK(cudaMemcpyAsync(resources.d_in, resources.h_in, resources.inBytes, cudaMemcpyHostToDevice, resources.stream), frame.result.ok = false);

    if (executionModel == ExecutionModel::Batched) {
        if (frame.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchKernels(resources, resources.stream)) {
                    frame.result.ok = false;
                    break;
                }
            }
        }
        if (frame.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchD2H(resources, resources.stream)) {
                    frame.result.ok = false;
                    break;
                }
            }
        }
    }
    else if (executionModel == ExecutionModel::Interleaved) {
        if (frame.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchKernels(resources, resources.stream) || !algorithm->launchD2H(resources, resources.stream)) {
                    frame.result.ok = false;
                    break;
                }
            }
        }
    }
    else {
        frame.result.ok = false;
    }

    CUDA_CHECK(cudaStreamSynchronize(resources.stream), frame.result.ok = false);
    if (frame.result.ok) {
        for (std::size_t index = 0; index < algorithms.size(); ++index) {
            if (!algorithms[index]->collectResult(resources, frame.result.outputs[index])) {
                frame.result.ok = false;
                break;
            }
        }
    }
    return frame.result.ok;
}

bool DummyTask::unload() {
    if (state == TaskLifecycle::Unloaded) {
        return true;
    }
    if (executing.load(std::memory_order_acquire)) {
        return false;
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
    if (resources.d_in != nullptr) {
        CUDA_CHECK(cudaFree(resources.d_in), ok = false);
        resources.d_in = nullptr;
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
