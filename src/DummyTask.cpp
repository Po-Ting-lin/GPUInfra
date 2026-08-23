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
#include "FrameSlot.h"
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

    if (runtime.frameW != expectedFrameSize){
        return false;
    }
    if (runtime.frameH != expectedFrameSize){
        return false;
    }
    if (runtime.inBytes != expectedInputBytes){
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
    if (!GpuContextManager::makeTaskCurrent(resources)){
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
    if (state != TaskLifecycle::Notified || !staticData.isInitialized()) {
        return false;
    }

    FrameSlot* selectedFrame = staticData.findFrameSlot(atom.metadata);
    if (selectedFrame == nullptr) {
        return false;
    }

    FrameSlot& frame = *selectedFrame;
    frame.result.id = frame.metadata.id;

    if (!atom.matchesMetadata(frame.metadata)) {
        frame.result.ok = false;
        return false;
    }
    if (frame.numaNode != node){ // check (node of this task) == (node of FrameSlot)
        frame.result.ok = false;
        return false;
    }
    if (atom.data.size() != frame.metadata.bytes || frame.metadata.bytes != resources.inBytes || frame.metadata.width != algoRuntime.frameW || frame.metadata.height != algoRuntime.frameH || frame.metadata.dtype != algoRuntime.frameDtype){ // input frame layout == task allocation layout
        frame.result.ok = false;
        return false;
    }
    if (!frame.deviceData.isInitialized() || frame.deviceData.bytes() != resources.inBytes){
        frame.result.ok = false;
        return false;
    }
    if (frame.result.outputs.size() != algorithms.size()){ // number of output of FrameSlot == number of algos
        frame.result.ok = false;
        return false;
    }

    frame.result.ok = GpuContextManager::makeTaskCurrent(resources);
    if (!frame.result.ok) {
        return false;
    }

    const bool needsUpload = !frame.deviceData.hasData(frame.metadata.id);
    const FrameGpuAccessMode accessMode = needsUpload ? FrameGpuAccessMode::Upload : FrameGpuAccessMode::Read;
    
    FrameGpuAccess access = frame.deviceData.acquire(accessMode, frame.metadata.id, resources);
    if (!access) {
        frame.result.ok = false;
        return false;
    }

    // pagable to pin
    // H2D 
    if (needsUpload) {
        std::memcpy(resources.h_in, atom.data.data(), atom.data.size());
        CUDA_CHECK(cudaMemcpyAsync(access.writableData(), resources.h_in, resources.inBytes, cudaMemcpyHostToDevice, resources.stream), frame.result.ok = false);
    }

    // Compute: all kernels -> all D2H
    if (executionModel == ExecutionModel::Batched) {
        if (frame.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchKernels(resources, access.data(), resources.stream)) {
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
    // Compute: kernel A -> H2D A -> kernel B -> H2D B ....  
    else if (executionModel == ExecutionModel::Interleaved) {
        if (frame.result.ok) {
            for (const std::unique_ptr<IAlgo>& algorithm : algorithms) {
                if (!algorithm->launchKernels(resources, access.data(), resources.stream) || !algorithm->launchD2H(resources, resources.stream)) {
                    frame.result.ok = false;
                    break;
                }
            }
        }
    }
    else {
        frame.result.ok = false;
    }

    if (!access.complete(frame.result.ok)) {
        frame.result.ok = false;
    }

    // pin to pagable
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
