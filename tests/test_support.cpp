#include "test_support.h"

namespace gpuinfra_tests {

AlgoRuntimeInfo makeRuntime(int factor) {
    AlgoRuntimeInfo runtime;
    runtime.sizeFactor = factor;
    runtime.frameW = ImageSizing::scaledDimension(factor, ImageSizing::INPUT_MULTIPLIER);
    runtime.frameH = runtime.frameW;
    runtime.inBytes = ImageSizing::squareBytes(runtime.frameW, sizeof(std::uint8_t));
    runtime.frameDtype = 1;
    return runtime;
}

FrameMetadata makeFrameMetadata(std::uint64_t frameId, const AlgoRuntimeInfo& runtime, std::uint32_t cameraId) {
    FrameMetadata metadata;
    metadata.key.frameId = frameId;
    metadata.key.cameraId = cameraId;
    metadata.bytes = runtime.inBytes;
    metadata.width = runtime.frameW;
    metadata.height = runtime.frameH;
    metadata.dtype = runtime.frameDtype;
    return metadata;
}

bool loadTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.load(); });
}

// Test bodies run inside NumaExecutor, including registration and error cases.

bool configureTaskParameters(DummyTask& task, ParameterRegistry& registry) {
    return task.registerParameters(registry) && registry.setString(DummyTask::NAME_PARAMETER, "test") && registry.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && registry.seal();
}

bool startTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.start(); });
}

bool stopTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.stop(); });
}

bool notifyTask(DummyTask& task, const ParameterSnapshot& parameters, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task, &parameters] { return task.notifyParameters(parameters); });
}

bool unloadTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.unload(); });
}

bool initializeStaticData(StaticData& staticData, const GpuLocation& location, const AlgoRuntimeInfo& runtime, std::size_t gpuCacheEntries) {
    return NumaExecutor(location.numaNode).run([&staticData, &runtime, gpuCacheEntries] {
        StaticDataConfig config;
        config.runtime = runtime;
        config.gpuCacheEntries = gpuCacheEntries;
        return staticData.init(config);
    });
}

bool releaseStaticData(StaticData& staticData, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&staticData] { return staticData.release(); });
}

GpuCacheRequest makeCacheRequest(const TaskGpuResources& resources) {
    GpuCacheRequest request;
    request.gpuId = resources.gpuId;
    request.stream = resources.stream;
    request.d_fallback = resources.d_input;
    request.fallbackBytes = resources.inBytes;
    return request;
}

bool initializeAccessResources(TaskGpuResources& resources, const GpuLocation& location, std::size_t bytes) {
    resources.gpuId = location.gpuId;
    resources.inBytes = bytes;
    if (cudaSetDevice(location.gpuId) != cudaSuccess || cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&resources.d_input, bytes) != cudaSuccess) {
        cudaStreamDestroy(resources.stream);
        resources.stream = nullptr;
        return false;
    }
    return true;
}

bool releaseAccessResources(TaskGpuResources& resources) {
    bool ok = true;
    if (resources.stream != nullptr && cudaStreamSynchronize(resources.stream) != cudaSuccess) {
        ok = false;
    }
    if (resources.d_input != nullptr && cudaFree(resources.d_input) != cudaSuccess) {
        ok = false;
    }
    resources.d_input = nullptr;
    if (resources.stream != nullptr && cudaStreamDestroy(resources.stream) != cudaSuccess) {
        ok = false;
    }
    resources.stream = nullptr;
    resources.inBytes = 0;
    return ok;
}

bool executeTask(DummyTask& task, FrameCpuAtom& atom, StaticData& staticData, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task, &atom, &staticData] { return staticData.execute() && task.execute(atom, staticData); });
}

std::uint32_t referenceValue(const FrameCpuAtom& atom, int dimension, int x, int y, int inputStride) {
    std::uint32_t sum = 0;
    for (int k = 0; k < dimension; ++k) {
        const std::uint32_t a = atom.data[static_cast<std::size_t>(y) * inputStride + k];
        const std::uint32_t b = atom.data[static_cast<std::size_t>(k) * inputStride + x];
        sum += a * b;
    }
    return sum;
}

bool verifyOutput(const FrameCpuAtom& atom, const AlgoOutput& output, int inputStride) {
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            std::uint32_t actual = 0;
            const std::size_t offset = (static_cast<std::size_t>(y) * output.width + x) * sizeof(actual);
            std::memcpy(&actual, output.data.data() + offset, sizeof(actual));
            if (actual != referenceValue(atom, output.width, x, y, inputStride)) {
                return false;
            }
        }
    }
    return true;
}

bool runGraphPhase(DummyGraph& graph, FramePhase phase) {
    PhaseGate gate;
    if (!graph.startPhase(phase, gate)) {
        gate.release();
        return false;
    }
    gate.release();
    return graph.waitForPhase();
}

GraphConfig makeGraphConfig(std::size_t tasks, std::size_t workers, ExecutionModel model) {
    GraphConfig config;
    config.taskInstancesPerGpu = tasks;
    config.graphThreads = workers;
    config.warmupFramesPerGpu = 0;
    config.timedFramesPerGpu = 8;
    config.executionModel = model;
    config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    config.parameters.name = "graph-test";
    return config;
}

}  // namespace gpuinfra_tests
