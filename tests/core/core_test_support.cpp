#include "core/core_test_support.h"

namespace gpuinfra_tests {

AlgoRuntimeInfo makeRuntime(int factor) {
    AlgoRuntimeInfo runtime;
    runtime.sizeFactor = factor;
    runtime.frameW = factor * 8;
    runtime.frameH = runtime.frameW;
    runtime.inBytes = static_cast<std::size_t>(runtime.frameW) * static_cast<std::size_t>(runtime.frameH);
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

}  // namespace gpuinfra_tests
