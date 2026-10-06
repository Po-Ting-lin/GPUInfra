#include "ResultPipeline/ResultTasks.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <thread>

#include "CudaCheck.h"
#include "Nvtx.h"

namespace result_pipeline {

ResultAtom makeAtom(std::uint64_t id, std::size_t bytes) {
    ResultAtom atom;
    atom.metadata.key.frameId = id;
    atom.metadata.bytes = bytes;
    atom.metadata.width = static_cast<int>(bytes);
    atom.metadata.height = 1;
    atom.metadata.dtype = 1;
    atom.frame.resize(bytes);
    atom.result.resize(bytes);
    for (std::size_t index = 0; index < bytes; ++index) {
        atom.frame[index] = static_cast<unsigned char>((id + index) & 0xffU);
    }
    return atom;
}

ResultTaskResources::~ResultTaskResources() {
    unload();
}

bool ResultTaskResources::load(std::size_t bytes) {
    if (gpu.ctx != nullptr || bytes == 0 || bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) return false;
    std::vector<int> gpuIds;
    if (!GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds) || !GpuContextManager::registerTask(gpuIds.front(), gpu)) return false;
    if (!GpuContextManager::makeTaskCurrent(gpu)) return false;
    gpu.inBytes = bytes;
    gpu.scratchBytes = bytes;
    CUDA_CHECK(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking), return false);
    CUDA_CHECK(cudaHostAlloc(&gpu.h_in, bytes, cudaHostAllocPortable), return false);
    CUDA_CHECK(cudaMalloc(&gpu.d_input, bytes), return false);
    CUDA_CHECK(cudaMalloc(&gpu.d_scratch, bytes), return false);
    return true;
}

bool ResultTaskResources::unload() {
    if (gpu.ctx == nullptr) return true;
    if (!GpuContextManager::makeTaskCurrent(gpu)) return false;
    bool ok = true;
    if (gpu.stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(gpu.stream), ok = false);
    }
    if (gpu.d_scratch != nullptr) {
        CUDA_CHECK(cudaFree(gpu.d_scratch), ok = false);
        gpu.d_scratch = nullptr;
    }
    if (gpu.d_input != nullptr) {
        CUDA_CHECK(cudaFree(gpu.d_input), ok = false);
        gpu.d_input = nullptr;
    }
    if (gpu.h_in != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu.h_in), ok = false);
        gpu.h_in = nullptr;
    }
    if (gpu.stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(gpu.stream), ok = false);
        gpu.stream = nullptr;
    }
    return GpuContextManager::unregisterTask(gpu) && ok;
}

GpuCacheRequest ResultTaskResources::request() const {
    return {gpu.gpuId, gpu.stream, gpu.d_input, gpu.inBytes};
}

bool TaskA::load(std::size_t bytes) {
    return resources.load(bytes);
}

bool TaskA::unload() {
    return resources.unload();
}

bool TaskA::execute(GpuDataCache& cache, ResultAtom& atom) {
    GPUINFRA_NVTX_SCOPE("ResultPipeline.TaskA");
    TaskGpuResources& gpu = resources.gpu;
    if (atom.ready || atom.frame.size() != gpu.inBytes || atom.result.size() != gpu.inBytes || atom.metadata.bytes != gpu.inBytes || !GpuContextManager::makeTaskCurrent(gpu)) return false;
    GpuDataAccess access = cache.getCacheData(atom.metadata, resources.request());
    // Each frame has a unique result ID; a producer hit is a harness error.
    if (!access || access.status() == CacheStatus::CacheHit) return false;
    std::memcpy(gpu.h_in, atom.frame.data(), gpu.inBytes);
    bool submitted = true;
    CUDA_CHECK(cudaMemcpyAsync(gpu.d_input, gpu.h_in, gpu.inBytes, cudaMemcpyHostToDevice, gpu.stream), submitted = false);
    if (submitted) submitted = runResultKernel(gpu.d_input, gpu.d_scratch, gpu.inBytes, gpu.stream);
    // On fallback d_input is reused only after the kernel has read it.
    if (submitted) {
        CUDA_CHECK(cudaMemcpyAsync(access.writableData(), gpu.d_scratch, gpu.inBytes, cudaMemcpyDeviceToDevice, gpu.stream), submitted = false);
    }
    if (submitted) {
        CUDA_CHECK(cudaMemcpyAsync(gpu.h_in, gpu.d_scratch, gpu.inBytes, cudaMemcpyDeviceToHost, gpu.stream), submitted = false);
    }
    const bool finished = access.freeCacheData(submitted);
    if (!submitted || !finished) return false;
    std::memcpy(atom.result.data(), gpu.h_in, gpu.inBytes);
    atom.ready = true;
    return true;
}

DummyCPUTask::DummyCPUTask(std::chrono::microseconds delay) : delay(delay) {}

bool DummyCPUTask::execute(ResultAtom& atom) const {
    GPUINFRA_NVTX_SCOPE("ResultPipeline.CPU");
    if (!atom.ready || atom.result.empty()) return false;
    // Read the CPU atom, never hold a GPU lease during the CPU gap.
    atom.checksum = std::accumulate(atom.result.begin(), atom.result.end(), std::uint64_t{0});
    if (delay.count() > 0) std::this_thread::sleep_for(delay);
    ++atom.cpuStages;
    return true;
}

bool TaskB::load(std::size_t bytes) {
    hits = 0;
    uploadedBytes = 0;
    return resources.load(bytes);
}

bool TaskB::unload() {
    return resources.unload();
}

bool TaskB::execute(GpuDataCache& cache, const ResultAtom& atom, std::size_t requiredCpuStages, CacheRetention retention) {
    GPUINFRA_NVTX_SCOPE("ResultPipeline.TaskB");
    TaskGpuResources& gpu = resources.gpu;
    if (!atom.ready || atom.cpuStages != requiredCpuStages || atom.frame.size() != gpu.inBytes || atom.result.size() != gpu.inBytes || atom.metadata.bytes != gpu.inBytes || !GpuContextManager::makeTaskCurrent(gpu)) return false;
    GpuDataAccess access = cache.getCacheData(atom.metadata, resources.request());
    if (!access) return false;
    bool submitted = true;
    if (access.status() == CacheStatus::CacheHit) {
        ++hits;
    }
    else {
        std::memcpy(gpu.h_in, atom.result.data(), gpu.inBytes);
        CUDA_CHECK(cudaMemcpyAsync(access.writableData(), gpu.h_in, gpu.inBytes, cudaMemcpyHostToDevice, gpu.stream), submitted = false);
        if (submitted) uploadedBytes += gpu.inBytes;
    }
    if (submitted) {
        CUDA_CHECK(cudaMemcpyAsync(gpu.d_scratch, access.data(), gpu.inBytes, cudaMemcpyDeviceToDevice, gpu.stream), submitted = false);
    }
    if (submitted) {
        CUDA_CHECK(cudaMemcpyAsync(gpu.h_in, gpu.d_scratch, gpu.inBytes, cudaMemcpyDeviceToHost, gpu.stream), submitted = false);
    }
    const bool finished = access.freeCacheData(submitted, retention);
    if (!submitted || !finished || std::memcmp(gpu.h_in, atom.result.data(), gpu.inBytes) != 0) return false;
    // Independent expected output catches corruption even after CPU refill.
    const unsigned char* h_result = static_cast<const unsigned char*>(gpu.h_in);
    for (std::size_t index = 0; index < gpu.inBytes; ++index) {
        if (h_result[index] != static_cast<unsigned char>(atom.frame[index] ^ 0x5aU)) return false;
    }
    return requiredCpuStages == 0 || atom.checksum == std::accumulate(atom.result.begin(), atom.result.end(), std::uint64_t{0});
}

}  // namespace result_pipeline
