#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "Cel.h"
#include "GpuContextManager.h"
#include "GraphStuff.h"
#include "IAlgo.h"
#include "Sdd.h"

namespace {

constexpr int THREADS_PER_GPU = 3;
constexpr std::uint64_t FRAMES_PER_GPU = 24;
constexpr std::size_t FRAME_BYTES = 1024U * 1024U;

IAlgo* makeCel() {
    return new Cel();
}

IAlgo* makeSdd() {
    return new Sdd();
}

void runGraphWorker(int numaNode, int gpuId, GraphFrameSource& source, GraphSink& sink, std::atomic<int>& workerFailures) {
    DummyGraph graph(numaNode, gpuId, sink);
    if (!graph.registerParameters() || !graph.load() || !graph.notifyParameters()) {
        std::cerr << "worker setup failed: gpu=" << gpuId << " numa=" << numaNode << '\n';
        ++workerFailures;
        graph.unload();
        return;
    }

    Frame frame;
    while (source.nextFrame(frame)) {
        if (!graph.execute(frame)) {
            std::cerr << "frame execution failed: gpu=" << gpuId << " frame=" << frame.id << '\n';
            ++workerFailures;
            break;
        }
    }
    graph.unload();
}

}  // namespace

int main() {
    std::vector<AlgoFactory> factories{
        {"cel", makeCel},
        {"sdd", makeSdd},
    };

    GpuInfraConfig config;
    config.threadsPerGpu = THREADS_PER_GPU;
    config.requireNuma = true;
    config.inputBytes = FRAME_BYTES;

    if (!GpuContextManager::init(config, factories)) {
        std::cerr << "GpuContextManager::init failed\n";
        return 1;
    }

    AlgoRuntimeInfo runtime;
    runtime.numThreads = THREADS_PER_GPU;
    runtime.inBytes = FRAME_BYTES;
    runtime.frameW = 1024;
    runtime.frameH = 1024;
    runtime.frameDtype = 1;
    runtime.params.name = "demo";
    if (!GpuContextManager::configure(runtime)) {
        std::cerr << "GpuContextManager::configure failed\n";
        GpuContextManager::shutdown();
        return 1;
    }

    const std::size_t gpuCount = GpuContextManager::gpuCount();
    GraphSink sink;
    std::atomic<int> workerFailures{0};

    std::vector<std::unique_ptr<GraphFrameSource>> sources;
    sources.reserve(gpuCount);
    for (std::size_t gpu = 0; gpu < gpuCount; ++gpu) {
        const int node = GpuContextManager::numaNodeForGpu(static_cast<int>(gpu));
        if (node < 0) {
            std::cerr << "No NUMA mapping for GPU " << gpu << '\n';
            GpuContextManager::shutdown();
            return 1;
        }
        sources.push_back(std::make_unique<GraphFrameSource>(FRAMES_PER_GPU, FRAME_BYTES, node, static_cast<std::uint64_t>(gpu) * FRAMES_PER_GPU));
    }

    std::vector<std::thread> workers;
    workers.reserve(gpuCount * static_cast<std::size_t>(THREADS_PER_GPU));
    for (std::size_t gpu = 0; gpu < gpuCount; ++gpu) {
        const int gpuId = static_cast<int>(gpu);
        const int node = GpuContextManager::numaNodeForGpu(gpuId);
        for (int worker = 0; worker < THREADS_PER_GPU; ++worker) {
            workers.emplace_back(runGraphWorker, node, gpuId, std::ref(*sources[gpu]), std::ref(sink), std::ref(workerFailures));
        }
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    const std::size_t delivered = sink.count();
    const std::size_t failedResults = sink.failureCount();
    const std::size_t expected = gpuCount * static_cast<std::size_t>(FRAMES_PER_GPU);
    GpuContextManager::shutdown();

    std::cout << "delivered " << delivered << " frames across " << gpuCount << " CUDA GPU(s), failures=" << failedResults << '\n';

    if (delivered != expected || failedResults != 0 || workerFailures.load() != 0) {
        std::cerr << "expected=" << expected << " worker_failures=" << workerFailures.load() << '\n';
        return 1;
    }
    return 0;
}
