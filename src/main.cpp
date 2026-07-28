#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Cel.h"
#include "GpuContextManager.h"
#include "GraphStuff.h"
#include "IAlgo.h"
#include "ImageSizing.h"
#include "Mi.h"
#include "Sdd.h"

namespace {

constexpr int THREADS_PER_GPU = 4;
constexpr std::uint64_t DEFAULT_TIMED_FRAMES_PER_GPU = 200;
constexpr std::uint64_t DEFAULT_WARMUP_FRAMES_PER_GPU = 20;

class WorkerGate {
public:
    void arriveAndWait() {
        std::unique_lock<std::mutex> guard(lock);
        ++arrivedWorkers;
        arrivalCondition.notify_one();
        releaseCondition.wait(guard, [this] { return released; });
    }

    void waitForWorkers(std::size_t expectedWorkers) {
        std::unique_lock<std::mutex> guard(lock);
        arrivalCondition.wait(guard, [this, expectedWorkers] { return arrivedWorkers >= expectedWorkers; });
    }

    void releaseAll() {
        {
            std::lock_guard<std::mutex> guard(lock);
            released = true;
        }
        releaseCondition.notify_all();
    }

private:
    std::mutex lock;
    std::condition_variable arrivalCondition;
    std::condition_variable releaseCondition;
    std::size_t arrivedWorkers = 0;
    bool released = false;
};

std::unique_ptr<IAlgo> makeCel() {
    return std::make_unique<Cel>();
}

std::unique_ptr<IAlgo> makeSdd() {
    return std::make_unique<Sdd>();
}

std::unique_ptr<IAlgo> makeMi() {
    return std::make_unique<Mi>();
}

bool parseFrameCount(const char* text, bool allowZero, std::uint64_t& output) {
    try {
        const std::string value(text);
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || (!allowZero && parsed == 0)) {
            return false;
        }
        output = static_cast<std::uint64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parseExecutionModel(const char* text, ExecutionModel& output) {
    const std::string value(text);
    if (value == "batched") {
        output = ExecutionModel::Batched;
        return true;
    }
    if (value == "interleaved") {
        output = ExecutionModel::Interleaved;
        return true;
    }
    return false;
}

bool parseSizeFactor(const char* text, int& output) {
    try {
        const std::string value(text);
        std::size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed > static_cast<unsigned long long>(ImageSizing::MAX_FACTOR)) {
            return false;
        }
        const int factor = static_cast<int>(parsed);
        if (!ImageSizing::isValidFactor(factor)) {
            return false;
        }
        output = factor;
        return true;
    } catch (...) {
        return false;
    }
}

const char* executionModelName(ExecutionModel model) {
    if (model == ExecutionModel::Batched) {
        return "batched";
    }
    return "interleaved";
}

bool executeFrames(DummyGraph& graph, GraphFrameSource& source, const char* phase, int gpuId, std::atomic<int>& workerFailures) {
    Frame frame;
    while (source.nextFrame(frame)) {
        if (!graph.execute(frame)) {
            std::cerr << phase << " frame execution failed: gpu=" << gpuId << " frame=" << frame.id << '\n';
            ++workerFailures;
            return false;
        }
    }
    return true;
}

void runGraphWorker(
    int numaNode,
    int gpuId,
    ExecutionModel executionModel,
    const std::vector<AlgoFactory>& factories,
    const AlgoRuntimeInfo& runtime,
    GraphFrameSource& warmupSource,
    GraphFrameSource& timedSource,
    GraphSink& sink,
    WorkerGate& startGate,
    WorkerGate& finishGate,
    std::atomic<int>& workerFailures
) {
    DummyGraph graph(numaNode, gpuId, executionModel, factories, runtime, sink);
    bool canRun = graph.registerParameters() && graph.load() && graph.notifyParameters();
    if (!canRun) {
        std::cerr << "worker setup failed: gpu=" << gpuId << " numa=" << numaNode << '\n';
        ++workerFailures;
    }
    if (canRun) {
        canRun = executeFrames(graph, warmupSource, "warmup", gpuId, workerFailures);
    }

    startGate.arriveAndWait();
    if (canRun) {
        executeFrames(graph, timedSource, "timed", gpuId, workerFailures);
    }
    finishGate.arriveAndWait();
    graph.unload();
}

}  // namespace

int main(int argc, char* argv[]) {
    std::uint64_t timedFramesPerGpu = DEFAULT_TIMED_FRAMES_PER_GPU;
    std::uint64_t warmupFramesPerGpu = DEFAULT_WARMUP_FRAMES_PER_GPU;
    ExecutionModel executionModel = ExecutionModel::Batched;
    int sizeFactor = ImageSizing::DEFAULT_FACTOR;
    if (argc > 5 || (argc >= 2 && !parseFrameCount(argv[1], false, timedFramesPerGpu)) || (argc >= 3 && !parseFrameCount(argv[2], true, warmupFramesPerGpu)) || (argc >= 4 && !parseExecutionModel(argv[3], executionModel)) || (argc == 5 && !parseSizeFactor(argv[4], sizeFactor))) {
        std::cerr << "Usage: " << argv[0] << " [timed_frames_per_gpu] [warmup_frames_per_gpu] [batched|interleaved] [size_factor:16-256,multiple-of-16]\n";
        return 2;
    }
    const int frameWidth = ImageSizing::scaledDimension(sizeFactor, ImageSizing::INPUT_MULTIPLIER);
    const int frameHeight = frameWidth;
    const int celSize = ImageSizing::scaledDimension(sizeFactor, ImageSizing::CEL_MULTIPLIER);
    const int sddSize = ImageSizing::scaledDimension(sizeFactor, ImageSizing::SDD_MULTIPLIER);
    const int miSize = ImageSizing::scaledDimension(sizeFactor, ImageSizing::MI_MULTIPLIER);
    const std::size_t frameBytes = ImageSizing::squareBytes(frameWidth, sizeof(std::uint8_t));

    std::vector<AlgoFactory> factories{
        {"cel", makeCel},
        {"sdd", makeSdd},
        {"mi", makeMi},
    };

    GpuInfraConfig config;
    config.threadsPerGpu = THREADS_PER_GPU;
    config.requireNuma = true;
    config.inputBytes = frameBytes;

    if (!GpuContextManager::init(config)) {
        std::cerr << "GpuContextManager::init failed\n";
        return 1;
    }

    AlgoRuntimeInfo runtime;
    runtime.inBytes = frameBytes;
    runtime.sizeFactor = sizeFactor;
    runtime.frameW = frameWidth;
    runtime.frameH = frameHeight;
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
    WorkerGate startGate;
    WorkerGate finishGate;

    std::vector<std::unique_ptr<GraphFrameSource>> warmupSources;
    std::vector<std::unique_ptr<GraphFrameSource>> timedSources;
    warmupSources.reserve(gpuCount);
    timedSources.reserve(gpuCount);
    for (std::size_t gpu = 0; gpu < gpuCount; ++gpu) {
        const int node = GpuContextManager::numaNodeForGpu(static_cast<int>(gpu));
        if (node < 0) {
            std::cerr << "No NUMA mapping for GPU " << gpu << '\n';
            GpuContextManager::shutdown();
            return 1;
        }
        const std::uint64_t idBase = static_cast<std::uint64_t>(gpu) * (warmupFramesPerGpu + timedFramesPerGpu);
        warmupSources.push_back(std::make_unique<GraphFrameSource>(warmupFramesPerGpu, frameBytes, node, idBase));
        timedSources.push_back(std::make_unique<GraphFrameSource>(timedFramesPerGpu, frameBytes, node, idBase + warmupFramesPerGpu));
    }

    const std::size_t workerCount = gpuCount * static_cast<std::size_t>(THREADS_PER_GPU);
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (std::size_t gpu = 0; gpu < gpuCount; ++gpu) {
        const int gpuId = static_cast<int>(gpu);
        const int node = GpuContextManager::numaNodeForGpu(gpuId);
        for (int worker = 0; worker < THREADS_PER_GPU; ++worker) {
            workers.emplace_back(runGraphWorker, node, gpuId, executionModel, std::cref(factories), std::cref(runtime), std::ref(*warmupSources[gpu]), std::ref(*timedSources[gpu]), std::ref(sink), std::ref(startGate), std::ref(finishGate), std::ref(workerFailures));
        }
    }

    startGate.waitForWorkers(workerCount);
    const auto timedStart = std::chrono::steady_clock::now();
    startGate.releaseAll();
    finishGate.waitForWorkers(workerCount);
    const auto timedEnd = std::chrono::steady_clock::now();
    finishGate.releaseAll();

    for (std::thread& worker : workers) {
        worker.join();
    }

    const std::size_t delivered = sink.count();
    const std::size_t failedResults = sink.failureCount();
    const std::size_t expected = gpuCount * static_cast<std::size_t>(warmupFramesPerGpu + timedFramesPerGpu);
    const std::uint64_t measuredFrames = static_cast<std::uint64_t>(gpuCount) * timedFramesPerGpu;
    const double elapsedSeconds = std::chrono::duration<double>(timedEnd - timedStart).count();
    const double elapsedMilliseconds = elapsedSeconds * 1000.0;
    const double throughput = static_cast<double>(measuredFrames) / elapsedSeconds;
    const double millisecondsPerFrame = elapsedMilliseconds / static_cast<double>(measuredFrames);
    GpuContextManager::shutdown();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "execution_model=" << executionModelName(executionModel) << '\n';
    std::cout << "size_factor=" << sizeFactor << " input=" << frameWidth << 'x' << frameHeight << " cel=" << celSize << 'x' << celSize << " sdd=" << sddSize << 'x' << sddSize << " mi=" << miSize << 'x' << miSize << '\n';
    std::cout << "warmup " << warmupFramesPerGpu << " frames/GPU, timed " << timedFramesPerGpu << " frames/GPU (" << measuredFrames << " total) in " << elapsedMilliseconds << " ms\n";
    std::cout << "throughput=" << throughput << " frames/s, average=" << millisecondsPerFrame << " ms/frame\n";
    std::cout << "delivered " << delivered << " frames across " << gpuCount << " CUDA GPU(s), failures=" << failedResults << '\n';

    if (delivered != expected || failedResults != 0 || workerFailures.load() != 0) {
        std::cerr << "expected=" << expected << " worker_failures=" << workerFailures.load() << '\n';
        return 1;
    }
    return 0;
}
