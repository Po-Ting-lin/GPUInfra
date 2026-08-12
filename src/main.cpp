#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "GpuContextManager.h"
#include "GraphStuff.h"
#include "ImageSizing.h"

namespace {

constexpr std::size_t TASKS_PER_GPU = 4;
constexpr std::uint64_t DEFAULT_TIMED_FRAMES_PER_GPU = 200;
constexpr std::uint64_t DEFAULT_WARMUP_FRAMES_PER_GPU = 20;

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
    return model == ExecutionModel::Batched ? "batched" : "interleaved";
}

bool runPhase(std::vector<std::unique_ptr<DummyGraph>>& graphs, FramePhase phase, std::atomic<bool>& cancellation) {
    PhaseGate gate;
    std::vector<bool> started(graphs.size(), false);
    bool ok = true;
    for (std::size_t index = 0; index < graphs.size(); ++index) {
        started[index] = graphs[index]->startPhase(phase, gate);
        if (!started[index]) {
            ok = false;
            cancellation.store(true, std::memory_order_release);
        }
    }
    gate.release();
    for (std::size_t index = 0; index < graphs.size(); ++index) {
        if (started[index] && !graphs[index]->waitForPhase()) {
            ok = false;
        }
    }
    return ok && !cancellation.load(std::memory_order_acquire);
}

bool startPhase(std::vector<std::unique_ptr<DummyGraph>>& graphs, FramePhase phase, PhaseGate& gate, std::vector<bool>& started, std::atomic<bool>& cancellation) {
    bool ok = true;
    started.assign(graphs.size(), false);
    for (std::size_t index = 0; index < graphs.size(); ++index) {
        started[index] = graphs[index]->startPhase(phase, gate);
        if (!started[index]) {
            ok = false;
            cancellation.store(true, std::memory_order_release);
        }
    }
    return ok;
}

bool waitForPhase(std::vector<std::unique_ptr<DummyGraph>>& graphs, const std::vector<bool>& started) {
    bool ok = true;
    for (std::size_t index = 0; index < graphs.size(); ++index) {
        if (started[index] && !graphs[index]->waitForPhase()) {
            ok = false;
        }
    }
    return ok;
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
    if (warmupFramesPerGpu > std::numeric_limits<std::uint64_t>::max() - timedFramesPerGpu) {
        std::cerr << "combined frame count is too large\n";
        return 2;
    }
    const std::uint64_t totalFramesPerGpu = warmupFramesPerGpu + timedFramesPerGpu;

    const int frameWidth = ImageSizing::scaledDimension(sizeFactor, ImageSizing::INPUT_MULTIPLIER);
    const int frameHeight = frameWidth;
    const int celSize = ImageSizing::scaledDimension(sizeFactor, ImageSizing::CEL_MULTIPLIER);
    const int sddSize = ImageSizing::scaledDimension(sizeFactor, ImageSizing::SDD_MULTIPLIER);
    const int miSize = ImageSizing::scaledDimension(sizeFactor, ImageSizing::MI_MULTIPLIER);
    const std::size_t frameBytes = ImageSizing::squareBytes(frameWidth, sizeof(std::uint8_t));

    GpuInfraConfig infrastructureConfig;
    infrastructureConfig.requireNuma = true;
    if (!GpuContextManager::init(infrastructureConfig)) {
        std::cerr << "GpuContextManager::init failed\n";
        return 1;
    }

    AlgoRuntimeInfo runtime;
    runtime.inBytes = frameBytes;
    runtime.sizeFactor = sizeFactor;
    runtime.frameW = frameWidth;
    runtime.frameH = frameHeight;
    runtime.frameDtype = 1;

    AlgoParams parameters;
    parameters.name = "demo";

    const std::vector<GpuLocation> locations = GpuContextManager::gpuLocations();
    std::map<int, std::vector<int>> gpusByNumaNode;
    for (const GpuLocation& location : locations) {
        gpusByNumaNode[location.numaNode].push_back(location.gpuId);
    }

    GraphSink sink;
    std::atomic<bool> cancellation{false};
    std::vector<std::unique_ptr<DummyGraph>> graphs;
    graphs.reserve(gpusByNumaNode.size());
    std::uint64_t firstFrameId = 0;
    bool graphConfigsValid = true;
    for (const auto& entry : gpusByNumaNode) {
        if (totalFramesPerGpu > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(entry.second.size())) {
            graphConfigsValid = false;
            break;
        }
        const std::uint64_t graphFrameCount = totalFramesPerGpu * static_cast<std::uint64_t>(entry.second.size());
        if (firstFrameId > std::numeric_limits<std::uint64_t>::max() - graphFrameCount) {
            graphConfigsValid = false;
            break;
        }
        GraphConfig graphConfig;
        graphConfig.numaNode = entry.first;
        graphConfig.gpuIds = entry.second;
        graphConfig.taskInstancesPerGpu = TASKS_PER_GPU;
        graphConfig.graphThreads = TASKS_PER_GPU * entry.second.size();
        graphConfig.warmupFramesPerGpu = warmupFramesPerGpu;
        graphConfig.timedFramesPerGpu = timedFramesPerGpu;
        graphConfig.firstFrameId = firstFrameId;
        graphConfig.executionModel = executionModel;
        graphConfig.runtime = runtime;
        graphConfig.parameters = parameters;
        graphs.push_back(std::make_unique<DummyGraph>(graphConfig, sink, cancellation));
        firstFrameId += graphFrameCount;
    }

    bool runSucceeded = graphConfigsValid && !graphs.empty() && graphs.size() == gpusByNumaNode.size();
    if (runSucceeded) {
        for (const std::unique_ptr<DummyGraph>& graph : graphs) {
            if (!graph->initialize()) {
                runSucceeded = false;
                cancellation.store(true, std::memory_order_release);
                break;
            }
        }
    }

    if (runSucceeded) {
        runSucceeded = runPhase(graphs, FramePhase::Warmup, cancellation);
    }

    std::chrono::steady_clock::time_point timedStart;
    std::chrono::steady_clock::time_point timedEnd;
    if (runSucceeded) {
        PhaseGate timedGate;
        std::vector<bool> started;
        runSucceeded = startPhase(graphs, FramePhase::Timed, timedGate, started, cancellation);
        timedStart = std::chrono::steady_clock::now();
        timedGate.release();
        if (!waitForPhase(graphs, started)) {
            runSucceeded = false;
        }
        timedEnd = std::chrono::steady_clock::now();
        if (cancellation.load(std::memory_order_acquire)) {
            runSucceeded = false;
        }
    }

    bool shutdownSucceeded = true;
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        if (!graph->shutdown()) {
            shutdownSucceeded = false;
        }
    }
    GpuContextManager::shutdown();

    const std::size_t gpuCount = locations.size();
    const std::size_t delivered = sink.count();
    const std::size_t failedResults = sink.failureCount();
    const bool expectedFits = totalFramesPerGpu <= std::numeric_limits<std::size_t>::max() / gpuCount;
    const std::size_t expected = expectedFits ? gpuCount * static_cast<std::size_t>(totalFramesPerGpu) : 0;
    runSucceeded = runSucceeded && shutdownSucceeded && expectedFits && delivered == expected && failedResults == 0;
    if (!runSucceeded) {
        std::cerr << "graph run failed: expected=" << expected << " delivered=" << delivered << " failures=" << failedResults << '\n';
        return 1;
    }

    const std::uint64_t measuredFrames = static_cast<std::uint64_t>(gpuCount) * timedFramesPerGpu;
    const double elapsedSeconds = std::chrono::duration<double>(timedEnd - timedStart).count();
    const double elapsedMilliseconds = elapsedSeconds * 1000.0;
    const double throughput = static_cast<double>(measuredFrames) / elapsedSeconds;
    const double millisecondsPerFrame = elapsedMilliseconds / static_cast<double>(measuredFrames);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "execution_model=" << executionModelName(executionModel) << '\n';
    std::cout << "size_factor=" << sizeFactor << " input=" << frameWidth << 'x' << frameHeight << " cel=" << celSize << 'x' << celSize << " sdd=" << sddSize << 'x' << sddSize << " mi=" << miSize << 'x' << miSize << '\n';
    std::cout << "warmup " << warmupFramesPerGpu << " frames/GPU, timed " << timedFramesPerGpu << " frames/GPU (" << measuredFrames << " total) in " << elapsedMilliseconds << " ms\n";
    std::cout << "throughput=" << throughput << " frames/s, average=" << millisecondsPerFrame << " ms/frame\n";
    std::cout << "delivered " << delivered << " frames across " << gpuCount << " CUDA GPU(s), failures=" << failedResults << '\n';
    return 0;
}
