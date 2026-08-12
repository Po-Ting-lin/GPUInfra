#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "DummyTask.h"
#include "GpuContextManager.h"
#include "GraphStuff.h"
#include "ImageSizing.h"
#include "ParameterRegistry.h"

namespace {

class TestContext {
public:
    void expect(bool condition, const std::string& description) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << description << '\n';
        }
    }

    int failureCount() const {
        return failures;
    }

private:
    int failures = 0;
};

AlgoRuntimeInfo makeRuntime(int factor) {
    AlgoRuntimeInfo runtime;
    runtime.sizeFactor = factor;
    runtime.frameW = ImageSizing::scaledDimension(factor, ImageSizing::INPUT_MULTIPLIER);
    runtime.frameH = runtime.frameW;
    runtime.inBytes = ImageSizing::squareBytes(runtime.frameW, sizeof(std::uint8_t));
    runtime.frameDtype = 1;
    return runtime;
}

bool loadTask(DummyTask& task) {
    bool loaded = false;
    std::thread setup([&task, &loaded] { loaded = task.load(); });
    setup.join();
    return loaded;
}

bool notifyTask(DummyTask& task) {
    ParameterRegistry registry;
    ParameterSnapshot snapshot;
    return task.registerParameters(registry) && registry.setString(DummyTask::NAME_PARAMETER, "test") && registry.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && registry.seal() && registry.snapshot(snapshot) && task.notifyParameters(snapshot);
}

bool unloadTask(DummyTask& task) {
    bool unloaded = false;
    std::thread teardown([&task, &unloaded] { unloaded = task.unload(); });
    teardown.join();
    return unloaded;
}

std::uint32_t referenceValue(const FrameSlot& frame, int dimension, int x, int y, int inputStride) {
    std::uint32_t sum = 0;
    for (int k = 0; k < dimension; ++k) {
        const std::uint32_t a = frame.input[static_cast<std::size_t>(y) * inputStride + k];
        const std::uint32_t b = frame.input[static_cast<std::size_t>(k) * inputStride + x];
        sum += a * b;
    }
    return sum;
}

bool verifyOutput(const FrameSlot& frame, const AlgoOutput& output, int inputStride) {
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            std::uint32_t actual = 0;
            const std::size_t offset = (static_cast<std::size_t>(y) * output.width + x) * sizeof(actual);
            std::memcpy(&actual, output.data.data() + offset, sizeof(actual));
            if (actual != referenceValue(frame, output.width, x, y, inputStride)) {
                return false;
            }
        }
    }
    return true;
}

void testParameterRegistry(TestContext& test) {
    ParameterRegistry registry;
    ParameterSnapshot snapshot;
    test.expect(registry.registerParameter("value", ParameterType::String), "register string parameter");
    test.expect(registry.registerParameter("value", ParameterType::String), "repeat matching schema registration");
    test.expect(!registry.registerParameter("value", ParameterType::Bytes), "reject schema type mismatch");
    test.expect(!registry.setBytes("value", {1}), "reject value type mismatch");
    test.expect(registry.setString("value", "ready"), "set registered string value");
    test.expect(registry.seal(), "seal complete registry");
    test.expect(registry.isSealed(), "registry exposes sealed state");
    test.expect(!registry.setString("value", "changed"), "reject update after seal");
    test.expect(!registry.registerParameter("late", ParameterType::String), "reject registration after seal");
    test.expect(registry.snapshot(snapshot), "create immutable snapshot");
    std::string value;
    test.expect(snapshot.getString("value", value) && value == "ready", "read typed snapshot value");
    std::vector<std::uint8_t> bytes;
    test.expect(!snapshot.getBytes("value", bytes), "reject snapshot type mismatch");
}

void testLifecycleAndResults(TestContext& test, const GpuLocation& location, ExecutionModel model, int taskId) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask task(taskId, location.numaNode, location.gpuId, model, runtime);
    ParameterRegistry prematureRegistry;
    ParameterSnapshot prematureSnapshot;
    FrameSlot prematureFrame(1, location.numaNode, FramePhase::Timed, runtime);
    test.expect(!task.registerParameters(prematureRegistry), "reject registerParameters before load");
    test.expect(!task.notifyParameters(prematureSnapshot), "reject notifyParameters before registration");
    test.expect(!task.execute(prematureFrame), "reject execute before notification");
    test.expect(loadTask(task), "load task resources");
    test.expect(!task.load(), "reject repeated load");
    test.expect(notifyTask(task), "register and notify shared parameters");

    FrameSlot firstFrame(7, location.numaNode, FramePhase::Timed, runtime);
    FrameSlot secondFrame(19, location.numaNode, FramePhase::Timed, runtime);
    bool firstSucceeded = false;
    bool secondSucceeded = false;
    std::thread::id firstThreadId;
    std::thread::id secondThreadId;
    std::mutex sequenceLock;
    std::condition_variable sequenceCondition;
    int turn = 0;
    std::thread firstWorker([&] {
        firstThreadId = std::this_thread::get_id();
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            firstSucceeded = task.execute(firstFrame);
        }
        {
            std::lock_guard<std::mutex> guard(sequenceLock);
            turn = 1;
        }
        sequenceCondition.notify_one();
    });
    std::thread secondWorker([&] {
        secondThreadId = std::this_thread::get_id();
        const bool pinned = GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode);
        {
            std::unique_lock<std::mutex> guard(sequenceLock);
            sequenceCondition.wait(guard, [&turn] { return turn == 1; });
        }
        if (pinned) {
            secondSucceeded = task.execute(secondFrame);
        }
    });
    firstWorker.join();
    secondWorker.join();

    test.expect(firstThreadId != secondThreadId, "mobility test uses distinct live host threads");
    test.expect(firstSucceeded && secondSucceeded, "one task executes sequential frames on different threads");
    for (const AlgoOutput& output : firstFrame.result.outputs) {
        test.expect(verifyOutput(firstFrame, output, runtime.frameW), "first frame matches CPU reference");
    }
    for (const AlgoOutput& output : secondFrame.result.outputs) {
        test.expect(verifyOutput(secondFrame, output, runtime.frameW), "second frame matches CPU reference");
    }

    FrameSlot malformedFrame(23, location.numaNode, FramePhase::Timed, runtime);
    malformedFrame.input.pop_back();
    test.expect(!task.execute(malformedFrame), "reject malformed frame input");
    test.expect(!malformedFrame.result.ok, "malformed frame records failed task status");
    test.expect(unloadTask(task), "unload task resources");
    test.expect(task.unload(), "repeated unload is harmless");
    test.expect(task.lifecycle() == TaskLifecycle::Unloaded, "task reaches unloaded lifecycle state");
}

void testConcurrentExecutionGuard(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MAX_FACTOR);
    DummyTask task(100, location.numaNode, location.gpuId, ExecutionModel::Batched, runtime);
    test.expect(loadTask(task) && notifyTask(task), "prepare task for concurrent execution guard");
    FrameSlot firstFrame(31, location.numaNode, FramePhase::Timed, runtime);
    FrameSlot secondFrame(32, location.numaNode, FramePhase::Timed, runtime);

    std::mutex startLock;
    std::condition_variable startCondition;
    int ready = 0;
    bool released = false;
    bool firstResult = false;
    bool secondResult = false;
    auto execute = [&](FrameSlot& frame, bool& result) {
        const bool pinned = GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode);
        {
            std::unique_lock<std::mutex> guard(startLock);
            ++ready;
            startCondition.notify_all();
            startCondition.wait(guard, [&released] { return released; });
        }
        if (pinned) {
            result = task.execute(frame);
        }
    };

    std::thread firstWorker(execute, std::ref(firstFrame), std::ref(firstResult));
    std::thread secondWorker(execute, std::ref(secondFrame), std::ref(secondResult));
    {
        std::unique_lock<std::mutex> guard(startLock);
        startCondition.wait(guard, [&ready] { return ready == 2; });
        released = true;
    }
    startCondition.notify_all();
    firstWorker.join();
    secondWorker.join();

    test.expect(firstResult != secondResult, "exactly one concurrent execute call acquires the task instance");
    test.expect(unloadTask(task), "unload guarded task");
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

GraphConfig makeGraphConfig(const GpuLocation& location, std::size_t tasks, std::size_t workers, ExecutionModel model) {
    GraphConfig config;
    config.numaNode = location.numaNode;
    config.gpuIds = {location.gpuId};
    config.taskInstancesPerGpu = tasks;
    config.graphThreads = workers;
    config.warmupFramesPerGpu = 0;
    config.timedFramesPerGpu = 8;
    config.executionModel = model;
    config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    config.parameters.name = "graph-test";
    return config;
}

void testIndependentPools(TestContext& test, const GpuLocation& location) {
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        DummyGraph graph(makeGraphConfig(location, 1, 2, ExecutionModel::Batched), sink, cancellation);
        test.expect(graph.initialize(), "initialize one-task two-worker graph");
        test.expect(runGraphPhase(graph, FramePhase::Timed), "run one-task two-worker graph");
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "task pool limits concurrency independently");
        test.expect(sink.count() == 8 && sink.failureCount() == 0, "one-task graph completes every frame once");
        test.expect(graph.shutdown(), "shutdown one-task graph");
    }
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        DummyGraph graph(makeGraphConfig(location, 2, 1, ExecutionModel::Interleaved), sink, cancellation);
        test.expect(graph.initialize(), "initialize two-task one-worker graph");
        test.expect(runGraphPhase(graph, FramePhase::Timed), "run two-task one-worker graph");
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "worker pool limits concurrency independently");
        test.expect(sink.count() == 8 && sink.failureCount() == 0, "one-worker graph completes every frame once");
        test.expect(graph.shutdown(), "shutdown one-worker graph");
    }
}

void testConditionalNumaGraphs(TestContext& test, const std::vector<GpuLocation>& locations) {
    std::map<int, std::vector<int>> grouped;
    for (const GpuLocation& location : locations) {
        grouped[location.numaNode].push_back(location.gpuId);
    }
    if (grouped.size() < 2) {
        std::cout << "SKIP: multi-NUMA graph test requires GPUs on at least two NUMA nodes\n";
        return;
    }

    GraphSink sink;
    std::atomic<bool> cancellation{false};
    std::vector<std::unique_ptr<DummyGraph>> graphs;
    for (const auto& entry : grouped) {
        GraphConfig config;
        config.numaNode = entry.first;
        config.gpuIds = entry.second;
        config.taskInstancesPerGpu = 1;
        config.graphThreads = entry.second.size();
        config.timedFramesPerGpu = 1;
        config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
        config.parameters.name = "numa-test";
        graphs.push_back(std::make_unique<DummyGraph>(config, sink, cancellation));
    }
    bool ok = true;
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->initialize() && ok;
    }
    PhaseGate gate;
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->startPhase(FramePhase::Timed, gate) && ok;
    }
    gate.release();
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->waitForPhase() && ok;
    }
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->shutdown() && ok;
    }
    test.expect(ok && sink.count() == locations.size(), "one graph copy runs on each GPU-bearing NUMA node");
}

void testGraphCancellation(TestContext& test, const GpuLocation& location) {
    GraphSink sink;
    std::atomic<bool> cancellation{false};
    GraphConfig config = makeGraphConfig(location, 1, 2, static_cast<ExecutionModel>(99));
    config.warmupFramesPerGpu = 2;
    config.timedFramesPerGpu = 4;
    DummyGraph graph(config, sink, cancellation);
    test.expect(graph.initialize(), "initialize graph used for failure propagation");
    test.expect(!runGraphPhase(graph, FramePhase::Warmup), "execution failure fails graph phase");
    test.expect(cancellation.load(std::memory_order_acquire), "execution failure raises global cancellation");
    test.expect(sink.count() == 2 && sink.failureCount() == 2, "active failed phase gives every frame one terminal result");
    test.expect(graph.shutdown(), "failed graph still unloads cleanly");
    test.expect(sink.count() == 6 && sink.failureCount() == 6, "shutdown cancels every preallocated future frame exactly once");
}

}  // namespace

int main() {
    TestContext test;
    testParameterRegistry(test);

    GpuInfraConfig config;
    config.requireNuma = true;
    if (!GpuContextManager::init(config)) {
        std::cerr << "FAIL: CUDA/NUMA infrastructure initialization\n";
        return 1;
    }
    const std::vector<GpuLocation> locations = GpuContextManager::gpuLocations();
    if (locations.empty()) {
        std::cerr << "FAIL: no CUDA GPU locations\n";
        GpuContextManager::shutdown();
        return 1;
    }

    testLifecycleAndResults(test, locations.front(), ExecutionModel::Batched, 1);
    testLifecycleAndResults(test, locations.front(), ExecutionModel::Interleaved, 2);
    testConcurrentExecutionGuard(test, locations.front());
    testIndependentPools(test, locations.front());
    testConditionalNumaGraphs(test, locations);
    testGraphCancellation(test, locations.front());

    GpuContextManager::shutdown();
    if (test.failureCount() != 0) {
        std::cerr << test.failureCount() << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All GPUInfra tests passed\n";
    return 0;
}
