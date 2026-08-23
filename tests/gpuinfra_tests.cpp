#include <algorithm>
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

#include <cuda_runtime.h>

#include "DummyGraph.h"
#include "DummyTask.h"
#include "FrameCpuAtom.h"
#include "FrameGpuAccess.h"
#include "FrameGpuData.h"
#include "FrameSlot.h"
#include "GpuContextManager.h"
#include "ImageSizing.h"
#include "ParameterRegistry.h"
#include "StaticData.h"
#include "TaskGpuResources.h"

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

FrameMetadata makeFrameMetadata(std::uint64_t frameId, const AlgoRuntimeInfo& runtime) {
    FrameMetadata metadata;
    metadata.id = frameId;
    metadata.bytes = runtime.inBytes;
    metadata.width = runtime.frameW;
    metadata.height = runtime.frameH;
    metadata.dtype = runtime.frameDtype;
    return metadata;
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

bool initializeFrame(FrameSlot& frame, const GpuLocation& location) {
    bool initialized = false;
    std::thread setup([&frame, &location, &initialized] {
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            initialized = frame.initializeGpuData({location.gpuId});
        }
    });
    setup.join();
    return initialized;
}

bool releaseFrame(FrameSlot& frame, const GpuLocation& location) {
    bool released = false;
    std::thread teardown([&frame, &location, &released] {
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            released = frame.releaseGpuData();
        }
    });
    teardown.join();
    return released;
}

bool initializeStaticData(StaticData& staticData, const GpuLocation& location, const AlgoRuntimeInfo& runtime, const std::vector<StaticFrameConfig>& frames) {
    bool initialized = false;
    std::thread setup([&staticData, &location, &runtime, &frames, &initialized] {
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            StaticDataConfig config;
            config.numaNode = location.numaNode;
            config.gpuIds = {location.gpuId};
            config.runtime = runtime;
            config.frames = frames;
            initialized = staticData.init(config);
        }
    });
    setup.join();
    return initialized;
}

bool releaseStaticData(StaticData& staticData, const GpuLocation& location) {
    bool released = false;
    std::thread teardown([&staticData, &location, &released] {
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            released = staticData.release();
        }
    });
    teardown.join();
    return released;
}

FrameSlot* findFrameSlot(StaticData& staticData, const FrameMetadata& metadata) {
    return staticData.findFrameSlot(metadata);
}

bool executeTask(DummyTask& task, FrameCpuAtom& atom, StaticData& staticData, const GpuLocation& location) {
    bool succeeded = false;
    std::thread worker([&task, &atom, &staticData, &location, &succeeded] {
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            succeeded = staticData.execute() && task.execute(atom, staticData);
        }
    });
    worker.join();
    return succeeded;
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

void testStaticDataValidation(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    std::vector<StaticFrameConfig> overflowFrames;
    overflowFrames.reserve(StaticData::FRAME_SLOT_POOL_SIZE + 1);
    for (std::size_t index = 0; index <= StaticData::FRAME_SLOT_POOL_SIZE; ++index) {
        overflowFrames.push_back({makeFrameMetadata(static_cast<std::uint64_t>(index), runtime), FramePhase::Timed});
    }

    StaticData overflowData;
    test.expect(!initializeStaticData(overflowData, location, runtime, overflowFrames), "reject more than 220 configured StaticData frames");
    test.expect(overflowData.frameSlotPoolSize() == 0 && overflowData.release(), "capacity rejection leaves StaticData empty");

    const FrameMetadata duplicateMetadata = makeFrameMetadata(41, runtime);
    StaticData duplicateData;
    test.expect(!initializeStaticData(duplicateData, location, runtime, {{duplicateMetadata, FramePhase::Warmup}, {duplicateMetadata, FramePhase::Timed}}), "reject duplicate StaticData frame IDs");
    test.expect(duplicateData.frameSlotPoolSize() == 0 && duplicateData.release(), "duplicate rejection leaves StaticData empty");
}

void testFrameGpuAccessState(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    const FrameMetadata metadata = makeFrameMetadata(0, runtime);
    FrameSlot frame(metadata, location.numaNode, FramePhase::Timed, runtime);
    const bool initialized = initializeFrame(frame, location);
    test.expect(initialized, "initialize frame GPU access state");
    if (!initialized) {
        return;
    }

    TaskGpuResources resources;
    resources.gpuId = location.gpuId;
    resources.numaNode = location.numaNode;
    bool streamReady = cudaSetDevice(location.gpuId) == cudaSuccess;
    if (streamReady) {
        streamReady = cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking) == cudaSuccess;
    }
    test.expect(streamReady, "create frame GPU access test stream");
    if (!streamReady) {
        test.expect(releaseFrame(frame, location), "release frame after stream setup failure");
        return;
    }

    test.expect(frame.deviceData.replicaCount() == 1 && frame.deviceData.bytes() == runtime.inBytes, "frame GPU data has one correctly sized replica");
    FrameGpuAccess prematureRead = frame.deviceData.acquire(FrameGpuAccessMode::Read, frame.metadata.id, resources);
    test.expect(!prematureRead, "reject read before first frame upload");

    FrameGpuAccess firstUpload = frame.deviceData.acquire(FrameGpuAccessMode::Upload, frame.metadata.id, resources);
    const bool beganFirstUpload = static_cast<bool>(firstUpload);
    test.expect(beganFirstUpload, "acquire first frame upload");
    void* d_frameData = nullptr;
    if (beganFirstUpload) {
        d_frameData = firstUpload.writableData();
        test.expect(firstUpload.data() != nullptr && d_frameData != nullptr, "upload access exposes tracked readable and writable views");
        const bool memsetSucceeded = cudaMemsetAsync(d_frameData, 0x2a, firstUpload.bytes(), resources.stream) == cudaSuccess;
        test.expect(firstUpload.complete(memsetSucceeded), "commit first frame upload");
    }
    test.expect(frame.deviceData.hasData(frame.metadata.id) && frame.deviceData.frameId() == frame.metadata.id, "uploaded GPU payload records the logical frame ID");

    FrameGpuAccess read = frame.deviceData.acquire(FrameGpuAccessMode::Read, frame.metadata.id, resources);
    const bool beganRead = static_cast<bool>(read);
    test.expect(beganRead, "acquire same-GPU frame read");
    if (beganRead) {
        test.expect(read.data() != nullptr && read.writableData() == nullptr, "read access does not expose a mutable frame pointer");
        test.expect(read.complete(true), "finish same-GPU frame read");
    }
    test.expect(frame.deviceData.hasData(frame.metadata.id) && frame.deviceData.frameId() == frame.metadata.id, "read keeps the same resident frame ID");

    const std::uint64_t nextFrameId = frame.metadata.id + 1;
    {
        FrameGpuAccess abandonedUpload = frame.deviceData.acquire(FrameGpuAccessMode::Upload, nextFrameId, resources);
        test.expect(static_cast<bool>(abandonedUpload), "acquire next-frame upload used for scope abort");
    }
    test.expect(!frame.deviceData.hasData(frame.metadata.id) && !frame.deviceData.hasData(nextFrameId), "starting a new upload invalidates the previous frame payload");

    FrameGpuAccess failedUpload = frame.deviceData.acquire(FrameGpuAccessMode::Upload, nextFrameId, resources);
    const bool beganFailedUpload = static_cast<bool>(failedUpload);
    test.expect(beganFailedUpload, "acquire next-frame upload used for explicit failure");
    if (beganFailedUpload) {
        test.expect(!failedUpload.complete(false), "failed upload does not publish the next frame ID");
    }
    test.expect(!frame.deviceData.hasData(nextFrameId), "failed upload leaves the GPU payload invalid");

    FrameGpuAccess secondUpload = frame.deviceData.acquire(FrameGpuAccessMode::Upload, nextFrameId, resources);
    const bool beganSecondUpload = static_cast<bool>(secondUpload);
    test.expect(beganSecondUpload, "acquire successful next-frame upload");
    if (beganSecondUpload) {
        test.expect(secondUpload.writableData() == d_frameData, "next frame reuses the existing device allocation");
        const bool memsetSucceeded = cudaMemsetAsync(secondUpload.writableData(), 0x17, secondUpload.bytes(), resources.stream) == cudaSuccess;
        test.expect(secondUpload.complete(memsetSucceeded), "commit successful next-frame upload");
    }
    test.expect(frame.deviceData.hasData(nextFrameId) && frame.deviceData.frameId() == nextFrameId, "successful upload publishes the next logical frame ID");

    FrameGpuAccess staleFrameRead = frame.deviceData.acquire(FrameGpuAccessMode::Read, frame.metadata.id, resources);
    test.expect(!staleFrameRead, "reject read using the previous logical frame ID");

    TaskGpuResources wrongGpuResources = resources;
    wrongGpuResources.gpuId = location.gpuId + 1;
    FrameGpuAccess wrongGpuRead = frame.deviceData.acquire(FrameGpuAccessMode::Read, nextFrameId, wrongGpuResources);
    test.expect(!wrongGpuRead, "reject access from a GPU without a replica");

    test.expect(cudaStreamDestroy(resources.stream) == cudaSuccess, "destroy frame GPU access test stream");
    resources.stream = nullptr;
    test.expect(releaseFrame(frame, location), "release frame GPU access state");
    test.expect(!frame.deviceData.isInitialized() && frame.deviceData.replicaCount() == 0, "frame GPU release resets state");
    test.expect(frame.releaseGpuData(), "repeated frame GPU release is harmless");
}

void testFrameDataAcrossTaskInstances(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask firstTask(90, location.numaNode, location.gpuId, ExecutionModel::Batched, runtime);
    DummyTask secondTask(91, location.numaNode, location.gpuId, ExecutionModel::Batched, runtime);
    const bool tasksReady = loadTask(firstTask) && loadTask(secondTask) && notifyTask(firstTask) && notifyTask(secondTask);
    test.expect(tasksReady, "prepare two task instances for frame GPU continuity");

    const FrameMetadata metadata = makeFrameMetadata(29, runtime);
    FrameCpuAtom atom(metadata);
    StaticData staticData;
    const bool frameReady = initializeStaticData(staticData, location, runtime, {{metadata, FramePhase::Timed}});
    FrameSlot* frame = findFrameSlot(staticData, metadata);
    test.expect(frameReady && frame != nullptr, "initialize shared StaticData frame GPU data");
    test.expect(staticData.frameSlotPoolSize() == StaticData::FRAME_SLOT_POOL_SIZE && staticData.frameSlotCount() == 1, "StaticData owns a fixed 220-slot pool with one bound frame");

    bool firstSucceeded = false;
    std::vector<AlgoOutput> firstOutputs;
    if (tasksReady && frameReady && frame != nullptr) {
        firstSucceeded = executeTask(firstTask, atom, staticData, location);
        firstOutputs = frame->result.outputs;
    }
    test.expect(firstSucceeded, "first task instance uploads and processes the frame");
    test.expect(frame != nullptr && frame->deviceData.hasData(frame->metadata.id) && frame->deviceData.frameId() == frame->metadata.id, "first task instance publishes the resident frame ID");

    std::fill(atom.data.begin(), atom.data.end(), 0);
    const bool secondSucceeded = tasksReady && frameReady && frame != nullptr && executeTask(secondTask, atom, staticData, location);
    test.expect(secondSucceeded, "second task instance consumes existing frame GPU data");
    test.expect(frame != nullptr && frame->deviceData.hasData(frame->metadata.id) && frame->deviceData.frameId() == frame->metadata.id, "read-only second task keeps the resident frame ID");

    bool outputsMatch = frame != nullptr && firstOutputs.size() == frame->result.outputs.size();
    for (std::size_t index = 0; outputsMatch && index < firstOutputs.size(); ++index) {
        const AlgoOutput& expected = firstOutputs[index];
        const AlgoOutput& actual = frame->result.outputs[index];
        outputsMatch = expected.algoName == actual.algoName && expected.width == actual.width && expected.height == actual.height && expected.data == actual.data;
    }
    test.expect(outputsMatch, "second task reads frame-owned GPU data instead of modified host input");

    test.expect(releaseStaticData(staticData, location), "release shared StaticData frame GPU data");
    const bool firstTaskUnloaded = unloadTask(firstTask);
    const bool secondTaskUnloaded = unloadTask(secondTask);
    test.expect(firstTaskUnloaded && secondTaskUnloaded, "unload both frame continuity task instances");
}

void testLifecycleAndResults(TestContext& test, const GpuLocation& location, ExecutionModel model, int taskId) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask task(taskId, location.numaNode, location.gpuId, model, runtime);
    ParameterRegistry prematureRegistry;
    ParameterSnapshot prematureSnapshot;
    const FrameMetadata prematureMetadata = makeFrameMetadata(1, runtime);
    const FrameMetadata firstMetadata = makeFrameMetadata(7, runtime);
    const FrameMetadata secondMetadata = makeFrameMetadata(19, runtime);
    const FrameMetadata malformedMetadata = makeFrameMetadata(23, runtime);
    const FrameMetadata mismatchedAtomMetadata = makeFrameMetadata(24, runtime);
    const FrameMetadata mismatchedSlotMetadata = makeFrameMetadata(25, runtime);
    FrameCpuAtom prematureAtom(prematureMetadata);
    test.expect(!task.registerParameters(prematureRegistry), "reject registerParameters before load");
    test.expect(!task.notifyParameters(prematureSnapshot), "reject notifyParameters before registration");
    test.expect(loadTask(task), "load task resources");

    StaticData staticData;
    const std::vector<StaticFrameConfig> frameConfigs = {
        {prematureMetadata, FramePhase::Timed},
        {firstMetadata, FramePhase::Timed},
        {secondMetadata, FramePhase::Timed},
        {malformedMetadata, FramePhase::Timed},
        {mismatchedSlotMetadata, FramePhase::Timed},
    };
    test.expect(initializeStaticData(staticData, location, runtime, frameConfigs), "initialize task StaticData pool");
    FrameMetadata wrongLayoutMetadata = firstMetadata;
    ++wrongLayoutMetadata.width;
    test.expect(staticData.findFrameSlot(firstMetadata) != nullptr, "find sparse frame ID through StaticData index");
    test.expect(staticData.findFrameSlot(wrongLayoutMetadata) == nullptr, "reject indexed frame ID with mismatched metadata");
    test.expect(staticData.execute() && !task.execute(prematureAtom, staticData), "reject execute before notification");
    test.expect(!task.load(), "reject repeated load");
    test.expect(notifyTask(task), "register and notify shared parameters");

    FrameCpuAtom firstAtom(firstMetadata);
    FrameCpuAtom secondAtom(secondMetadata);
    FrameSlot* firstFrame = findFrameSlot(staticData, firstMetadata);
    FrameSlot* secondFrame = findFrameSlot(staticData, secondMetadata);
    test.expect(firstFrame != nullptr && secondFrame != nullptr, "find bound task frames in StaticData");
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
            firstSucceeded = staticData.execute() && task.execute(firstAtom, staticData);
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
            secondSucceeded = staticData.execute() && task.execute(secondAtom, staticData);
        }
    });
    firstWorker.join();
    secondWorker.join();

    test.expect(firstThreadId != secondThreadId, "mobility test uses distinct live host threads");
    test.expect(firstSucceeded && secondSucceeded, "one task executes sequential frames on different threads");
    if (firstFrame != nullptr) {
        for (const AlgoOutput& output : firstFrame->result.outputs) {
            test.expect(verifyOutput(firstAtom, output, runtime.frameW), "first frame matches CPU reference");
        }
    }
    if (secondFrame != nullptr) {
        for (const AlgoOutput& output : secondFrame->result.outputs) {
            test.expect(verifyOutput(secondAtom, output, runtime.frameW), "second frame matches CPU reference");
        }
    }

    FrameCpuAtom malformedAtom(malformedMetadata);
    FrameSlot* malformedFrame = findFrameSlot(staticData, malformedMetadata);
    test.expect(malformedFrame != nullptr, "find malformed-input frame in StaticData");
    malformedAtom.data.pop_back();
    test.expect(staticData.execute() && !task.execute(malformedAtom, staticData), "reject malformed frame input");
    test.expect(malformedFrame != nullptr && !malformedFrame->result.ok, "malformed frame records failed task status");

    FrameCpuAtom mismatchedAtom(mismatchedAtomMetadata);
    FrameSlot* mismatchedFrame = findFrameSlot(staticData, mismatchedSlotMetadata);
    test.expect(mismatchedFrame != nullptr, "find intentionally mismatched slot in StaticData");
    test.expect(staticData.execute() && !task.execute(mismatchedAtom, staticData), "reject atom without matching StaticData slot metadata");
    test.expect(staticData.resultFor(mismatchedAtomMetadata) == nullptr && mismatchedFrame != nullptr && mismatchedFrame->result.ok, "missing slot does not mutate an unrelated frame result");

    test.expect(releaseStaticData(staticData, location), "release task StaticData pool");
    test.expect(unloadTask(task), "unload task resources");
    test.expect(task.unload(), "repeated unload is harmless");
    test.expect(task.lifecycle() == TaskLifecycle::Unloaded, "task reaches unloaded lifecycle state");
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

void testTemporaryTopologyGuard(TestContext& test, const GpuLocation& location) {
    GraphSink sink;
    std::atomic<bool> cancellation{false};
    GraphConfig config = makeGraphConfig(location, 1, 1, ExecutionModel::Batched);
    config.gpuIds = {location.gpuId, location.gpuId};
    DummyGraph graph(config, sink, cancellation);
    test.expect(!graph.initialize(), "reject more than one GPU per NUMA graph copy in temporary scope");
    test.expect(!cancellation.load(std::memory_order_acquire) && sink.count() == 0, "topology rejection occurs before graph execution");
    test.expect(graph.shutdown(), "unsupported topology has no resources to clean up");
}

void testIndependentPools(TestContext& test, const GpuLocation& location) {
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        DummyGraph graph(makeGraphConfig(location, 1, 2, ExecutionModel::Batched), sink, cancellation);
        test.expect(graph.initialize(), "initialize one-task two-worker graph");
        test.expect(runGraphPhase(graph, FramePhase::Timed), "run one-task two-worker graph");
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "DummyGraph free-task pool prevents concurrent reuse of its sole task instance");
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
    for (const auto& entry : grouped) {
        if (entry.second.size() != 1) {
            std::cout << "SKIP: temporary topology supports exactly one GPU in each NUMA graph copy\n";
            return;
        }
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
    testStaticDataValidation(test, locations.front());
    testFrameGpuAccessState(test, locations.front());
    testFrameDataAcrossTaskInstances(test, locations.front());
    testTemporaryTopologyGuard(test, locations.front());
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
