#include "test_support.h"

namespace gpuinfra_tests {

void testParameterRegistry(TestContext& test) {
    ParameterRegistry registry;
    ParameterSnapshot snapshot;
    test.expect(registry.registerParameter("value", ParameterType::String), "register string parameter");
    test.expect(registry.registerParameter("value", ParameterType::String), "repeat matching schema registration");
    test.expect(!registry.registerParameter("value", ParameterType::Bytes), "reject schema type mismatch");
    test.expect(!registry.setBytes("value", {1}), "reject value type mismatch");
    test.expect(registry.setString("value", "ready"), "set registered string value");
    const std::uint64_t initialRevision = registry.revision();
    test.expect(registry.seal(), "seal complete registry");
    test.expect(registry.isSealed(), "registry exposes sealed schema state");
    test.expect(!registry.registerParameter("late", ParameterType::String), "reject schema registration after seal");
    test.expect(registry.snapshot(snapshot) && snapshot.revision() == initialRevision, "snapshot captures the initial parameter revision");
    std::string value;
    test.expect(snapshot.getString("value", value) && value == "ready", "read typed snapshot value");
    std::vector<std::uint8_t> bytes;
    test.expect(!snapshot.getBytes("value", bytes), "reject snapshot type mismatch");
    test.expect(registry.setString("value", "ready") && registry.revision() == initialRevision, "same value preserves parameter revision");
    test.expect(registry.setString("value", "changed") && registry.revision() == initialRevision + 1, "changed value advances parameter revision after schema seal");
    test.expect(registry.snapshot(snapshot) && snapshot.revision() == initialRevision + 1 && snapshot.getString("value", value) && value == "changed", "updated snapshot exposes changed parameter value");
}

void testFrameDataAcrossTaskInstances(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask firstTask(90, ExecutionModel::Batched, runtime);
    DummyTask secondTask(91, ExecutionModel::Batched, runtime);
    ParameterRegistry parameters;
    const bool parametersReady = firstTask.registerParameters(parameters) && secondTask.registerParameters(parameters) && parameters.setString(DummyTask::NAME_PARAMETER, "test") && parameters.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && parameters.seal();
    const bool tasksReady = parametersReady && loadTask(firstTask, location) && loadTask(secondTask, location) && startTask(firstTask, location) && startTask(secondTask, location);
    test.expect(tasksReady, "prepare two task instances for frame GPU continuity");

    const FrameMetadata metadata = makeFrameMetadata(29, runtime);
    FrameCpuAtom atom(metadata, runtime);
    StaticData staticData;
    const bool frameReady = initializeStaticData(staticData, location, runtime, 1);
    const bool frameAccepted = staticData.validateFrame(metadata);
    test.expect(frameReady && frameAccepted, "initialize shared StaticData frame GPU data");
    test.expect(staticData.isInitialized() && staticData.gpuCacheEntryCount() == 1, "StaticData accepts incoming frames independently from cache capacity");

    bool firstSucceeded = false;
    std::vector<AlgoOutput> firstOutputs;
    if (tasksReady && frameReady && frameAccepted) {
        firstSucceeded = executeTask(firstTask, atom, staticData, location);
        firstOutputs = atom.result.outputs;
    }
    test.expect(firstSucceeded, "first task instance uploads and processes the frame");

    std::fill(atom.data.begin(), atom.data.end(), 0);
    const bool secondSucceeded = tasksReady && frameReady && frameAccepted && executeTask(secondTask, atom, staticData, location);
    test.expect(secondSucceeded, "second task instance consumes existing frame GPU data");

    bool outputsMatch = firstSucceeded && firstOutputs.size() == atom.result.outputs.size();
    for (std::size_t index = 0; outputsMatch && index < firstOutputs.size(); ++index) {
        const AlgoOutput& expected = firstOutputs[index];
        const AlgoOutput& actual = atom.result.outputs[index];
        outputsMatch = expected.algoName == actual.algoName && expected.width == actual.width && expected.height == actual.height && expected.data == actual.data;
    }
    test.expect(outputsMatch, "second task reads frame-owned GPU data instead of modified host input");

    const bool firstTaskStopped = stopTask(firstTask, location);
    const bool secondTaskStopped = stopTask(secondTask, location);
    test.expect(firstTaskStopped && secondTaskStopped, "stop both frame continuity task instances");
    test.expect(releaseStaticData(staticData, location), "release shared StaticData frame GPU data");
    const bool firstTaskUnloaded = unloadTask(firstTask, location);
    const bool secondTaskUnloaded = unloadTask(secondTask, location);
    test.expect(firstTaskUnloaded && secondTaskUnloaded, "unload both frame continuity task instances");
}

void testTaskFallbackExecution(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask task(92, ExecutionModel::Batched, runtime);
    ParameterRegistry parameters;
    const bool taskReady = configureTaskParameters(task, parameters) && loadTask(task, location) && startTask(task, location);
    test.expect(taskReady, "prepare task for zero-capacity fallback execution");

    const FrameMetadata metadata = makeFrameMetadata(30, runtime);
    FrameCpuAtom atom(metadata, runtime);
    StaticData staticData;
    const bool frameReady = initializeStaticData(staticData, location, runtime, 0);
    const bool frameAccepted = staticData.validateFrame(metadata);
    test.expect(frameReady && frameAccepted && staticData.gpuCacheEntryCount() == 0, "initialize arbitrary-frame input with GPU cache disabled");

    const bool succeeded = taskReady && frameReady && executeTask(task, atom, staticData, location);
    test.expect(succeeded, "task executes correctly through its fallback input buffer");
    if (frameAccepted) {
        for (const AlgoOutput& output : atom.result.outputs) {
            test.expect(verifyOutput(atom, output, runtime.frameW), "fallback execution matches CPU reference");
        }
    }

    test.expect(stopTask(task, location), "stop fallback execution task");
    test.expect(releaseStaticData(staticData, location), "release zero-capacity StaticData");
    test.expect(unloadTask(task, location), "unload fallback execution task");
}

void testLifecycleAndResults(TestContext& test, const GpuLocation& location, ExecutionModel model, int taskId) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask task(taskId, model, runtime);
    ParameterRegistry parameters;
    ParameterSnapshot parameterSnapshot;
    const FrameMetadata prematureMetadata = makeFrameMetadata(1, runtime);
    const FrameMetadata firstMetadata = makeFrameMetadata(7, runtime);
    const FrameMetadata secondMetadata = makeFrameMetadata(19, runtime);
    const FrameMetadata malformedMetadata = makeFrameMetadata(23, runtime);
    FrameMetadata mismatchedAtomMetadata = makeFrameMetadata(24, runtime);
    ++mismatchedAtomMetadata.dtype;
    const FrameMetadata mismatchedRecordMetadata = makeFrameMetadata(25, runtime);
    FrameCpuAtom prematureAtom(prematureMetadata, runtime);
    test.expect(prematureAtom.result.id == prematureMetadata.key.frameId && prematureAtom.result.ok && prematureAtom.result.outputs.size() == 3, "FrameCpuAtom owns a preallocated result");
    test.expect(!loadTask(task, location), "reject load before parameter registration");
    test.expect(!task.notifyParameters(parameterSnapshot), "reject notifyParameters before registration");
    test.expect(task.registerParameters(parameters), "register parameters immediately after construction");
    test.expect(task.lifecycle() == TaskLifecycle::Registered, "task reaches registered lifecycle state before load");
    test.expect(!task.registerParameters(parameters), "reject repeated parameter registration");
    test.expect(!loadTask(task, location), "reject load before initial parameters are complete");
    test.expect(parameters.setString(DummyTask::NAME_PARAMETER, "test") && parameters.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && parameters.seal() && parameters.snapshot(parameterSnapshot), "define initial parameters before load");
    test.expect(task.gpuId() == -1, "task has no GPU binding before load");
    test.expect(loadTask(task, location), "load task resources");
    test.expect(task.gpuId() == location.gpuId, "load derives GPU binding from the framework NUMA environment");
    test.expect(task.lifecycle() == TaskLifecycle::Loaded, "load applies initial parameters without a notify callback");

    StaticData staticData;
    test.expect(initializeStaticData(staticData, location, runtime), "initialize task StaticData cache");
    FrameMetadata wrongLayoutMetadata = firstMetadata;
    ++wrongLayoutMetadata.width;
    test.expect(staticData.validateFrame(firstMetadata), "accept an arbitrary frame with the configured layout");
    test.expect(!staticData.validateFrame(wrongLayoutMetadata), "reject an arbitrary frame with mismatched metadata");
    test.expect(staticData.execute() && !task.execute(prematureAtom, staticData), "reject execute before start");
    test.expect(!task.load(), "reject repeated load");
    test.expect(!stopTask(task, location), "reject stop before start");
    test.expect(startTask(task, location), "start task execution cycle");
    test.expect(task.lifecycle() == TaskLifecycle::Started, "task reaches started lifecycle state");
    test.expect(!startTask(task, location), "reject repeated task start");

    const std::uint64_t initialParameterRevision = parameters.revision();
    test.expect(parameters.setString(DummyTask::NAME_PARAMETER, "test") && parameters.revision() == initialParameterRevision && parameters.snapshot(parameterSnapshot) && notifyTask(task, parameterSnapshot, location), "same parameter values preserve the applied revision");
    test.expect(parameters.setString(DummyTask::NAME_PARAMETER, "changed") && parameters.revision() == initialParameterRevision + 1 && parameters.snapshot(parameterSnapshot) && notifyTask(task, parameterSnapshot, location), "changed parameters notify a started task at a quiescent boundary");

    FrameCpuAtom firstAtom(firstMetadata, runtime);
    FrameCpuAtom secondAtom(secondMetadata, runtime);
    const bool firstFrameAccepted = staticData.validateFrame(firstMetadata);
    const bool secondFrameAccepted = staticData.validateFrame(secondMetadata);
    test.expect(firstFrameAccepted && secondFrameAccepted, "accept unrelated incoming task frames without registration");
    bool firstSucceeded = false;
    bool secondSucceeded = false;
    std::thread::id firstThreadId;
    std::thread::id secondThreadId;
    std::mutex sequenceLock;
    std::condition_variable sequenceCondition;
    int turn = 0;
    std::thread firstWorker = NumaExecutor(location.numaNode).start([&](bool numaReady) {
        firstThreadId = std::this_thread::get_id();
        if (numaReady) {
            firstSucceeded = staticData.execute() && task.execute(firstAtom, staticData);
        }
        {
            std::lock_guard<std::mutex> guard(sequenceLock);
            turn = 1;
        }
        sequenceCondition.notify_one();
    });
    std::thread secondWorker = NumaExecutor(location.numaNode).start([&](bool numaReady) {
        secondThreadId = std::this_thread::get_id();
        {
            std::unique_lock<std::mutex> guard(sequenceLock);
            sequenceCondition.wait(guard, [&turn] { return turn == 1; });
        }
        if (numaReady) {
            secondSucceeded = staticData.execute() && task.execute(secondAtom, staticData);
        }
    });
    firstWorker.join();
    secondWorker.join();

    test.expect(firstThreadId != secondThreadId, "mobility test uses distinct live host threads");
    test.expect(firstSucceeded && secondSucceeded, "one task executes sequential frames on different threads");
    if (firstFrameAccepted) {
        for (const AlgoOutput& output : firstAtom.result.outputs) {
            test.expect(verifyOutput(firstAtom, output, runtime.frameW), "first frame matches CPU reference");
        }
    }
    if (secondFrameAccepted) {
        for (const AlgoOutput& output : secondAtom.result.outputs) {
            test.expect(verifyOutput(secondAtom, output, runtime.frameW), "second frame matches CPU reference");
        }
    }

    FrameCpuAtom malformedAtom(malformedMetadata, runtime);
    const bool malformedFrameAccepted = staticData.validateFrame(malformedMetadata);
    test.expect(malformedFrameAccepted, "accept metadata before the CPU input is malformed");
    malformedAtom.data.pop_back();
    test.expect(staticData.execute() && !task.execute(malformedAtom, staticData), "reject malformed frame input");
    test.expect(malformedFrameAccepted && !malformedAtom.result.ok, "malformed atom records failed task status");

    FrameCpuAtom mismatchedAtom(mismatchedAtomMetadata, runtime);
    const bool unrelatedFrameAccepted = staticData.validateFrame(mismatchedRecordMetadata);
    test.expect(unrelatedFrameAccepted, "accept unrelated metadata with the configured layout");
    test.expect(staticData.execute() && !task.execute(mismatchedAtom, staticData), "reject an atom with a mismatched dtype");
    test.expect(!mismatchedAtom.result.ok && staticData.validateFrame(mismatchedRecordMetadata), "wrong-layout frame fails without affecting other incoming frames");

    test.expect(stopTask(task, location), "stop task execution cycle");
    test.expect(task.lifecycle() == TaskLifecycle::Stopped, "task reaches stopped lifecycle state");
    test.expect(!task.execute(prematureAtom, staticData), "reject execute after stop");
    test.expect(releaseStaticData(staticData, location), "release task StaticData pool");
    test.expect(unloadTask(task, location), "unload task resources");
    test.expect(task.unload(), "repeated unload is harmless");
    test.expect(task.lifecycle() == TaskLifecycle::Unloaded && task.gpuId() == -1, "unload clears the task GPU binding");
}

void testGpuTopologySelection(TestContext& test) {
    const std::vector<GpuLocation> locations = {{7, 2}, {3, 0}, {8, 2}, {5, 1}};
    std::vector<int> gpuIds = {99};
    test.expect(!GpuTopology::resolveGpuIds(3, locations, gpuIds) && gpuIds.empty(), "reject a NUMA node with zero GPUs and clear stale results");
    test.expect(!GpuTopology::resolveGpuIds(2, locations, gpuIds) && gpuIds.empty(), "reject multiple GPUs instead of selecting the first");
    test.expect(GpuTopology::resolveGpuIds(1, locations, gpuIds) && gpuIds == std::vector<int>{5}, "resolve the sole local GPU without equating GPU and NUMA IDs");
    test.expect(GpuTopology::resolveGpuIds(0, locations, gpuIds) && gpuIds == std::vector<int>{3}, "ignore GPUs belonging to other NUMA nodes");
    test.expect(!GpuTopology::resolveGpuIds(-1, locations, gpuIds) && gpuIds.empty(), "reject failed NUMA detection");
    test.expect(!GpuTopology::resolveGpuIds(0, {}, gpuIds) && gpuIds.empty(), "reject an empty GPU topology");
    test.expect(!GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds) && gpuIds.empty(), "reject lookup before infrastructure initialization");
}

void testNumaExecutionBoundary(TestContext& test, const GpuLocation& location) {
    const NumaExecutor executor(location.numaNode);
    test.expect(executor.run([&location] {
        std::vector<int> gpuIds;
        return GpuTopology::currentNumaNode() == location.numaNode && GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds) && gpuIds == std::vector<int>{location.gpuId};
    }), "framework affinity establishes the NUMA identity used for GPU lookup");

    bool callbackCalled = false;
    test.expect(!NumaExecutor(-1).run([&callbackCalled] { callbackCalled = true; return true; }) && !callbackCalled, "failed framework affinity never invokes a task callback");
    GraphSink sink;
    std::atomic<bool> cancellation{false};
    GraphConfig config = makeGraphConfig(1, 1, ExecutionModel::Batched);
    DummyGraph graph(NumaExecutor(-1), config, sink, cancellation);
    test.expect(!graph.initialize(), "reject a graph without a valid NUMA execution environment");
    test.expect(graph.taskCount() == 0 && !cancellation.load(std::memory_order_acquire) && sink.count() == 0 && graph.shutdown(), "NUMA rejection occurs before task allocation or global cancellation");

    GraphConfig defaultWorkers = makeGraphConfig(3, 0, ExecutionModel::Batched);
    DummyGraph sizedGraph(executor, defaultWorkers, sink, cancellation);
    test.expect(sizedGraph.initialize() && sizedGraph.taskCount() == 3 && sizedGraph.workerCount() == 3, "derive task and default worker counts from resolved GPU count");
    test.expect(sizedGraph.shutdown(), "release graph with derived task and worker counts");
}

void testIndependentPools(TestContext& test, const GpuLocation& location) {
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        GraphConfig config = makeGraphConfig(1, 2, ExecutionModel::Batched);
        config.warmupFramesPerGpu = 1;
        DummyGraph graph(NumaExecutor(location.numaNode), config, sink, cancellation);
        test.expect(graph.initialize(), "initialize one-task two-worker graph");
        PhaseGate beforeStartGate;
        test.expect(!graph.startPhase(FramePhase::Warmup, beforeStartGate), "reject frame submission before graph start");
        beforeStartGate.release();
        AlgoParams loadedParameters = config.parameters;
        loadedParameters.blob = {4, 5, 6};
        test.expect(graph.changeParameters(loadedParameters), "notify changed parameters while graph tasks are loaded and quiescent");
        test.expect(graph.start(), "start one graph execution cycle");
        test.expect(graph.changeParameters(loadedParameters), "same graph parameters require no task notification");
        PhaseGate invalidPhaseGate;
        test.expect(!graph.startPhase(static_cast<FramePhase>(99), invalidPhaseGate), "reject an invalid graph-owned frame phase");
        invalidPhaseGate.release();

        PhaseGate warmupGate;
        test.expect(graph.startPhase(FramePhase::Warmup, warmupGate), "submit warmup inside the started execution cycle");
        AlgoParams changedParameters = loadedParameters;
        changedParameters.name = "graph-test-updated";
        test.expect(!graph.changeParameters(changedParameters), "reject parameter changes while a phase is active");
        test.expect(!graph.stop(), "reject graph stop while a phase is active");
        warmupGate.release();
        test.expect(graph.waitForPhase(), "finish warmup before changing parameters");
        test.expect(graph.changeParameters(changedParameters), "notify changed parameters at a quiescent phase boundary");

        test.expect(runGraphPhase(graph, FramePhase::Timed), "run one-task two-worker graph");
        PhaseGate repeatedPhaseGate;
        test.expect(!graph.startPhase(FramePhase::Timed, repeatedPhaseGate), "graph-owned phase state rejects repeated submission");
        repeatedPhaseGate.release();
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "DummyGraph free-task pool prevents concurrent reuse of its sole task instance");
        test.expect(sink.count() == 9 && sink.failureCount() == 0, "one-task graph completes every frame once across both phases");
        test.expect(graph.stop(), "stop graph after all frame executions finish");
        AlgoParams stoppedParameters = changedParameters;
        stoppedParameters.name = "graph-test-stopped";
        test.expect(graph.changeParameters(stoppedParameters), "notify changed parameters after the execution cycle stops and before unload");
        test.expect(!graph.stop(), "reject repeated graph stop");
        test.expect(graph.shutdown(), "shutdown one-task graph");
    }
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        DummyGraph graph(NumaExecutor(location.numaNode), makeGraphConfig(2, 1, ExecutionModel::Interleaved), sink, cancellation);
        test.expect(graph.initialize(), "initialize two-task one-worker graph");
        test.expect(graph.start(), "start two-task one-worker graph");
        test.expect(runGraphPhase(graph, FramePhase::Timed), "run two-task one-worker graph");
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "worker pool limits concurrency independently");
        test.expect(sink.count() == 8 && sink.failureCount() == 0, "one-worker graph completes every frame once");
        test.expect(graph.stop(), "stop two-task one-worker graph");
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
        config.taskInstancesPerGpu = 1;
        config.graphThreads = entry.second.size();
        config.timedFramesPerGpu = 1;
        config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
        config.parameters.name = "numa-test";
        graphs.push_back(std::make_unique<DummyGraph>(NumaExecutor(entry.first), config, sink, cancellation));
    }
    bool ok = true;
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->initialize() && ok;
    }
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->start() && ok;
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
        ok = graph->stop() && ok;
    }
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->shutdown() && ok;
    }
    test.expect(ok && sink.count() == locations.size(), "one graph copy runs on each GPU-bearing NUMA node");
}

void testGraphCancellation(TestContext& test, const GpuLocation& location) {
    GraphSink sink;
    std::atomic<bool> cancellation{false};
    GraphConfig config = makeGraphConfig(1, 2, static_cast<ExecutionModel>(99));
    config.warmupFramesPerGpu = 2;
    config.timedFramesPerGpu = 4;
    DummyGraph graph(NumaExecutor(location.numaNode), config, sink, cancellation);
    test.expect(graph.initialize(), "initialize graph used for failure propagation");
    test.expect(graph.start(), "start graph used for failure propagation");
    test.expect(!runGraphPhase(graph, FramePhase::Warmup), "execution failure fails graph phase");
    test.expect(cancellation.load(std::memory_order_acquire), "execution failure raises global cancellation");
    test.expect(sink.count() == 2 && sink.failureCount() == 2, "active failed phase gives every frame one terminal result");
    test.expect(graph.stop(), "failed graph still stops after in-flight execution ends");
    test.expect(graph.shutdown(), "failed graph still unloads cleanly");
    test.expect(sink.count() == 6 && sink.failureCount() == 6, "shutdown cancels every preallocated future frame exactly once");
}

}  // namespace gpuinfra_tests
