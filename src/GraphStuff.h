#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "DummyTask.h"
#include "GraphTypes.h"
#include "IAlgo.h"
#include "ParameterRegistry.h"

class PhaseGate {
public:
    void wait();
    void release();

private:
    std::mutex lock;
    std::condition_variable condition;
    bool released = false;
};

class GraphSink {
public:
    void deliver(const JobResult& result);
    std::size_t count() const;
    std::size_t failureCount() const;

    GraphSink() = default;
    GraphSink(const GraphSink&) = delete;
    GraphSink& operator=(const GraphSink&) = delete;

private:
    mutable std::mutex lock;
    std::size_t deliveredResults = 0;
    std::size_t failedResults = 0;
};

struct GraphConfig {
    int numaNode = -1;
    std::vector<int> gpuIds;
    std::size_t taskInstancesPerGpu = 4;
    std::size_t graphThreads = 0;
    std::uint64_t warmupFramesPerGpu = 0;
    std::uint64_t timedFramesPerGpu = 0;
    std::uint64_t firstFrameId = 0;
    ExecutionModel executionModel = ExecutionModel::Batched;
    AlgoRuntimeInfo runtime;
    AlgoParams parameters;
};

class DummyGraph {
public:
    DummyGraph(const GraphConfig& graphConfig, GraphSink& outputSink, std::atomic<bool>& globalCancellation);
    ~DummyGraph();

    bool initialize();
    bool startPhase(FramePhase phase, PhaseGate& gate);
    bool waitForPhase();
    bool shutdown();

    int numaNode() const;
    std::size_t taskCount() const;
    std::size_t workerCount() const;
    std::size_t lastMaxConcurrentExecutions() const;

    DummyGraph(const DummyGraph&) = delete;
    DummyGraph& operator=(const DummyGraph&) = delete;

private:
    bool initializeOnNumaNode();
    bool unloadOnNumaNode();
    void workerLoop();
    void cancelReadyFramesLocked();
    void cancelPreparedFramesLocked();
    void finishPhaseIfCompleteLocked();
    std::vector<std::unique_ptr<FrameSlot>>& slotsForPhase(FramePhase phase);

    GraphConfig config;
    GraphSink* sink;
    std::atomic<bool>* cancellation;
    ParameterRegistry parameterRegistry;
    ParameterSnapshot parameterSnapshot;
    std::vector<std::unique_ptr<DummyTask>> tasks;
    std::vector<std::unique_ptr<FrameSlot>> warmupSlots;
    std::vector<std::unique_ptr<FrameSlot>> timedSlots;
    std::vector<std::thread> workers;

    mutable std::mutex schedulerLock;
    std::condition_variable workCondition;
    std::condition_variable phaseCondition;
    std::condition_variable startupCondition;
    std::deque<FrameSlot*> readyFrames;
    std::vector<DummyTask*> freeTasks;
    PhaseGate* phaseGate = nullptr;
    std::size_t readyWorkers = 0;
    std::size_t phaseTotal = 0;
    std::size_t phaseTerminal = 0;
    std::size_t inFlight = 0;
    std::size_t lastMaxInFlight = 0;
    bool workerStartupFailed = false;
    bool phaseActive = false;
    bool phaseSucceeded = true;
    bool initialized = false;
    bool stopping = false;
};
