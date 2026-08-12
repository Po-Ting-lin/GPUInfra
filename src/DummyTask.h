#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "GraphTypes.h"
#include "ParameterRegistry.h"
#include "TaskGpuResources.h"

class IAlgo;

enum class TaskLifecycle {
    Constructed,
    Loaded,
    Registered,
    Notified,
    Failed,
    Unloaded,
};

class DummyTask {
public:
    inline static constexpr const char* NAME_PARAMETER = "dummy.name";
    inline static constexpr const char* BLOB_PARAMETER = "dummy.blob";

    DummyTask(int instanceId, int numaNode, int gpuId, ExecutionModel model, const AlgoRuntimeInfo& runtime);
    ~DummyTask();

    bool load();
    bool registerParameters(ParameterRegistry& registry);
    bool notifyParameters(const ParameterSnapshot& parameters);
    bool execute(FrameSlot& frame);
    bool unload();

    int instanceId() const;
    int gpuId() const;
    int numaNode() const;
    TaskLifecycle lifecycle() const;

    DummyTask(const DummyTask&) = delete;
    DummyTask& operator=(const DummyTask&) = delete;

private:
    bool releaseResources();

    int id;
    int node;
    int gpu;
    ExecutionModel executionModel;
    AlgoRuntimeInfo algoRuntime;
    TaskGpuResources resources;
    std::vector<std::unique_ptr<IAlgo>> algorithms;
    std::atomic<bool> executing{false};
    TaskLifecycle state = TaskLifecycle::Constructed;
};
