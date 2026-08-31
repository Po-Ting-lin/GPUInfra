#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "GraphTypes.h"
#include "Algo/IAlgo.h"
#include "ParameterRegistry.h"
#include "TaskGpuResources.h"

struct FrameCpuAtom;
class StaticData;

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

    DummyTask(int instanceId, int gpuId, ExecutionModel model, const AlgoRuntimeInfo& runtime);
    ~DummyTask();

    bool load();
    bool registerParameters(ParameterRegistry& registry);
    bool notifyParameters(const ParameterSnapshot& parameters);

    // DummyGraph owns execute/unload serialization. Direct concurrent calls are unsupported.
    bool execute(FrameCpuAtom& atom, StaticData& staticData);
    bool unload();

    int instanceId() const;
    int gpuId() const;
    TaskLifecycle lifecycle() const;

    DummyTask(const DummyTask&) = delete;
    DummyTask& operator=(const DummyTask&) = delete;

private:
    bool releaseResources();

    int id;
    int gpu;
    ExecutionModel executionModel;
    AlgoRuntimeInfo algoRuntime;
    TaskGpuResources resources;
    std::vector<std::unique_ptr<IAlgo>> algorithms;
    TaskLifecycle state = TaskLifecycle::Constructed;
};
