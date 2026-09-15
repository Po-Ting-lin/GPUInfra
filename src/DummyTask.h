#pragma once

#include <cstddef>
#include <cstdint>
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
    Registered,
    Loaded,
    Started,
    Stopped,
    Failed,
    Unloaded,
};

class DummyTask {
public:
    inline static constexpr const char* NAME_PARAMETER = "dummy.name";
    inline static constexpr const char* BLOB_PARAMETER = "dummy.blob";

    DummyTask(int instanceId, ExecutionModel model, const AlgoRuntimeInfo& runtime);
    ~DummyTask();

    bool registerParameters(ParameterRegistry& registry);
    bool load();
    bool notifyParameters(const ParameterSnapshot& parameters);
    bool start();

    // DummyGraph owns lifecycle and execute serialization. Direct concurrent
    // calls are unsupported.
    bool execute(FrameCpuAtom& atom, StaticData& staticData);
    bool stop();
    bool unload();

    int instanceId() const;
    int gpuId() const;
    TaskLifecycle lifecycle() const;

    DummyTask(const DummyTask&) = delete;
    DummyTask& operator=(const DummyTask&) = delete;

private:
    bool applyParameters(const ParameterSnapshot& parameters);
    bool releaseResources();

    int id;
    ExecutionModel executionModel;
    AlgoRuntimeInfo algoRuntime;
    TaskGpuResources resources;
    std::vector<std::unique_ptr<IAlgo>> algorithms;

    // Simulates parameter-table access supplied by the real task base class.
    // DummyGraph owns the registry and keeps it alive longer than every task.
    ParameterRegistry* parameterRegistry = nullptr;
    std::uint64_t appliedParameterRevision = 0;
    TaskLifecycle state = TaskLifecycle::Constructed;
};
