#include "demo/demo_test_support.h"

namespace gpuinfra_tests {

bool loadTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.load(); });
}

// Test bodies run inside NumaExecutor, including registration and error cases.

bool configureTaskParameters(DummyTask& task, ParameterRegistry& registry) {
    return task.registerParameters(registry) && registry.setString(DummyTask::NAME_PARAMETER, "test") && registry.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && registry.seal();
}

bool startTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.start(); });
}

bool stopTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.stop(); });
}

bool notifyTask(DummyTask& task, const ParameterSnapshot& parameters, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task, &parameters] { return task.notifyParameters(parameters); });
}

bool unloadTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.unload(); });
}

bool executeTask(DummyTask& task, FrameCpuAtom& atom, StaticData& staticData, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task, &atom, &staticData] { return staticData.execute() && task.execute(atom, staticData); });
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

bool runGraphPhase(DummyGraph& graph, FramePhase phase) {
    PhaseGate gate;
    if (!graph.startPhase(phase, gate)) {
        gate.release();
        return false;
    }
    gate.release();
    return graph.waitForPhase();
}

GraphConfig makeGraphConfig(std::size_t tasks, std::size_t workers, ExecutionModel model) {
    GraphConfig config;
    config.taskInstancesPerGpu = tasks;
    config.graphThreads = workers;
    config.warmupFramesPerGpu = 0;
    config.timedFramesPerGpu = 8;
    config.executionModel = model;
    config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    config.parameters.name = "graph-test";
    return config;
}

}  // namespace gpuinfra_tests
