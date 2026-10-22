#pragma once

// Test support for the demo framework simulation (DummyTask, DummyGraph, result
// pipeline). Core-only helpers live in core/core_test_support.h.

#include "core/core_test_support.h"
#include "Grape/DummyGraph.h"
#include "DummyTask.h"
#include "Grape/FrameCpuAtom.h"
#include "ImageSizing.h"
#include "ParameterRegistry.h"

namespace gpuinfra_tests {

bool loadTask(DummyTask& task, const GpuLocation& location);
bool configureTaskParameters(DummyTask& task, ParameterRegistry& registry);
bool startTask(DummyTask& task, const GpuLocation& location);
bool stopTask(DummyTask& task, const GpuLocation& location);
bool notifyTask(DummyTask& task, const ParameterSnapshot& parameters, const GpuLocation& location);
bool unloadTask(DummyTask& task, const GpuLocation& location);
bool executeTask(DummyTask& task, FrameCpuAtom& atom, StaticData& staticData, const GpuLocation& location);
std::uint32_t referenceValue(const FrameCpuAtom& atom, int dimension, int x, int y, int inputStride);
bool verifyOutput(const FrameCpuAtom& atom, const AlgoOutput& output, int inputStride);
bool runGraphPhase(DummyGraph& graph, FramePhase phase);
GraphConfig makeGraphConfig(std::size_t tasks, std::size_t workers, ExecutionModel model);

void testDemoRuntimeConsistency(TestContext& test);
void testParameterRegistry(TestContext& test);
void testFrameDataAcrossTaskInstances(TestContext& test, const GpuLocation& location);
void testTaskFallbackExecution(TestContext& test, const GpuLocation& location);
void testLifecycleAndResults(TestContext& test, const GpuLocation& location, ExecutionModel model, int taskId);
void testResultPipeline(TestContext& test, const GpuLocation& location);
void testNumaExecutionBoundary(TestContext& test, const GpuLocation& location);
void testIndependentPools(TestContext& test, const GpuLocation& location);
void testConditionalNumaGraphs(TestContext& test, const std::vector<GpuLocation>& locations);
void testGraphShutdownReleasesPhaseGate(TestContext& test, const GpuLocation& location);
void testGraphCancellation(TestContext& test, const GpuLocation& location);

}  // namespace gpuinfra_tests
