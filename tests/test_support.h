#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "DataCache/GpuCacheManager.h"
#include "DataCache/GpuDataAccess.h"
#include "DataCache/GpuResidencyTable.h"
#include "Grape/DummyGraph.h"
#include "DummyTask.h"
#include "Grape/FrameCpuAtom.h"
#include "Context/GpuContextManager.h"
#include "CudaCheck.h"
#include "ImageSizing.h"
#include "Grape/NumaExecutor.h"
#include "ParameterRegistry.h"
#include "StaticData/StaticData.h"
#include "Types/TaskGpuResources.h"

namespace gpuinfra_tests {

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

AlgoRuntimeInfo makeRuntime(int factor);
FrameMetadata makeFrameMetadata(std::uint64_t frameId, const AlgoRuntimeInfo& runtime, std::uint32_t cameraId = 0);
bool loadTask(DummyTask& task, const GpuLocation& location);
bool configureTaskParameters(DummyTask& task, ParameterRegistry& registry);
bool startTask(DummyTask& task, const GpuLocation& location);
bool stopTask(DummyTask& task, const GpuLocation& location);
bool notifyTask(DummyTask& task, const ParameterSnapshot& parameters, const GpuLocation& location);
bool unloadTask(DummyTask& task, const GpuLocation& location);
bool initializeStaticData(StaticData& staticData, const GpuLocation& location, const AlgoRuntimeInfo& runtime, std::size_t gpuCacheEntries = 4);
bool releaseStaticData(StaticData& staticData, const GpuLocation& location);
GpuCacheRequest makeCacheRequest(const TaskGpuResources& resources);
bool initializeAccessResources(TaskGpuResources& resources, const GpuLocation& location, std::size_t bytes);
bool releaseAccessResources(TaskGpuResources& resources);
bool executeTask(DummyTask& task, FrameCpuAtom& atom, StaticData& staticData, const GpuLocation& location);
std::uint32_t referenceValue(const FrameCpuAtom& atom, int dimension, int x, int y, int inputStride);
bool verifyOutput(const FrameCpuAtom& atom, const AlgoOutput& output, int inputStride);
void testUnifiedGpuLogger(TestContext& test);
void testCufftIntegration(TestContext& test, const GpuLocation& location);
void testGpuDiagnostics(TestContext& test);
void testParameterRegistry(TestContext& test);
void testGpuResidencyTable(TestContext& test);
void testStaticDataValidation(TestContext& test, const GpuLocation& location);
void testGpuDataAccessMoves(TestContext& test, const GpuLocation& location);
void testPayloadIdentity(TestContext& test, const GpuLocation& location);
void testGpuDataAccessState(TestContext& test, const GpuLocation& location);
void testIndependentPayloadCaches(TestContext& test, const GpuLocation& location, std::size_t capacity);
void testGpuResultCacheHandoff(TestContext& test, const GpuLocation& location, CacheEvictionPolicy policy = CacheEvictionPolicy::LRU);
void testConcurrentCacheRequests(TestContext& test, const GpuLocation& location);
void testCacheWaitWakeup(TestContext& test, const GpuLocation& location, int scenario);
void testCacheWaitSharedDeadline(TestContext& test, const GpuLocation& location);
void testCacheWaitTimeout(TestContext& test, const GpuLocation& location);
void testCacheStatistics(TestContext& test, const GpuLocation& location);
void testStaticGpuData(TestContext& test, const GpuLocation& location);
void testStaticDataRegions(TestContext& test, const GpuLocation& location);
void testGpuCacheResetBoundaries(TestContext& test, const GpuLocation& location);
void testFrameDataAcrossTaskInstances(TestContext& test, const GpuLocation& location);
void testGpuCacheManagerLru(TestContext& test, const GpuLocation& location);
void testCacheEvictionPolicies(TestContext& test, const GpuLocation& location);
void testCacheDiscard(TestContext& test, const GpuLocation& location);
void testCacheDiscardWaiter(TestContext& test, const GpuLocation& location);
void testStaticDataCachePolicy(TestContext& test, const GpuLocation& location);
void testTaskFallbackExecution(TestContext& test, const GpuLocation& location);
void testLifecycleAndResults(TestContext& test, const GpuLocation& location, ExecutionModel model, int taskId);
bool runGraphPhase(DummyGraph& graph, FramePhase phase);
GraphConfig makeGraphConfig(std::size_t tasks, std::size_t workers, ExecutionModel model);
void testGpuTopologySelection(TestContext& test);
void testNumaExecutionBoundary(TestContext& test, const GpuLocation& location);
void testIndependentPools(TestContext& test, const GpuLocation& location);
void testConditionalNumaGraphs(TestContext& test, const std::vector<GpuLocation>& locations);
void testGraphCancellation(TestContext& test, const GpuLocation& location);

}  // namespace gpuinfra_tests
