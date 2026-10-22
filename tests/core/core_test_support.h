#pragma once

// Test support for the gpuinfra core library. Must not include demo headers.

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

#include "DataCache/GpuDataCache.h"
#include "DataCache/GpuDataAccess.h"
#include "DataCache/GpuResidencyTable.h"
#include "Context/GpuContextManager.h"
#include "CudaCheck.h"
#include "CudaDeviceGuard.h"
#include "Grape/NumaExecutor.h"
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

// How a named test is run by runTestSuite().
//   Pure:   before GPU infrastructure exists; locations is empty.
//   Numa:   inside the first GPU's NUMA environment, like a framework callback.
//   Direct: after initialization on the caller thread; may use every location.
enum class TestKind {
    Pure,
    Numa,
    Direct
};

struct TestCase {
    const char* name;
    TestKind kind;
    std::function<void(TestContext&, const std::vector<GpuLocation>&)> run;
};

TestCase pureTest(const char* name, std::function<void(TestContext&)> test);
TestCase numaTest(const char* name, std::function<void(TestContext&, const GpuLocation&)> test);
TestCase directTest(const char* name, std::function<void(TestContext&, const GpuLocation&)> test);
TestCase allLocationsTest(const char* name, std::function<void(TestContext&, const std::vector<GpuLocation>&)> test);

// No argument runs every test. `--list` prints the names; otherwise each
// argument selects one test by exact name. Returns a process exit code.
int runTestSuite(const char* suiteName, const std::vector<TestCase>& tests, int argc, char* argv[]);

constexpr int TEST_FRAME_FACTOR = 16;

// Frame layout used by tests: a square byte image whose side is 8 * factor. The
// demo tests verify that this matches demo/ImageSizing.h.
AlgoRuntimeInfo makeRuntime(int factor);
FrameMetadata makeFrameMetadata(std::uint64_t frameId, const AlgoRuntimeInfo& runtime, std::uint32_t cameraId = 0);
bool initializeStaticData(StaticData& staticData, const GpuLocation& location, const AlgoRuntimeInfo& runtime, std::size_t gpuCacheEntries = 4);
bool releaseStaticData(StaticData& staticData, const GpuLocation& location);
GpuCacheRequest makeCacheRequest(const TaskGpuResources& resources);
bool initializeAccessResources(TaskGpuResources& resources, const GpuLocation& location, std::size_t bytes);
bool releaseAccessResources(TaskGpuResources& resources);

void testUnifiedGpuLogger(TestContext& test);
void testCufftIntegration(TestContext& test, const GpuLocation& location);
void testGpuDiagnostics(TestContext& test);
void testGpuTopologySelection(TestContext& test);
void testGpuResidencyTable(TestContext& test);
void testCudaDeviceGuard(TestContext& test, const GpuLocation& location);
void testStaticDataValidation(TestContext& test, const GpuLocation& location);
void testStaticGpuData(TestContext& test, const GpuLocation& location);
void testStaticDataRegions(TestContext& test, const GpuLocation& location);
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
void testGpuCacheResetBoundaries(TestContext& test, const GpuLocation& location);
void testGpuDataCacheLru(TestContext& test, const GpuLocation& location);
void testCacheEvictionPolicies(TestContext& test, const GpuLocation& location);
void testFifoDiscardPositions(TestContext& test, const GpuLocation& location);
void testCacheDiscard(TestContext& test, const GpuLocation& location);
void testCacheCompletionFailures(TestContext& test, const GpuLocation& location);
void testCacheDiscardWaiter(TestContext& test, const GpuLocation& location);
void testStaticDataCachePolicy(TestContext& test, const GpuLocation& location);

}  // namespace gpuinfra_tests
