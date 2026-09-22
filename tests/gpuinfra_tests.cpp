#include "test_support.h"

int main() {
    using namespace gpuinfra_tests;
    TestContext test;
    testGpuDiagnostics(test);
    testUnifiedGpuLogger(test);
    testParameterRegistry(test);
    testGpuResidencyTable(test);
    testGpuTopologySelection(test);

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

    const bool callbacksRan = NumaExecutor(locations.front().numaNode).run([&test, &locations] {
        testLifecycleAndResults(test, locations.front(), ExecutionModel::Batched, 1);
        testLifecycleAndResults(test, locations.front(), ExecutionModel::Interleaved, 2);
        testStaticDataValidation(test, locations.front());
        testGpuDataAccessState(test, locations.front());
        testIndependentPayloadCaches(test, locations.front(), 0);
        testIndependentPayloadCaches(test, locations.front(), 1);
        testConcurrentCacheRequests(test, locations.front());
        for (int scenario = 0; scenario < 4; ++scenario) {
            testCacheWaitWakeup(test, locations.front(), scenario);
        }
        testCacheWaitTimeout(test, locations.front());
        testCacheWaitSharedDeadline(test, locations.front());
        testCufftIntegration(test, locations.front());
        testCacheStatistics(test, locations.front());
        testStaticGpuData(test, locations.front());
        testStaticDataRegions(test, locations.front());
        testGpuCacheResetBoundaries(test, locations.front());
        testGpuCacheManagerLru(test, locations.front());
        testFrameDataAcrossTaskInstances(test, locations.front());
        testTaskFallbackExecution(test, locations.front());
        return true;
    });
    test.expect(callbacksRan, "all direct task callbacks execute inside the graph NUMA environment");
    testNumaExecutionBoundary(test, locations.front());
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
