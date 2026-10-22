#include "core/core_test_support.h"

int main() {
    using namespace gpuinfra_tests;
    TestContext test;
    testGpuDiagnostics(test);
    testUnifiedGpuLogger(test);
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
        testCudaDeviceGuard(test, locations.front());
        testStaticDataValidation(test, locations.front());
        testGpuDataAccessState(test, locations.front());
        testPayloadIdentity(test, locations.front());
        testGpuDataAccessMoves(test, locations.front());
        testIndependentPayloadCaches(test, locations.front(), 0);
        testIndependentPayloadCaches(test, locations.front(), 1);
        testGpuResultCacheHandoff(test, locations.front());
        testGpuResultCacheHandoff(test, locations.front(), CacheEvictionPolicy::FIFO);
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
        testGpuDataCacheLru(test, locations.front());
        testCacheEvictionPolicies(test, locations.front());
        testFifoDiscardPositions(test, locations.front());
        testCacheDiscard(test, locations.front());
        testCacheCompletionFailures(test, locations.front());
        testCacheDiscardWaiter(test, locations.front());
        testStaticDataCachePolicy(test, locations.front());
        return true;
    });
    test.expect(callbacksRan, "all core tests execute inside the NUMA environment");

    GpuContextManager::shutdown();
    if (test.failureCount() != 0) {
        std::cerr << test.failureCount() << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All GPUInfra core tests passed\n";
    return 0;
}
