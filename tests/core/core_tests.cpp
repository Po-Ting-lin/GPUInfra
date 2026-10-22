#include "core/core_test_support.h"

int main(int argc, char* argv[]) {
    using namespace gpuinfra_tests;
    const std::vector<TestCase> tests = {
        pureTest("gpu_diagnostics", testGpuDiagnostics),
        pureTest("unified_logger", testUnifiedGpuLogger),
        pureTest("residency_table", testGpuResidencyTable),
        pureTest("topology_selection", testGpuTopologySelection),
        numaTest("cuda_device_guard", testCudaDeviceGuard),
        numaTest("static_data_validation", testStaticDataValidation),
        numaTest("data_access_state", testGpuDataAccessState),
        numaTest("payload_identity", testPayloadIdentity),
        numaTest("data_access_moves", testGpuDataAccessMoves),
        numaTest("independent_payload_caches_0", [](TestContext& test, const GpuLocation& location) { testIndependentPayloadCaches(test, location, 0); }),
        numaTest("independent_payload_caches_1", [](TestContext& test, const GpuLocation& location) { testIndependentPayloadCaches(test, location, 1); }),
        numaTest("result_cache_handoff_lru", [](TestContext& test, const GpuLocation& location) { testGpuResultCacheHandoff(test, location); }),
        numaTest("result_cache_handoff_fifo", [](TestContext& test, const GpuLocation& location) { testGpuResultCacheHandoff(test, location, CacheEvictionPolicy::FIFO); }),
        numaTest("concurrent_requests", testConcurrentCacheRequests),
        numaTest("wait_wakeup_0", [](TestContext& test, const GpuLocation& location) { testCacheWaitWakeup(test, location, 0); }),
        numaTest("wait_wakeup_1", [](TestContext& test, const GpuLocation& location) { testCacheWaitWakeup(test, location, 1); }),
        numaTest("wait_wakeup_2", [](TestContext& test, const GpuLocation& location) { testCacheWaitWakeup(test, location, 2); }),
        numaTest("wait_wakeup_3", [](TestContext& test, const GpuLocation& location) { testCacheWaitWakeup(test, location, 3); }),
        numaTest("wait_timeout", testCacheWaitTimeout),
        numaTest("wait_shared_deadline", testCacheWaitSharedDeadline),
        numaTest("cufft_integration", testCufftIntegration),
        numaTest("statistics", testCacheStatistics),
        numaTest("static_gpu_data", testStaticGpuData),
        numaTest("static_data_regions", testStaticDataRegions),
        numaTest("reset_boundaries", testGpuCacheResetBoundaries),
        numaTest("lru", testGpuDataCacheLru),
        numaTest("eviction_policies", testCacheEvictionPolicies),
        numaTest("fifo_discard_positions", testFifoDiscardPositions),
        numaTest("discard", testCacheDiscard),
        numaTest("completion_failures", testCacheCompletionFailures),
        numaTest("discard_waiter", testCacheDiscardWaiter),
        numaTest("static_data_cache_policy", testStaticDataCachePolicy),
    };
    return runTestSuite("core", tests, argc, argv);
}
