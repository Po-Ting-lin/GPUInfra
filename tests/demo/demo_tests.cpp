#include "demo/demo_test_support.h"

int main() {
    using namespace gpuinfra_tests;
    TestContext test;
    testDemoRuntimeConsistency(test);
    testParameterRegistry(test);

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
        testResultPipeline(test, locations.front());
        testFrameDataAcrossTaskInstances(test, locations.front());
        testTaskFallbackExecution(test, locations.front());
        return true;
    });
    test.expect(callbacksRan, "all direct task callbacks execute inside the graph NUMA environment");
    testNumaExecutionBoundary(test, locations.front());
    testIndependentPools(test, locations.front());
    testConditionalNumaGraphs(test, locations);
    testGraphShutdownReleasesPhaseGate(test, locations.front());
    testGraphCancellation(test, locations.front());

    GpuContextManager::shutdown();
    if (test.failureCount() != 0) {
        std::cerr << test.failureCount() << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All GPUInfra demo tests passed\n";
    return 0;
}
