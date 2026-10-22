#include "demo/demo_test_support.h"

int main(int argc, char* argv[]) {
    using namespace gpuinfra_tests;
    const std::vector<TestCase> tests = {
        pureTest("demo_runtime_consistency", testDemoRuntimeConsistency),
        pureTest("parameter_registry", testParameterRegistry),
        numaTest("lifecycle_batched", [](TestContext& test, const GpuLocation& location) { testLifecycleAndResults(test, location, ExecutionModel::Batched, 1); }),
        numaTest("lifecycle_interleaved", [](TestContext& test, const GpuLocation& location) { testLifecycleAndResults(test, location, ExecutionModel::Interleaved, 2); }),
        numaTest("result_pipeline", testResultPipeline),
        numaTest("frame_data_across_tasks", testFrameDataAcrossTaskInstances),
        numaTest("task_fallback", testTaskFallbackExecution),
        directTest("numa_execution_boundary", testNumaExecutionBoundary),
        directTest("independent_pools", testIndependentPools),
        allLocationsTest("conditional_numa_graphs", testConditionalNumaGraphs),
        directTest("graph_shutdown_gate", testGraphShutdownReleasesPhaseGate),
        directTest("graph_cancellation", testGraphCancellation),
    };
    return runTestSuite("demo", tests, argc, argv);
}
