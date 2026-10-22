#include "core/core_test_support.h"

namespace gpuinfra_tests {

TestCase pureTest(const char* name, std::function<void(TestContext&)> test) {
    return {name, TestKind::Pure, [test](TestContext& context, const std::vector<GpuLocation>&) { test(context); }};
}

TestCase numaTest(const char* name, std::function<void(TestContext&, const GpuLocation&)> test) {
    return {name, TestKind::Numa, [test](TestContext& context, const std::vector<GpuLocation>& locations) { test(context, locations.front()); }};
}

TestCase directTest(const char* name, std::function<void(TestContext&, const GpuLocation&)> test) {
    return {name, TestKind::Direct, [test](TestContext& context, const std::vector<GpuLocation>& locations) { test(context, locations.front()); }};
}

TestCase allLocationsTest(const char* name, std::function<void(TestContext&, const std::vector<GpuLocation>&)> test) {
    return {name, TestKind::Direct, test};
}

int runTestSuite(const char* suiteName, const std::vector<TestCase>& tests, int argc, char* argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--list") {
        for (const TestCase& testCase : tests) {
            std::cout << testCase.name << '\n';
        }
        return 0;
    }

    std::vector<const TestCase*> selected;
    for (int index = 1; index < argc; ++index) {
        const auto match = std::find_if(tests.begin(), tests.end(), [&](const TestCase& testCase) { return std::string(argv[index]) == testCase.name; });
        if (match == tests.end()) {
            std::cerr << "unknown test '" << argv[index] << "'; use --list\n";
            return 2;
        }
        selected.push_back(&*match);
    }
    if (argc == 1) {
        for (const TestCase& testCase : tests) {
            selected.push_back(&testCase);
        }
    }

    TestContext test;
    bool needsGpu = false;
    for (const TestCase* testCase : selected) {
        if (testCase->kind == TestKind::Pure) {
            testCase->run(test, {});
        }
        else {
            needsGpu = true;
        }
    }

    if (needsGpu) {
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

        const bool callbacksRan = NumaExecutor(locations.front().numaNode).run([&] {
            for (const TestCase* testCase : selected) {
                if (testCase->kind == TestKind::Numa) {
                    testCase->run(test, locations);
                }
            }
            return true;
        });
        test.expect(callbacksRan, "all NUMA-environment tests execute");
        for (const TestCase* testCase : selected) {
            if (testCase->kind == TestKind::Direct) {
                testCase->run(test, locations);
            }
        }
        GpuContextManager::shutdown();
    }

    if (test.failureCount() != 0) {
        std::cerr << test.failureCount() << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All GPUInfra " << suiteName << " tests passed\n";
    return 0;
}

AlgoRuntimeInfo makeRuntime(int factor) {
    AlgoRuntimeInfo runtime;
    runtime.sizeFactor = factor;
    runtime.frameW = factor * 8;
    runtime.frameH = runtime.frameW;
    runtime.inBytes = static_cast<std::size_t>(runtime.frameW) * static_cast<std::size_t>(runtime.frameH);
    runtime.frameDtype = 1;
    return runtime;
}

FrameMetadata makeFrameMetadata(std::uint64_t frameId, const AlgoRuntimeInfo& runtime, std::uint32_t cameraId) {
    FrameMetadata metadata;
    metadata.key.frameId = frameId;
    metadata.key.cameraId = cameraId;
    metadata.bytes = runtime.inBytes;
    metadata.width = runtime.frameW;
    metadata.height = runtime.frameH;
    metadata.dtype = runtime.frameDtype;
    return metadata;
}

bool initializeStaticData(StaticData& staticData, const GpuLocation& location, const AlgoRuntimeInfo& runtime, std::size_t gpuCacheEntries) {
    return NumaExecutor(location.numaNode).run([&staticData, &runtime, gpuCacheEntries] {
        StaticDataConfig config;
        config.runtime = runtime;
        config.gpuCacheEntries = gpuCacheEntries;
        return staticData.init(config);
    });
}

bool releaseStaticData(StaticData& staticData, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&staticData] { return staticData.release(); });
}

GpuCacheRequest makeCacheRequest(const TaskGpuResources& resources) {
    GpuCacheRequest request;
    request.gpuId = resources.gpuId;
    request.stream = resources.stream;
    request.d_fallback = resources.d_input;
    request.fallbackBytes = resources.inBytes;
    return request;
}

bool initializeAccessResources(TaskGpuResources& resources, const GpuLocation& location, std::size_t bytes) {
    resources.gpuId = location.gpuId;
    resources.inBytes = bytes;
    if (cudaSetDevice(location.gpuId) != cudaSuccess || cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&resources.d_input, bytes) != cudaSuccess) {
        cudaStreamDestroy(resources.stream);
        resources.stream = nullptr;
        return false;
    }
    return true;
}

bool releaseAccessResources(TaskGpuResources& resources) {
    bool ok = true;
    if (resources.stream != nullptr && cudaStreamSynchronize(resources.stream) != cudaSuccess) {
        ok = false;
    }
    if (resources.d_input != nullptr && cudaFree(resources.d_input) != cudaSuccess) {
        ok = false;
    }
    resources.d_input = nullptr;
    if (resources.stream != nullptr && cudaStreamDestroy(resources.stream) != cudaSuccess) {
        ok = false;
    }
    resources.stream = nullptr;
    resources.inBytes = 0;
    return ok;
}

}  // namespace gpuinfra_tests
