#include "test_support.h"

namespace gpuinfra_tests {

void testUnifiedGpuLogger(TestContext& test) {
    std::FILE* captured = std::tmpfile();
    test.expect(captured != nullptr, "create error log capture");
    if (captured == nullptr) {
        return;
    }
    std::fflush(stderr);
    const int savedStderr = dup(fileno(stderr));
    if (savedStderr < 0 || dup2(fileno(captured), fileno(stderr)) < 0) {
        test.expect(false, "redirect error log for validation");
        if (savedStderr >= 0) {
            close(savedStderr);
        }
        std::fclose(captured);
        return;
    }
    GpuDiagnosticInfo scoped;
    scoped.gpuId = 2;
    scoped.resourceId = 8;
    scoped.stage = "test.scope";
    int calls = 0;
    int failures = 0;
    {
        const GpuDiagnosticScope scope(scoped);
        CUDA_CHECK((++calls, cudaSuccess), ++failures);
        CUDA_CHECK((++calls, CUDA_SUCCESS), ++failures);
        CUDA_CHECK((++calls, cudaErrorInvalidValue), ++failures);
        CUDA_CHECK((++calls, CUDA_ERROR_INVALID_VALUE), ++failures);
        CUDA_CHECK((++calls, CUFFT_SUCCESS), ++failures);
        CUDA_CHECK((++calls, CUFFT_INVALID_VALUE), ++failures);
        CUDA_CHECK((++calls, static_cast<cufftResult>(999)), ++failures);
        GPUINFRA_REPORT_FAILURE("application failure");
        GpuDiagnosticInfo explicitInfo;
        explicitInfo.gpuId = 7;
        explicitInfo.stage = "explicit.caller";
        GPUINFRA_REPORT_FAILURE_WITH_INFO("explicit failure", explicitInfo);
        test.expect(currentGpuDiagnosticInfo().gpuId == 2, "explicit logging preserves scoped identity");
    }
    std::fflush(stderr);
    const bool restored = dup2(savedStderr, fileno(stderr)) >= 0;
    close(savedStderr);
    std::rewind(captured);
    std::string output;
    char chunk[1024];
    while (std::fgets(chunk, sizeof(chunk), captured) != nullptr) {
        output += chunk;
    }
    std::fclose(captured);
    test.expect(restored && calls == 7 && failures == 4, "CUDA macros evaluate once and preserve onError behavior");
    test.expect(std::count(output.begin(), output.end(), '\n') == 6, "one log per failure and none on success");
    test.expect(output.find("[CUDA Runtime]") != std::string::npos && output.find("error=cudaErrorInvalidValue code=1") != std::string::npos, "runtime error preserves CUDA name and code");
    test.expect(output.find("[CUDA Driver]") != std::string::npos && output.find("error=CUDA_ERROR_INVALID_VALUE code=1") != std::string::npos, "driver error preserves CUDA name and code");
    test.expect(output.find("[cuFFT]") != std::string::npos && output.find("error=CUFFT_INVALID_VALUE code=4") != std::string::npos && output.find("error=CUFFT_UNKNOWN_ERROR code=999") != std::string::npos, "cuFFT preserves its own source, names and unknown codes");
    test.expect(output.find("error=Failure code=none message=application failure") != std::string::npos, "application errors do not masquerade as CUDA errors");
    test.expect(output.find(std::string(__FILE__) + ":") != std::string::npos && output.find("bound_gpu=2") != std::string::npos && output.find("bound_gpu=7") != std::string::npos, "unified logger includes source location and explicit or scoped context");
}

void testCufftIntegration(TestContext& test, const GpuLocation& location) {
    const GpuDiagnosticScope diagnosticScope(location.gpuId, nullptr, GpuDataKey(), "test.cufft");
    constexpr int SAMPLE_COUNT = 8;
    std::array<cufftComplex, SAMPLE_COUNT> h_data{};
    h_data[0].x = 1.0f;
    cufftComplex* d_data = nullptr;
    cufftHandle plan = 0;
    bool succeeded = true;
    bool planCreated = false;
    CUDA_CHECK(cudaSetDevice(location.gpuId), succeeded = false);
    if (succeeded) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(h_data)), succeeded = false);
    }
    if (succeeded) {
        CUDA_CHECK(cufftPlan1d(&plan, SAMPLE_COUNT, CUFFT_C2C, 1), succeeded = false);
        planCreated = succeeded;
    }
    if (succeeded) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), sizeof(h_data), cudaMemcpyHostToDevice), succeeded = false);
    }
    if (succeeded) {
        CUDA_CHECK(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), succeeded = false);
    }
    if (succeeded) {
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, sizeof(h_data), cudaMemcpyDeviceToHost), succeeded = false);
    }
    bool correct = succeeded;
    for (const cufftComplex& sample : h_data) {
        correct = correct && sample.x == 1.0f && sample.y == 0.0f;
    }
    test.expect(correct, "cuFFT impulse transform works through the unified CUDA_CHECK API and transitive link");
    if (planCreated) {
        CUDA_CHECK(cufftDestroy(plan), succeeded = false);
    }
    if (d_data != nullptr) {
        CUDA_CHECK(cudaFree(d_data), succeeded = false);
    }
    test.expect(succeeded, "clean up cuFFT plan and device allocation");
}

void testGpuDiagnostics(TestContext& test) {
    char text[512];
    formatGpuDiagnosticInfo(text, sizeof(text));
    test.expect(std::strstr(text, "driver_context_id=unknown") != nullptr && std::strstr(text, "frame=unknown") != nullptr, "unknown diagnostic fields are explicit");
    GpuDiagnosticInfo known;
    known.gpuId = 2;
    known.numaNode = 1;
    known.resourceId = 7;
    known.driverContextId = 42;
    known.driverContextIdKnown = true;
    known.key = {1234, 5};
    known.frameKnown = true;
    known.stage = "test.execute";
    {
        const GpuDiagnosticScope outer(known);
        {
            const GpuDiagnosticScope inner("test.H2D");
            formatGpuDiagnosticInfo(text, sizeof(text));
            test.expect(std::strstr(text, "bound_gpu=2 numa=1 resource=7") != nullptr && std::strstr(text, "driver_context_id=42") != nullptr && std::strstr(text, "frame=1234 camera=5 stage=test.H2D") != nullptr, "nested diagnostic stage retains immutable resource identity");
        }
        test.expect(std::strcmp(currentGpuDiagnosticInfo().stage, "test.execute") == 0, "nested stage restores its caller");
        bool isolated = false;
        std::thread other([&isolated] {
            isolated = currentGpuDiagnosticInfo().gpuId == -1 && !currentGpuDiagnosticInfo().frameKnown;
        });
        other.join();
        test.expect(isolated && currentGpuDiagnosticInfo().gpuId == 2, "diagnostic state does not leak across worker threads");
        {
            const GpuDiagnosticScope otherGpu(3, nullptr, GpuDataKey{55, 6}, "other.cache");
            test.expect(currentGpuDiagnosticInfo().resourceId == -1 && !currentGpuDiagnosticInfo().driverContextIdKnown && currentGpuDiagnosticInfo().gpuId == 3, "cross-GPU access cannot inherit another GPU context identity");
        }
    }
    test.expect(currentGpuDiagnosticInfo().gpuId == -1 && !currentGpuDiagnosticInfo().frameKnown, "returning from task scope clears frame and resource identity");
}

}  // namespace gpuinfra_tests
