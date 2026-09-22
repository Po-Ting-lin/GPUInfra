#pragma once

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "DataCache/GpuDataKey.h"

struct TaskGpuResources;

// Known resource identity, not a query of the driver's current state.
// A snapshot owns no CUDA resource and is valid only for its lexical scope.
struct GpuDiagnosticInfo {
    int gpuId = -1;
    int numaNode = -1;
    int resourceId = -1;
    CUcontext context = nullptr;
    unsigned long long driverContextId = 0;
    bool driverContextIdKnown = false;
    cudaStream_t stream = nullptr;
    GpuDataKey key;
    bool frameKnown = false;
    const char* stage = "unknown";
};

GpuDiagnosticInfo currentGpuDiagnosticInfo();
void formatGpuDiagnosticInfo(char* output, std::size_t capacity);
enum class GpuErrorSource {
    Application,
    CudaRuntime,
    CudaDriver,
    Cufft
};

struct GpuErrorRecord {
    GpuErrorSource source = GpuErrorSource::Application;
    int code = 0; // Used only for CUDA sources; Application has no CUDA code.
    const char* name = "Failure";
    const char* message = "unknown";
    const char* operation = "unknown";
    const char* file = "unknown";
    int line = 0;
};

// One output implementation; explicit identity never changes the current scope.
void logGpuError(const GpuErrorRecord& error, const GpuDiagnosticInfo& info);
void reportGpuFailure(const char* reason, const char* file = "unknown", int line = 0);
void reportGpuFailure(const char* reason, const GpuDiagnosticInfo& info, const char* file = "unknown", int line = 0);
bool reportCudaError(cudaError_t status, const char* call, const char* file, int line);
bool reportCudaError(CUresult status, const char* call, const char* file, int line);
bool reportCudaError(cufftResult status, const char* call, const char* file, int line);

#define GPUINFRA_REPORT_FAILURE(reason) reportGpuFailure((reason), __FILE__, __LINE__)
#define GPUINFRA_REPORT_FAILURE_WITH_INFO(reason, info) reportGpuFailure((reason), (info), __FILE__, __LINE__)


class GpuDiagnosticScope {
public:
    explicit GpuDiagnosticScope(const GpuDiagnosticInfo& info);
    GpuDiagnosticScope(const TaskGpuResources& resources, const char* stage, const GpuDataKey* key = nullptr);
    explicit GpuDiagnosticScope(const char* stage);
    GpuDiagnosticScope(int gpuId, cudaStream_t stream, const GpuDataKey& key, const char* stage);
    ~GpuDiagnosticScope();

    GpuDiagnosticScope(const GpuDiagnosticScope&) = delete;
    GpuDiagnosticScope& operator=(const GpuDiagnosticScope&) = delete;

private:
    GpuDiagnosticInfo previous;
};

// Exactly one evaluation; overload resolution preserves each API's error type.
// Unsupported types (including plain int) are deliberately not accepted.
#define CUDA_CHECK(call, onError) \
    do { \
        if (!reportCudaError((call), #call, __FILE__, __LINE__)) { \
            onError; \
        } \
    } while (false)
