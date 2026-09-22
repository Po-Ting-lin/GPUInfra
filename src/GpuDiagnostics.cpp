#include "CudaCheck.h"

#include <cstdio>
#include <unistd.h>

#include "Context/GpuContext.h"
#include "Types/TaskGpuResources.h"

namespace {
thread_local GpuDiagnosticInfo diagnosticInfo;
}

GpuDiagnosticInfo currentGpuDiagnosticInfo() {
    return diagnosticInfo;
}

GpuDiagnosticScope::GpuDiagnosticScope(const GpuDiagnosticInfo& info) : previous(diagnosticInfo) {
    diagnosticInfo = info;
}

GpuDiagnosticScope::GpuDiagnosticScope(const TaskGpuResources& resources, const char* stage, const GpuDataKey* key) : previous(diagnosticInfo) {
    diagnosticInfo = GpuDiagnosticInfo();
    diagnosticInfo.gpuId = resources.gpuId;
    diagnosticInfo.resourceId = resources.resourceId;
    diagnosticInfo.stream = resources.stream;
    diagnosticInfo.stage = stage;
    if (resources.ctx != nullptr) {
        diagnosticInfo.numaNode = resources.ctx->numaNode;
        diagnosticInfo.context = resources.ctx->primaryCtx;
        diagnosticInfo.driverContextId = resources.ctx->driverContextId;
        diagnosticInfo.driverContextIdKnown = resources.ctx->driverContextIdKnown;
    }
    if (key != nullptr) {
        diagnosticInfo.key = *key;
        diagnosticInfo.frameKnown = true;
    }
    else if (previous.gpuId == resources.gpuId && previous.resourceId == resources.resourceId) {
        diagnosticInfo.key = previous.key;
        diagnosticInfo.frameKnown = previous.frameKnown;
    }
}

GpuDiagnosticScope::GpuDiagnosticScope(const char* stage) : previous(diagnosticInfo) {
    diagnosticInfo.stage = stage;
}

GpuDiagnosticScope::GpuDiagnosticScope(int gpuId, cudaStream_t stream, const GpuDataKey& key, const char* stage) : previous(diagnosticInfo) {
    if (diagnosticInfo.gpuId != gpuId) {
        diagnosticInfo = GpuDiagnosticInfo();
    }
    else if (diagnosticInfo.stream != stream) {
        diagnosticInfo.resourceId = -1;
    }
    diagnosticInfo.gpuId = gpuId;
    diagnosticInfo.stream = stream;
    diagnosticInfo.key = key;
    diagnosticInfo.frameKnown = true;
    diagnosticInfo.stage = stage;
}

GpuDiagnosticScope::~GpuDiagnosticScope() {
    diagnosticInfo = previous;
}

static void formatDiagnosticInfo(char* output, std::size_t capacity, const GpuDiagnosticInfo& info) {
    char driverId[32] = "unknown";
    char frame[32] = "unknown";
    char camera[32] = "unknown";
    if (info.driverContextIdKnown) {
        std::snprintf(driverId, sizeof(driverId), "%llu", info.driverContextId);
    }
    if (info.frameKnown) {
        std::snprintf(frame, sizeof(frame), "%llu", static_cast<unsigned long long>(info.key.frameId));
        std::snprintf(camera, sizeof(camera), "%u", static_cast<unsigned int>(info.key.cameraId));
    }
    std::snprintf(output, capacity, "pid=%ld bound_gpu=%d numa=%d resource=%d context_handle=%p driver_context_id=%s stream=%p frame=%s camera=%s stage=%s", static_cast<long>(getpid()), info.gpuId, info.numaNode, info.resourceId, static_cast<void*>(info.context), driverId, static_cast<void*>(info.stream), frame, camera, info.stage == nullptr ? "unknown" : info.stage);
}

void formatGpuDiagnosticInfo(char* output, std::size_t capacity) {
    formatDiagnosticInfo(output, capacity, diagnosticInfo);
}

void logGpuError(const GpuErrorRecord& error, const GpuDiagnosticInfo& info) {
    char details[512];
    formatDiagnosticInfo(details, sizeof(details), info);
    const char* source = "GPUInfra";
    switch (error.source) {
        case GpuErrorSource::Application: break;
        case GpuErrorSource::CudaRuntime: source = "CUDA Runtime"; break;
        case GpuErrorSource::CudaDriver: source = "CUDA Driver"; break;
        case GpuErrorSource::Cufft: source = "cuFFT"; break;
    }
    char code[32] = "none";
    if (error.source != GpuErrorSource::Application) {
        std::snprintf(code, sizeof(code), "%d", error.code);
    }
    std::fprintf(stderr, "[%s] %s:%d operation=%s error=%s code=%s message=%s %s\n", source, error.file == nullptr ? "unknown" : error.file, error.line, error.operation == nullptr ? "unknown" : error.operation, error.name == nullptr ? "unknown" : error.name, code, error.message == nullptr ? "unknown" : error.message, details);
}

void reportGpuFailure(const char* reason, const GpuDiagnosticInfo& info, const char* file, int line) {
    GpuErrorRecord error;
    error.message = reason;
    error.operation = info.stage;
    error.file = file;
    error.line = line;
    logGpuError(error, info);
}

void reportGpuFailure(const char* reason, const char* file, int line) {
    reportGpuFailure(reason, diagnosticInfo, file, line);
}

bool reportCudaError(cudaError_t status, const char* call, const char* file, int line) {
    if (status == cudaSuccess) {
        return true;
    }
    GpuErrorRecord error;
    error.source = GpuErrorSource::CudaRuntime;
    error.code = static_cast<int>(status);
    error.name = cudaGetErrorName(status);
    error.message = cudaGetErrorString(status);
    error.operation = call;
    error.file = file;
    error.line = line;
    logGpuError(error, diagnosticInfo);
    return false;
}

bool reportCudaError(CUresult status, const char* call, const char* file, int line) {
    if (status == CUDA_SUCCESS) {
        return true;
    }
    GpuErrorRecord error;
    error.source = GpuErrorSource::CudaDriver;
    error.code = static_cast<int>(status);
    cuGetErrorName(status, &error.name);
    cuGetErrorString(status, &error.message);
    error.operation = call;
    error.file = file;
    error.line = line;
    logGpuError(error, diagnosticInfo);
    return false;
}

namespace {

const char* cufftErrorName(cufftResult status) {
    switch (status) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        case CUFFT_MISSING_DEPENDENCY: return "CUFFT_MISSING_DEPENDENCY";
#if CUDA_VERSION >= 12000
        case CUFFT_NVRTC_FAILURE: return "CUFFT_NVRTC_FAILURE";
        case CUFFT_NVJITLINK_FAILURE: return "CUFFT_NVJITLINK_FAILURE";
#endif
#if CUFFT_VERSION >= 12000
        case CUFFT_NVSHMEM_FAILURE: return "CUFFT_NVSHMEM_FAILURE";
#endif
    }
    return "CUFFT_UNKNOWN_ERROR";
}

}  // namespace

bool reportCudaError(cufftResult status, const char* call, const char* file, int line) {
    if (status == CUFFT_SUCCESS) {
        return true;
    }
    GpuErrorRecord error;
    error.source = GpuErrorSource::Cufft;
    error.code = static_cast<int>(status);
    error.name = cufftErrorName(status);
    error.message = error.name;
    error.operation = call;
    error.file = file;
    error.line = line;
    logGpuError(error, diagnosticInfo);
    return false;
}
