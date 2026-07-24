#pragma once

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

inline bool reportCudaError(cudaError_t status, const char* call, const char* file, int line) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "[CUDA Runtime] %s:%d: %s -> %s (%d): %s\n", file, line, call, cudaGetErrorName(status), static_cast<int>(status), cudaGetErrorString(status));
    return false;
}

inline bool reportCuError(CUresult status, const char* call, const char* file, int line) {
    if (status == CUDA_SUCCESS) {
        return true;
    }
    const char* name = "unknown";
    const char* message = "unknown";
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    std::fprintf(stderr, "[CUDA Driver] %s:%d: %s -> %s (%d): %s\n", file, line, call, name, static_cast<int>(status), message);
    return false;
}

#define CUDA_CHECK(call, onError)                                           \
    do {                                                                    \
        if (!reportCudaError((call), #call, __FILE__, __LINE__)) {          \
            onError;                                                        \
        }                                                                   \
    } while (false)

#define CU_CHECK(call, onError)                                             \
    do {                                                                    \
        if (!reportCuError((call), #call, __FILE__, __LINE__)) {            \
            onError;                                                        \
        }                                                                   \
    } while (false)
