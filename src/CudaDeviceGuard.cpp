#include "CudaDeviceGuard.h"

#include <cuda_runtime.h>

#include "CudaCheck.h"

CudaDeviceGuard::CudaDeviceGuard(int gpuId) {
    if (gpuId < 0) {
        return;
    }

    CUDA_CHECK(cudaGetDevice(&previousDevice), return);
    if (previousDevice != gpuId) {
        CUDA_CHECK(cudaSetDevice(gpuId), return);
        restorePrevious = true;
    }
    selected = true;
}

CudaDeviceGuard::~CudaDeviceGuard() {
    if (restorePrevious) {
        CUDA_CHECK(cudaSetDevice(previousDevice), );
    }
}

bool CudaDeviceGuard::ok() const {
    return selected;
}
