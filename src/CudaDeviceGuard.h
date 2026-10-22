#pragma once

// Cold-path helper: selects a CUDA device for the calling host thread and
// restores the previous current device on destruction, so library calls do
// not change the device a caller thread was using. Check ok() before use.
// Hot-path code keeps using GpuContextManager::makeTaskCurrent().
class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int gpuId);
    ~CudaDeviceGuard();

    bool ok() const;

    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

private:
    int previousDevice = -1;
    bool selected = false;
    bool restorePrevious = false;
};
