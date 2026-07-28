#include "Cel.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "ImageSizing.h"
#include "ThreadSlot.h"

namespace {

constexpr unsigned int CEL_BLOCK_DIM = 16;

// Square the top-left nx-by-nx region of the row-major input frame.
__global__ void celMatrixMultiplicationKernel(const std::uint8_t* d_input, std::uint32_t* d_outputMatrix, int nx, int inputStride) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= nx) return;

    const std::size_t inputRowOffset = static_cast<std::size_t>(y) * inputStride;
    const std::size_t outputRowOffset = static_cast<std::size_t>(y) * nx;
    std::uint32_t sum = 0;

    for (int r = 0; r < 3; ++r) {
        sum = 0;
        for (int k = 0; k < nx; ++k) {
            const std::uint32_t a = d_input[inputRowOffset + k];
            const std::uint32_t b = d_input[static_cast<std::size_t>(k) * inputStride + x];
            sum += a * b;
        }
        d_outputMatrix[outputRowOffset + x] = sum;
    }
}

}  // namespace

struct Cel::Impl {
    int gpuId = -1;
    int numaNode = -1;
    int threadId = -1;
    int inputStride = 0;
    int matrixSize = 0;
    std::size_t inBytes = 0;
    std::size_t matrixBytes = 0;
    std::uint32_t* d_outputMatrix = nullptr;
    std::uint32_t* h_outputMatrix = nullptr;
    bool staticReady = false;
    bool runtimeReady = false;
    bool outputReady = false;
    AlgoParams params;
};

Cel::Cel() : impl(new Impl()) {}

Cel::~Cel() {
    close();
    delete impl;
    impl = nullptr;
}

bool Cel::initStatic(const AlgoStaticInfo& info) {
    if (impl == nullptr || info.gpuId < 0 || info.numaNode < 0) {
        return false;
    }
    impl->gpuId = info.gpuId;
    impl->numaNode = info.numaNode;
    impl->staticReady = true;
    return true;
}

bool Cel::configure(const AlgoRuntimeInfo& info) {
    if (impl == nullptr || !impl->staticReady || !ImageSizing::isValidFactor(info.sizeFactor)) {
        return false;
    }
    const int matrixSize = ImageSizing::scaledDimension(info.sizeFactor, ImageSizing::CEL_MULTIPLIER);
    if (info.frameW < matrixSize || info.frameH < matrixSize) {
        return false;
    }
    const std::size_t frameBytes = static_cast<std::size_t>(info.frameW) * info.frameH;
    if (info.inBytes < frameBytes) {
        return false;
    }

    if (!close()) {
        return false;
    }
    impl->inputStride = info.frameW;
    impl->matrixSize = matrixSize;
    impl->inBytes = info.inBytes;
    impl->matrixBytes = ImageSizing::squareBytes(matrixSize, sizeof(std::uint32_t));
    impl->params = info.params;
    impl->runtimeReady = true;
    return true;
}

bool Cel::allocateOutputBuffers(const ThreadSlot& slot) {
    if (impl == nullptr || !impl->runtimeReady || impl->outputReady || slot.threadId < 0 || slot.gpuId != impl->gpuId || slot.numaNode != impl->numaNode) {
        return false;
    }

    CUDA_CHECK(cudaSetDevice(impl->gpuId), return false);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->d_outputMatrix), impl->matrixBytes), return false);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&impl->h_outputMatrix), impl->matrixBytes, cudaHostAllocPortable), {
        CUDA_CHECK(cudaFree(impl->d_outputMatrix), );
        impl->d_outputMatrix = nullptr;
        return false;
    });

    impl->threadId = slot.threadId;
    impl->outputReady = true;
    return true;
}

std::size_t Cel::scratchBytesNeeded() const {
    return 0;
}

bool Cel::launchKernels(const ThreadSlot& slot, cudaStream_t stream) {
    if (impl == nullptr || !impl->runtimeReady || !impl->outputReady || slot.threadId != impl->threadId || slot.gpuId != impl->gpuId || slot.d_in == nullptr || stream == nullptr || stream != slot.stream) {
        return false;
    }

    const auto* d_input = static_cast<const std::uint8_t*>(slot.d_in);
    dim3 block(CEL_BLOCK_DIM, CEL_BLOCK_DIM);
    dim3 grid((impl->matrixSize + block.x - 1U) / block.x, (impl->matrixSize + block.y - 1U) / block.y);
    celMatrixMultiplicationKernel << <grid, block, 0, stream >> > (d_input, impl->d_outputMatrix, impl->matrixSize, impl->inputStride);
    CUDA_CHECK(cudaGetLastError(), return false);
    return true;
}

bool Cel::launchD2H(const ThreadSlot& slot, cudaStream_t stream) {
    if (impl == nullptr || !impl->runtimeReady || !impl->outputReady || slot.threadId != impl->threadId || slot.gpuId != impl->gpuId || stream == nullptr || stream != slot.stream) {
        return false;
    }

    CUDA_CHECK(cudaMemcpyAsync(impl->h_outputMatrix, impl->d_outputMatrix, impl->matrixBytes, cudaMemcpyDeviceToHost, stream), return false);
    return true;
}

bool Cel::prepareOutput(AlgoOutput& output) const {
    if (impl == nullptr || !impl->runtimeReady || !impl->outputReady) {
        return false;
    }

    output.algoName = "cel";
    output.width = impl->matrixSize;
    output.height = impl->matrixSize;
    output.data.resize(impl->matrixBytes);
    return true;
}

bool Cel::collectResult(const ThreadSlot& slot, AlgoOutput& output) const {
    if (impl == nullptr || !impl->runtimeReady || !impl->outputReady || slot.threadId != impl->threadId || slot.gpuId != impl->gpuId) {
        return false;
    }

    if (output.data.size() != impl->matrixBytes) {
        return false;
    }

    std::memcpy(output.data.data(), impl->h_outputMatrix, impl->matrixBytes);
    return true;
}

bool Cel::close() {
    if (impl == nullptr) {
        return true;
    }
    bool ok = true;
    if (impl->d_outputMatrix != nullptr || impl->h_outputMatrix != nullptr) {
        CUDA_CHECK(cudaSetDevice(impl->gpuId), return false);
    }
    if (impl->d_outputMatrix != nullptr) {
        CUDA_CHECK(cudaFree(impl->d_outputMatrix), ok = false);
        impl->d_outputMatrix = nullptr;
    }
    if (impl->h_outputMatrix != nullptr) {
        CUDA_CHECK(cudaFreeHost(impl->h_outputMatrix), ok = false);
        impl->h_outputMatrix = nullptr;
    }
    impl->threadId = -1;
    impl->inputStride = 0;
    impl->matrixSize = 0;
    impl->inBytes = 0;
    impl->matrixBytes = 0;
    impl->runtimeReady = false;
    impl->outputReady = false;
    return ok;
}
