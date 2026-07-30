#include "Mi.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "ImageSizing.h"
#include "ThreadSlot.h"

namespace {

constexpr unsigned int MI_BLOCK_DIM = 16;

// Square the top-left nx-by-nx region of the row-major input frame.
__global__ void miMatrixMultiplicationKernel(const std::uint8_t* d_input, std::uint32_t* d_outputMatrix, int nx, int inputStride) {
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

Mi::~Mi() {
    close();
}

bool Mi::init(const AlgoRuntimeInfo& info, const ThreadSlot& slot, AlgoOutput& output, std::size_t& scratchBytes) {
    scratchBytes = 0;
    if (slot.threadId < 0 || slot.gpuId < 0 || slot.numaNode < 0 || slot.stream == nullptr || !ImageSizing::isValidFactor(info.sizeFactor)) {
        return false;
    }
    const int outputSize = ImageSizing::scaledDimension(info.sizeFactor, ImageSizing::MI_MULTIPLIER);
    if (info.frameW < outputSize || info.frameH < outputSize) {
        return false;
    }
    const std::size_t frameBytes = static_cast<std::size_t>(info.frameW) * info.frameH;
    if (info.inBytes < frameBytes) {
        return false;
    }

    if (!close()) {
        return false;
    }
    gpuId = slot.gpuId;
    numaNode = slot.numaNode;
    threadId = slot.threadId;
    frameW = info.frameW;
    frameH = info.frameH;
    matrixSize = outputSize;
    inBytes = info.inBytes;
    matrixBytes = ImageSizing::squareBytes(matrixSize, sizeof(std::uint32_t));

    output.algoName = "mi";
    output.width = matrixSize;
    output.height = matrixSize;
    output.data.resize(matrixBytes);

    CUDA_CHECK(cudaSetDevice(gpuId), return false);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_outputMatrix), matrixBytes), {
        close();
        return false;
    });
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_outputMatrix), matrixBytes, cudaHostAllocPortable), {
        close();
        return false;
    });
    std::memset(h_outputMatrix, 0, matrixBytes);
    CUDA_CHECK(cudaMemsetAsync(d_outputMatrix, 0, matrixBytes, slot.stream), {
        close();
        return false;
    });

    initialized = true;
    return true;
}

bool Mi::notifyParameter(const AlgoParams& params) {
    if (!initialized) {
        return false;
    }
    // The demo accepts any name/blob. Real algorithms validate their schema here.
    algoParams = params;
    return true;
}

bool Mi::launchKernels(const ThreadSlot& slot, cudaStream_t stream) {
    if (!initialized || slot.threadId != threadId || slot.gpuId != gpuId || slot.d_in == nullptr || stream == nullptr || stream != slot.stream) {
        return false;
    }

    const auto* d_input = static_cast<const std::uint8_t*>(slot.d_in);
    dim3 block(MI_BLOCK_DIM, MI_BLOCK_DIM);
    dim3 grid((matrixSize + block.x - 1U) / block.x, (matrixSize + block.y - 1U) / block.y);
    miMatrixMultiplicationKernel << <grid, block, 0, stream >> > (d_input, d_outputMatrix, matrixSize, frameW);
    CUDA_CHECK(cudaGetLastError(), return false);
    return true;
}

bool Mi::launchD2H(const ThreadSlot& slot, cudaStream_t stream) {
    if (!initialized || slot.threadId != threadId || slot.gpuId != gpuId || stream == nullptr || stream != slot.stream) {
        return false;
    }

    CUDA_CHECK(cudaMemcpyAsync(h_outputMatrix, d_outputMatrix, matrixBytes, cudaMemcpyDeviceToHost, stream), return false);
    return true;
}

bool Mi::collectResult(const ThreadSlot& slot, AlgoOutput& output) const {
    if (!initialized || slot.threadId != threadId || slot.gpuId != gpuId) {
        return false;
    }

    if (output.data.size() != matrixBytes) {
        return false;
    }

    std::memcpy(output.data.data(), h_outputMatrix, matrixBytes);
    return true;
}

bool Mi::close() {
    bool ok = true;
    if (d_outputMatrix != nullptr || h_outputMatrix != nullptr) {
        CUDA_CHECK(cudaSetDevice(gpuId), return false);
    }
    if (d_outputMatrix != nullptr) {
        CUDA_CHECK(cudaFree(d_outputMatrix), ok = false);
        d_outputMatrix = nullptr;
    }
    if (h_outputMatrix != nullptr) {
        CUDA_CHECK(cudaFreeHost(h_outputMatrix), ok = false);
        h_outputMatrix = nullptr;
    }
    gpuId = -1;
    numaNode = -1;
    threadId = -1;
    frameW = 0;
    frameH = 0;
    matrixSize = 0;
    inBytes = 0;
    matrixBytes = 0;
    initialized = false;
    algoParams = {};
    return ok;
}
