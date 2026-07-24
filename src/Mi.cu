#include "Mi.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

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

struct PerThreadState {
    std::uint32_t* d_outputMatrix = nullptr;
    std::uint32_t* h_outputMatrix = nullptr;
    std::size_t matrixBytes = 0;
};

}  // namespace

struct Mi::Impl {
    int gpuId = -1;
    int numaNode = -1;
    int numThreads = 0;
    int inputStride = 0;
    int matrixSize = 0;
    std::size_t inBytes = 0;
    std::size_t matrixBytes = 0;
    bool staticReady = false;
    bool runtimeReady = false;
    AlgoParams params;
    std::vector<PerThreadState> perThread;
};

Mi::Mi() : impl(new Impl()) {}

Mi::~Mi() {
    close();
    delete impl;
    impl = nullptr;
}

bool Mi::initStatic(const AlgoStaticInfo& info) {
    if (impl == nullptr || info.gpuId < 0 || info.numaNode < 0) {
        return false;
    }
    impl->gpuId = info.gpuId;
    impl->numaNode = info.numaNode;
    impl->staticReady = true;
    return true;
}

bool Mi::configureAndAlloc(const AlgoRuntimeInfo& info) {
    if (impl == nullptr || !impl->staticReady || info.numThreads <= 0 || !ImageSizing::isValidFactor(info.sizeFactor)) {
        return false;
    }
    const int matrixSize = ImageSizing::scaledDimension(info.sizeFactor, ImageSizing::MI_MULTIPLIER);
    if (info.frameW < matrixSize || info.frameH < matrixSize) {
        return false;
    }
    const std::size_t frameBytes = static_cast<std::size_t>(info.frameW) * info.frameH;
    if (info.inBytes < frameBytes) {
        return false;
    }

    close();
    impl->numThreads = info.numThreads;
    impl->inputStride = info.frameW;
    impl->matrixSize = matrixSize;
    impl->inBytes = info.inBytes;
    impl->matrixBytes = ImageSizing::squareBytes(matrixSize, sizeof(std::uint32_t));
    impl->perThread.resize(static_cast<std::size_t>(info.numThreads));

    for (PerThreadState& state : impl->perThread) {
        state.matrixBytes = impl->matrixBytes;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_outputMatrix), state.matrixBytes), {
            close();
            return false;
        });
        // The complete result matrix is copied asynchronously into this pinned buffer.
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&state.h_outputMatrix), state.matrixBytes, cudaHostAllocPortable), {
            close();
            return false;
        });
        std::memset(state.h_outputMatrix, 0, state.matrixBytes);
        CUDA_CHECK(cudaMemset(state.d_outputMatrix, 0, state.matrixBytes), {
            close();
            return false;
        });
    }

    impl->params = info.params;
    impl->runtimeReady = true;
    return true;
}

std::size_t Mi::scratchBytesNeeded() const {
    return 0;
}

bool Mi::launchKernels(const ThreadSlot& slot, cudaStream_t stream) {
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads || slot.d_in == nullptr || stream == nullptr || stream != slot.stream) {
        return false;
    }

    const PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    const auto* d_input = static_cast<const std::uint8_t*>(slot.d_in);
    dim3 block(MI_BLOCK_DIM, MI_BLOCK_DIM);
    dim3 grid((impl->matrixSize + block.x - 1U) / block.x, (impl->matrixSize + block.y - 1U) / block.y);
    miMatrixMultiplicationKernel << <grid, block, 0, stream >> > (d_input, state.d_outputMatrix, impl->matrixSize, impl->inputStride);
    CUDA_CHECK(cudaGetLastError(), return false);
    return true;
}

bool Mi::launchD2H(const ThreadSlot& slot, cudaStream_t stream) {
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads || stream == nullptr || stream != slot.stream) {
        return false;
    }

    const PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    CUDA_CHECK(cudaMemcpyAsync(state.h_outputMatrix, state.d_outputMatrix, state.matrixBytes, cudaMemcpyDeviceToHost, stream), return false);
    return true;
}

bool Mi::prepareOutput(AlgoOutput& output) const {
    if (impl == nullptr || !impl->runtimeReady) {
        return false;
    }

    output.algoName = "mi";
    output.width = impl->matrixSize;
    output.height = impl->matrixSize;
    output.data.resize(impl->matrixBytes);
    return true;
}

bool Mi::collectResult(const ThreadSlot& slot, AlgoOutput& output) const {
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads) {
        return false;
    }

    const PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    if (output.data.size() != state.matrixBytes) {
        return false;
    }

    std::memcpy(output.data.data(), state.h_outputMatrix, state.matrixBytes);
    return true;
}

bool Mi::close() {
    if (impl == nullptr) {
        return true;
    }
    bool ok = true;
    for (PerThreadState& state : impl->perThread) {
        if (state.d_outputMatrix != nullptr) {
            CUDA_CHECK(cudaFree(state.d_outputMatrix), ok = false);
            state.d_outputMatrix = nullptr;
        }
        if (state.h_outputMatrix != nullptr) {
            CUDA_CHECK(cudaFreeHost(state.h_outputMatrix), ok = false);
            state.h_outputMatrix = nullptr;
        }
        state.matrixBytes = 0;
    }
    impl->perThread.clear();
    impl->numThreads = 0;
    impl->inputStride = 0;
    impl->matrixSize = 0;
    impl->inBytes = 0;
    impl->matrixBytes = 0;
    impl->runtimeReady = false;
    return ok;
}
