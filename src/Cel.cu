#include "Cel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "ThreadSlot.h"

namespace {

constexpr std::size_t CEL_OUT_BYTES = 128;
constexpr int CEL_OUT_W = 128;
constexpr int CEL_OUT_H = 1;
constexpr unsigned int CEL_THREADS_PER_BLOCK = 256;
constexpr int CEL_WORK_ROUNDS = 8;

// Synthetic compute kernel used to create a measurable CEL-like GPU stage.
// The grid collectively reads the entire input. Each block reduces its
// portion to one output byte, so the compiler cannot discard the work.
__global__ void celSimulatedLoadKernel(const std::uint8_t* d_input, std::size_t inputBytes, std::uint8_t* d_output) {
    __shared__ std::uint32_t partial[CEL_THREADS_PER_BLOCK];

    const unsigned int lane = threadIdx.x;
    const std::size_t globalThread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + lane;
    const std::size_t gridStride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::uint32_t accumulator = 2166136261U ^ static_cast<std::uint32_t>(globalThread);

    for (std::size_t index = globalThread; index < inputBytes; index += gridStride) {
        std::uint32_t value = static_cast<std::uint32_t>(d_input[index]) + static_cast<std::uint32_t>(index);
#pragma unroll
        for (int round = 0; round < CEL_WORK_ROUNDS; ++round) {
            value ^= value >> 13U;
            value *= 0x85ebca6bU;
            value ^= value << 7U;
            value += static_cast<std::uint32_t>(round) + 0x9e3779b9U;
        }
        accumulator ^= value;
        accumulator *= 16777619U;
    }

    partial[lane] = accumulator;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
        if (lane < offset) {
            partial[lane] ^= partial[lane + offset];
        }
        __syncthreads();
    }

    if (lane == 0U) {
        const std::uint32_t value = partial[0];
        d_output[blockIdx.x] = static_cast<std::uint8_t>(value ^ (value >> 8U) ^ (value >> 16U) ^ (value >> 24U));
    }
}

struct PerThreadState {
    std::uint8_t* d_out = nullptr;
    std::uint8_t* h_out = nullptr;
    std::size_t outBytes = 0;
};

}  // namespace

struct Cel::Impl {
    int gpuId = -1;
    int numaNode = -1;
    int numThreads = 0;
    std::size_t inBytes = 0;
    bool staticReady = false;
    bool runtimeReady = false;
    AlgoParams params;
    std::vector<PerThreadState> perThread;
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

bool Cel::configureAndAlloc(const AlgoRuntimeInfo& info) {
    if (impl == nullptr || !impl->staticReady || info.numThreads <= 0 || info.inBytes == 0) {
        return false;
    }

    close();
    impl->numThreads = info.numThreads;
    impl->inBytes = info.inBytes;
    impl->perThread.resize(static_cast<std::size_t>(info.numThreads));

    for (PerThreadState& state : impl->perThread) {
        state.outBytes = CEL_OUT_BYTES;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_out), state.outBytes), {
            close();
            return false;
        });
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&state.h_out), state.outBytes, cudaHostAllocPortable), {
            close();
            return false;
        });
        std::memset(state.h_out, 0, state.outBytes);
        CUDA_CHECK(cudaMemset(state.d_out, 0, state.outBytes), {
            close();
            return false;
        });
    }

    impl->params = info.params;
    impl->runtimeReady = true;
    return true;
}

std::size_t Cel::scratchBytesNeeded() const {
    return 0;
}

bool Cel::launchKernels(const ThreadSlot& slot, cudaStream_t stream) {
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads || slot.d_in == nullptr || stream == nullptr || stream != slot.stream) {
        return false;
    }

    PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    const auto* d_input = static_cast<const std::uint8_t*>(slot.d_in);
    dim3 block(CEL_THREADS_PER_BLOCK);
    dim3 grid(static_cast<unsigned int>(state.outBytes));
    celSimulatedLoadKernel << <grid, block, 0, stream >> > (d_input, impl->inBytes, state.d_out);
    CUDA_CHECK(cudaGetLastError(), return false);
    return true;
}

bool Cel::launchD2H(const ThreadSlot& slot, cudaStream_t stream) {
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads || stream == nullptr || stream != slot.stream) {
        return false;
    }

    PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    CUDA_CHECK(cudaMemcpyAsync(state.h_out, state.d_out, state.outBytes, cudaMemcpyDeviceToHost, stream), return false);
    return true;
}

AlgoOutput Cel::collectResult(const ThreadSlot& slot) {
    AlgoOutput output;
    output.algoName = "cel";
    output.width = CEL_OUT_W;
    output.height = CEL_OUT_H;
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads) {
        return output;
    }

    const PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    output.data.assign(state.h_out, state.h_out + state.outBytes);
    return output;
}

bool Cel::close() {
    if (impl == nullptr) {
        return true;
    }
    bool ok = true;
    for (PerThreadState& state : impl->perThread) {
        if (state.d_out != nullptr) {
            CUDA_CHECK(cudaFree(state.d_out), ok = false);
            state.d_out = nullptr;
        }
        if (state.h_out != nullptr) {
            CUDA_CHECK(cudaFreeHost(state.h_out), ok = false);
            state.h_out = nullptr;
        }
        state.outBytes = 0;
    }
    impl->perThread.clear();
    impl->numThreads = 0;
    impl->inBytes = 0;
    impl->runtimeReady = false;
    return ok;
}
