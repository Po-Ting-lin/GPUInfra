#include "Sdd.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "ThreadSlot.h"

namespace {

constexpr std::size_t SDD_OUT_BYTES = 256;
constexpr int SDD_OUT_W = 256;
constexpr int SDD_OUT_H = 1;
constexpr unsigned int SDD_THREADS_PER_BLOCK = 256;
constexpr int SDD_WORK_ROUNDS = 12;

// Synthetic compute kernel used to create a measurable SDD-like GPU stage.
// It intentionally uses a different mixing function and a larger grid than
// CEL so profiling tools show two distinct kernel stages.
__global__ void sddSimulatedLoadKernel(const std::uint8_t* d_input, std::size_t inputBytes, std::uint8_t* d_output) {
    __shared__ std::uint32_t partial[SDD_THREADS_PER_BLOCK];

    const unsigned int lane = threadIdx.x;
    const std::size_t globalThread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + lane;
    const std::size_t gridStride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::uint32_t accumulator = 0x6d2b79f5U + static_cast<std::uint32_t>(globalThread);

    for (std::size_t index = globalThread; index < inputBytes; index += gridStride) {
        std::uint32_t value = static_cast<std::uint32_t>(d_input[index]) | (static_cast<std::uint32_t>(index) << 8U);
#pragma unroll
        for (int round = 0; round < SDD_WORK_ROUNDS; ++round) {
            value += 0x7ed55d16U + (value << 12U);
            value ^= 0xc761c23cU ^ (value >> 19U);
            value += static_cast<std::uint32_t>(round) * 0x165667b1U;
            value ^= value << 5U;
        }
        accumulator += value;
        accumulator ^= accumulator >> 15U;
    }

    partial[lane] = accumulator;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2U; offset > 0U; offset >>= 1U) {
        if (lane < offset) {
            partial[lane] += partial[lane + offset];
            partial[lane] ^= partial[lane] >> 16U;
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

struct Sdd::Impl {
    int gpuId = -1;
    int numaNode = -1;
    int numThreads = 0;
    std::size_t inBytes = 0;
    bool staticReady = false;
    bool runtimeReady = false;
    AlgoParams params;
    std::vector<PerThreadState> perThread;
};

Sdd::Sdd() : impl(new Impl()) {}

Sdd::~Sdd() {
    close();
    delete impl;
    impl = nullptr;
}

bool Sdd::initStatic(const AlgoStaticInfo& info) {
    if (impl == nullptr || info.gpuId < 0 || info.numaNode < 0) {
        return false;
    }
    impl->gpuId = info.gpuId;
    impl->numaNode = info.numaNode;
    impl->staticReady = true;
    return true;
}

bool Sdd::configureAndAlloc(const AlgoRuntimeInfo& info) {
    if (impl == nullptr || !impl->staticReady || info.numThreads <= 0 || info.inBytes == 0) {
        return false;
    }

    close();
    impl->numThreads = info.numThreads;
    impl->inBytes = info.inBytes;
    impl->perThread.resize(static_cast<std::size_t>(info.numThreads));

    for (PerThreadState& state : impl->perThread) {
        state.outBytes = SDD_OUT_BYTES;
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

std::size_t Sdd::scratchBytesNeeded() const {
    return 0;
}

bool Sdd::launchKernels(const ThreadSlot& slot, cudaStream_t stream) {
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads || slot.d_in == nullptr || stream == nullptr || stream != slot.stream) {
        return false;
    }

    PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    const auto* d_input = static_cast<const std::uint8_t*>(slot.d_in);
    dim3 block(SDD_THREADS_PER_BLOCK);
    dim3 grid(static_cast<unsigned int>(state.outBytes));
    sddSimulatedLoadKernel << <grid, block, 0, stream >> > (d_input, impl->inBytes, state.d_out);
    CUDA_CHECK(cudaGetLastError(), return false);
    return true;
}

bool Sdd::launchD2H(const ThreadSlot& slot, cudaStream_t stream) {
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads || stream == nullptr || stream != slot.stream) {
        return false;
    }

    PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    CUDA_CHECK(cudaMemcpyAsync(state.h_out, state.d_out, state.outBytes, cudaMemcpyDeviceToHost, stream), return false);
    return true;
}

AlgoOutput Sdd::collectResult(const ThreadSlot& slot) {
    AlgoOutput output;
    output.algoName = "sdd";
    output.width = SDD_OUT_W;
    output.height = SDD_OUT_H;
    if (impl == nullptr || !impl->runtimeReady || slot.threadId < 0 || slot.threadId >= impl->numThreads) {
        return output;
    }

    const PerThreadState& state = impl->perThread[static_cast<std::size_t>(slot.threadId)];
    output.data.assign(state.h_out, state.h_out + state.outBytes);
    return output;
}

bool Sdd::close() {
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
