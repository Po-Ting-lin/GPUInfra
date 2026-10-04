#include "ResultPipeline/ResultTasks.h"
#include "CudaCheck.h"

namespace result_pipeline {
namespace {

__global__ void transformResultKernel(const unsigned char* input, unsigned char* output, std::size_t bytes) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= bytes) return;
    output[index] = static_cast<unsigned char>(input[index] ^ 0x5aU);
}

}  // namespace

bool runResultKernel(const void* input, void* output, std::size_t bytes, cudaStream_t stream) {
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned int>((bytes + 255) / 256));
    transformResultKernel << <grid, block, 0, stream >> > (static_cast<const unsigned char*>(input), static_cast<unsigned char*>(output), bytes);
    CUDA_CHECK(cudaGetLastError(), return false);
    return true;
}

}  // namespace result_pipeline
