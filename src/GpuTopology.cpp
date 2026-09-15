#include "GpuTopology.h"

#include <cstdio>
#include <limits>

#include <sys/syscall.h>
#include <unistd.h>

int GpuTopology::currentNumaNode() {
    unsigned int node = 0;
    if (syscall(SYS_getcpu, nullptr, &node, nullptr) != 0 || node > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "[GPUInfra] cannot determine calling thread NUMA node\n");
        return -1;
    }
    return static_cast<int>(node);
}

bool GpuTopology::resolveGpuIds(int numaNode, const std::vector<GpuLocation>& locations, std::vector<int>& gpuIds) {
    gpuIds.clear();
    if (numaNode < 0) {
        std::fprintf(stderr, "[GPUInfra] invalid calling thread NUMA node=%d\n", numaNode);
        return false;
    }

    std::size_t count = 0;
    int gpuId = -1;
    for (const GpuLocation& location : locations) {
        if (location.numaNode == numaNode) {
            ++count;
            gpuId = location.gpuId;
        }
    }
    if (count != 1 || gpuId < 0) {
        std::fprintf(stderr, "[GPUInfra] unsupported NUMA GPU topology node=%d gpu_count=%zu; exactly one GPU is required\n", numaNode, count);
        return false;
    }
    gpuIds.push_back(gpuId);
    return true;
}
