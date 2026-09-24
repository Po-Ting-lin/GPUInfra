#include "Context/GpuTopology.h"

#include <cstdio>
#include <limits>
#include <sstream>

#include <sys/syscall.h>
#include <unistd.h>

namespace {

int singleOnlineNode(const std::string& onlineNodes) {
    std::istringstream input(onlineNodes);
    input >> std::ws;
    if (input.peek() < '0' || input.peek() > '9') {
        return -1;
    }
    int first = -1;
    if (!(input >> first)) {
        return -1;
    }
    if (input.peek() == '-') {
        input.get();
        if (input.peek() < '0' || input.peek() > '9') {
            return -1;
        }
        int last = -1;
        if (!(input >> last) || last != first) {
            return -1;
        }
    }
    input >> std::ws;
    return input.eof() ? first : -1;
}

}  // namespace

int GpuTopology::resolveGpuNumaNode(int reportedNode, const std::string& onlineNodes, bool requireNuma) {
    if (reportedNode >= 0) {
        return reportedNode;
    }
    const int onlyNode = singleOnlineNode(onlineNodes);
    if (onlyNode >= 0) {
        return onlyNode;
    }
    return requireNuma ? -1 : 0;
}

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
