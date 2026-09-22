#pragma once

#include <vector>

struct GpuLocation {
    int gpuId = -1;
    int numaNode = -1;
};

// Stateless discovery helpers. GpuContextManager owns the authoritative map.
class GpuTopology {
public:
    // The framework guarantees the caller stays on its graph's NUMA node.
    static int currentNumaNode();
    // Current scope requires exactly one GPU on the requested NUMA node.
    static bool resolveGpuIds(int numaNode, const std::vector<GpuLocation>& locations, std::vector<int>& gpuIds);
};
