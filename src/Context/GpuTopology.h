#pragma once

#include <string>
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
    // Preserve known PCI placement. For unknown placement, infer only an exact
    // singleton from the system-wide node/online list, never thread affinity.
    // Strict mode returns -1 when ambiguous; non-strict mode explicitly uses 0.
    static int resolveGpuNumaNode(int reportedNode, const std::string& onlineNodes, bool requireNuma);
    // Current scope requires exactly one GPU on the requested NUMA node.
    static bool resolveGpuIds(int numaNode, const std::vector<GpuLocation>& locations, std::vector<int>& gpuIds);
};
