#include "core/core_test_support.h"

namespace gpuinfra_tests {

void testGpuTopologySelection(TestContext& test) {
    test.expect(GpuTopology::resolveGpuNumaNode(3, "0-3", true) == 3, "preserve known PCI NUMA placement on multi-node hosts");
    test.expect(GpuTopology::resolveGpuNumaNode(3, "", true) == 3, "known PCI placement does not require fallback topology");
    test.expect(GpuTopology::resolveGpuNumaNode(-1, "0\n", true) == 0, "infer node zero only from a confirmed singleton");
    test.expect(GpuTopology::resolveGpuNumaNode(-1, "7\n", true) == 7, "infer actual singleton node without assuming zero");
    test.expect(GpuTopology::resolveGpuNumaNode(-1, " 2-2\n", true) == 2, "accept a singleton range");
    const std::vector<std::string> inconclusive = {"", " \n", "0-1", "0,2", "0-0,2", "-1", "+1", "garbage", "0junk", "0\n1", "2-", "2-1", "999999999999999999999", "0-999999999999999999999"};
    for (const std::string& onlineNodes : inconclusive) {
        test.expect(GpuTopology::resolveGpuNumaNode(-1, onlineNodes, true) == -1, "strict mode rejects ambiguous, unavailable or malformed NUMA topology");
        test.expect(GpuTopology::resolveGpuNumaNode(-1, onlineNodes, false) == 0, "node zero guessing requires explicit non-strict mode");
    }
    test.expect(GpuTopology::resolveGpuNumaNode(-1, "7", false) == 7, "non-strict mode still prefers confirmed singleton placement");
    const std::vector<GpuLocation> locations = {{7, 2}, {3, 0}, {8, 2}, {5, 1}};
    std::vector<int> gpuIds = {99};
    test.expect(!GpuTopology::resolveGpuIds(3, locations, gpuIds) && gpuIds.empty(), "reject a NUMA node with zero GPUs and clear stale results");
    test.expect(!GpuTopology::resolveGpuIds(2, locations, gpuIds) && gpuIds.empty(), "reject multiple GPUs instead of selecting the first");
    test.expect(GpuTopology::resolveGpuIds(1, locations, gpuIds) && gpuIds == std::vector<int>{5}, "resolve the sole local GPU without equating GPU and NUMA IDs");
    test.expect(GpuTopology::resolveGpuIds(0, locations, gpuIds) && gpuIds == std::vector<int>{3}, "ignore GPUs belonging to other NUMA nodes");
    test.expect(!GpuTopology::resolveGpuIds(-1, locations, gpuIds) && gpuIds.empty(), "reject failed NUMA detection");
    test.expect(!GpuTopology::resolveGpuIds(0, {}, gpuIds) && gpuIds.empty(), "reject an empty GPU topology");
    test.expect(!GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds) && gpuIds.empty(), "reject lookup before infrastructure initialization");
}

}  // namespace gpuinfra_tests
