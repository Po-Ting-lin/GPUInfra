#include "NumaExecutor.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

#include <pthread.h>
#include <sched.h>

namespace {

bool addCpuRange(cpu_set_t& set, const std::string& token) {
    const std::size_t separator = token.find('-');
    try {
        const int first = std::stoi(token.substr(0, separator));
        const int last = separator == std::string::npos ? first : std::stoi(token.substr(separator + 1));
        if (first < 0 || last < first) {
            return false;
        }
        for (int cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu) {
            CPU_SET(cpu, &set);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool nodeCpuSet(int node, cpu_set_t& set) {
    std::ifstream input("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string cpuList;
    if (!(input >> cpuList)) {
        return false;
    }

    CPU_ZERO(&set);
    std::istringstream tokens(cpuList);
    std::string token;
    while (std::getline(tokens, token, ',')) {
        if (!addCpuRange(set, token)) {
            return false;
        }
    }
    return true;
}

bool pinToNumaNode(int node) {
    if (node < 0) {
        return false;
    }

    cpu_set_t nodeSet;
    cpu_set_t allowedSet;
    if (!nodeCpuSet(node, nodeSet) || pthread_getaffinity_np(pthread_self(), sizeof(allowedSet), &allowedSet) != 0) {
        return false;
    }

    // Respect a container or cgroup's existing CPU allowance.
    CPU_AND(&nodeSet, &nodeSet, &allowedSet);
    if (CPU_COUNT(&nodeSet) == 0) {
        return false;
    }
    return pthread_setaffinity_np(pthread_self(), sizeof(nodeSet), &nodeSet) == 0;
}

}  // namespace

NumaExecutor::NumaExecutor(int numaNode) : node(numaNode) {}

std::thread NumaExecutor::start(const std::function<void(bool)>& callback) const {
    return std::thread([numaNode = node, callback] {
        const bool ready = pinToNumaNode(numaNode);
        if (!ready) {
            std::fprintf(stderr, "[GPUInfra] cannot establish framework NUMA affinity node=%d\n", numaNode);
        }
        // A worker must receive failure too, so its startup waiter can finish.
        callback(ready);
    });
}

bool NumaExecutor::run(const std::function<bool()>& callback) const {
    bool succeeded = false;
    try {
        std::thread thread = start([&callback, &succeeded](bool ready) {
            if (!ready) {
                return;
            }
            try {
                succeeded = callback();
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[GPUInfra] framework callback failed: %s\n", error.what());
            }
        });
        thread.join();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[GPUInfra] framework dispatch failed: %s\n", error.what());
    }
    return succeeded;
}
