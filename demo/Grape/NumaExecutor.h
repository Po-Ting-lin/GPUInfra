#pragma once

#include <functional>
#include <thread>

// Simulation-only framework boundary. Every callback starts with CPU affinity
// restricted to the assigned graph NUMA node; production provides this itself.
class NumaExecutor {
public:
    explicit NumaExecutor(int numaNode);

    bool run(const std::function<bool()>& callback) const;
    std::thread start(const std::function<void(bool)>& callback) const;

private:
    int node;
};
