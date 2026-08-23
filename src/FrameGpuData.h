#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "FrameGpuAccess.h"

struct TaskGpuResources;

struct FrameGpuReplica {
    int gpuId = -1;
    void* d_data = nullptr;
    std::uint64_t frameId = 0;
    bool valid = false;
};

///////////////////////////////////////////////////////////////////////////////////

// Frame-private GPU buffer state. initialize() allocates once and release() frees
// at graph teardown; Upload replaces payload contents without reallocating.
// The graph protocol owns same-frame serialization; concurrent acquire() is unsupported.
class FrameGpuData {
public:
    FrameGpuData() = default;
    ~FrameGpuData();

    bool initialize(const std::vector<int>& gpuIds, std::size_t bytes);
    FrameGpuAccess acquire(FrameGpuAccessMode mode, std::uint64_t frameId, const TaskGpuResources& resources);
    bool release();

    bool isInitialized() const;
    bool hasData(std::uint64_t frameId) const;
    std::size_t bytes() const;
    std::size_t replicaCount() const;
    std::uint64_t frameId() const;

    FrameGpuData(const FrameGpuData&) = delete;
    FrameGpuData& operator=(const FrameGpuData&) = delete;
    FrameGpuData(FrameGpuData&&) = delete;
    FrameGpuData& operator=(FrameGpuData&&) = delete;

private:
    friend class FrameGpuAccess;

    bool completeAccess(FrameGpuAccess& access, bool succeeded);
    void abortAccess(FrameGpuAccess& access);

    std::vector<FrameGpuReplica> replicas;
    std::size_t dataBytes = 0;
    std::uint64_t residentFrameId = 0;
    bool payloadValid = false;
    bool initialized = false;
};
