#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "FrameMetadata.h"
#include "FrameSlot.h"
#include "GraphTypes.h"
#include "IAlgo.h"

struct StaticFrameConfig {
    FrameMetadata metadata;
    FramePhase phase = FramePhase::Warmup;
};

struct StaticDataConfig {
    int numaNode = -1;
    std::vector<int> gpuIds;
    AlgoRuntimeInfo runtime;
    std::vector<StaticFrameConfig> frames;
};

// Graph-copy scoped owner of the fixed FrameSlot pool. The graph serializes
// scheduling-state transitions; workers may inspect distinct bound slots.
class StaticData {
public:
    inline static constexpr std::size_t FRAME_SLOT_POOL_SIZE = 220;

    StaticData() = default;
    ~StaticData();

    bool init(const StaticDataConfig& config);
    bool execute() const;
    bool release();

    std::size_t frameSlotCount() const;
    std::size_t frameSlotPoolSize() const;
    std::size_t frameCount(FramePhase phase) const;
    FrameSlot* frameSlotAt(std::size_t index);
    const FrameSlot* frameSlotAt(std::size_t index) const;
    FrameSlot* findFrameSlot(const FrameMetadata& metadata);
    const FrameSlot* findFrameSlot(const FrameMetadata& metadata) const;

    bool matchesFrame(const FrameMetadata& metadata, FramePhase phase, FrameState state) const;
    bool preparePhase(FramePhase phase);
    bool beginFrameExecution(const FrameMetadata& metadata);
    bool finishFrameExecution(const FrameMetadata& metadata, bool succeeded, bool cancelled);
    bool cancelFrame(const FrameMetadata& metadata, FrameState expectedState);
    const JobResult* resultFor(const FrameMetadata& metadata) const;

    bool isInitialized() const;

    StaticData(const StaticData&) = delete;
    StaticData& operator=(const StaticData&) = delete;

private:
    std::vector<std::unique_ptr<FrameSlot>> frameSlotPool;
    std::unordered_map<std::uint64_t, std::size_t> frameIdToSlotIndex;
    std::size_t boundFrameCount = 0;
    bool initialized = false;
};
