#include "StaticData.h"

#include <cstdio>
#include <limits>
#include <memory>
#include <unordered_map>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "FrameSlot.h"

namespace {

bool matchesRuntime(const FrameMetadata& metadata, const AlgoRuntimeInfo& runtime) {
    return metadata.bytes != 0 && metadata.bytes == runtime.inBytes && metadata.width == runtime.frameW && metadata.height == runtime.frameH && metadata.dtype == runtime.frameDtype;
}

bool sameMetadata(const FrameMetadata& first, const FrameMetadata& second) {
    return first.id == second.id && first.bytes == second.bytes && first.width == second.width && first.height == second.height && first.dtype == second.dtype;
}

bool validPhase(FramePhase phase) {
    return phase == FramePhase::Warmup || phase == FramePhase::Timed;
}

}  // namespace

StaticData::~StaticData() {
    release();
}

bool StaticData::init(const StaticDataConfig& config) {
    if (initialized || !frameSlotPool.empty() || !frameIdToSlotIndex.empty() || config.numaNode < 0 || config.gpuIds.size() != 1 || config.gpuIds.front() < 0 || config.runtime.inBytes == 0) {
        return false;
    }
    if (config.frames.size() > FRAME_SLOT_POOL_SIZE) {
        std::fprintf(stderr, "[GPUInfra] StaticData frame capacity exceeded configured=%zu capacity=%zu\n", config.frames.size(), FRAME_SLOT_POOL_SIZE);
        return false;
    }

    std::unordered_map<std::uint64_t, std::size_t> newFrameIdToSlotIndex;
    try {
        newFrameIdToSlotIndex.reserve(config.frames.size());
        for (std::size_t index = 0; index < config.frames.size(); ++index) {
            const StaticFrameConfig& frame = config.frames[index];
            if (!validPhase(frame.phase) || !matchesRuntime(frame.metadata, config.runtime)) {
                std::fprintf(stderr, "[GPUInfra] invalid StaticData frame definition index=%zu frame_id=%llu\n", index, static_cast<unsigned long long>(frame.metadata.id));
                return false;
            }
            if (!newFrameIdToSlotIndex.emplace(frame.metadata.id, index).second) {
                std::fprintf(stderr, "[GPUInfra] duplicate StaticData frame ID frame_id=%llu\n", static_cast<unsigned long long>(frame.metadata.id));
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    if (!config.frames.empty() && config.runtime.inBytes > std::numeric_limits<std::size_t>::max() / config.frames.size()) {
        return false;
    }

    const std::size_t frameGpuBytes = config.frames.size() * config.runtime.inBytes;
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    CUDA_CHECK(cudaSetDevice(config.gpuIds.front()), return false);
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes), return false);
    if (frameGpuBytes > freeBytes) {
        std::fprintf(stderr, "[GPUInfra] insufficient StaticData frame GPU memory gpu=%d required=%zu free=%zu total=%zu\n", config.gpuIds.front(), frameGpuBytes, freeBytes, totalBytes);
        return false;
    }

    try {
        frameSlotPool.reserve(FRAME_SLOT_POOL_SIZE);
        for (std::size_t index = 0; index < FRAME_SLOT_POOL_SIZE; ++index) {
            frameSlotPool.push_back(std::make_unique<FrameSlot>());
        }
        for (std::size_t index = 0; index < config.frames.size(); ++index) {
            const StaticFrameConfig& frame = config.frames[index];
            FrameSlot& slot = *frameSlotPool[index];
            if (!slot.bind(frame.metadata, config.numaNode, frame.phase, config.runtime) || !slot.initializeGpuData(config.gpuIds)) {
                release();
                return false;
            }
        }
    } catch (...) {
        release();
        return false;
    }

    frameIdToSlotIndex.swap(newFrameIdToSlotIndex);
    boundFrameCount = config.frames.size();
    initialized = true;
    std::fprintf(stderr, "[GPUInfra] StaticData frame GPU plan gpu=%d pool_slots=%zu bound_frames=%zu bytes_per_frame=%zu allocated_bytes=%zu\n", config.gpuIds.front(), frameSlotPool.size(), boundFrameCount, config.runtime.inBytes, frameGpuBytes);
    return true;
}

bool StaticData::execute() const {
    return initialized;
}

bool StaticData::release() {
    initialized = false;

    bool ok = true;
    bool allReleased = true;
    for (const std::unique_ptr<FrameSlot>& slot : frameSlotPool) {
        if (slot == nullptr) {
            continue;
        }
        if (!slot->releaseGpuData()) {
            ok = false;
        }
        if (slot->deviceData.isInitialized()) {
            allReleased = false;
        }
    }
    if (allReleased) {
        frameSlotPool.clear();
        frameIdToSlotIndex.clear();
        boundFrameCount = 0;
    }
    return ok;
}

std::size_t StaticData::frameSlotCount() const {
    return initialized ? boundFrameCount : 0;
}

std::size_t StaticData::frameSlotPoolSize() const {
    return frameSlotPool.size();
}

std::size_t StaticData::frameCount(FramePhase phase) const {
    if (!initialized || !validPhase(phase)) {
        return 0;
    }

    std::size_t count = 0;
    for (std::size_t index = 0; index < boundFrameCount; ++index) {
        if (frameSlotPool[index]->phase == phase) {
            ++count;
        }
    }
    return count;
}

FrameSlot* StaticData::frameSlotAt(std::size_t index) {
    if (!initialized || index >= boundFrameCount) {
        return nullptr;
    }
    return frameSlotPool[index].get();
}

const FrameSlot* StaticData::frameSlotAt(std::size_t index) const {
    if (!initialized || index >= boundFrameCount) {
        return nullptr;
    }
    return frameSlotPool[index].get();
}

FrameSlot* StaticData::findFrameSlot(const FrameMetadata& metadata) {
    const StaticData* staticData = this;
    return const_cast<FrameSlot*>(staticData->findFrameSlot(metadata));
}

const FrameSlot* StaticData::findFrameSlot(const FrameMetadata& metadata) const {
    if (!initialized) {
        return nullptr;
    }

    const auto match = frameIdToSlotIndex.find(metadata.id);
    if (match == frameIdToSlotIndex.end() || match->second >= boundFrameCount) {
        return nullptr;
    }

    const FrameSlot* slot = frameSlotPool[match->second].get();
    return slot != nullptr && slot->isBound() && sameMetadata(slot->metadata, metadata) ? slot : nullptr;
}

bool StaticData::matchesFrame(const FrameMetadata& metadata, FramePhase phase, FrameState state) const {
    const FrameSlot* slot = findFrameSlot(metadata);
    return slot != nullptr && slot->phase == phase && slot->state == state;
}

bool StaticData::preparePhase(FramePhase phase) {
    if (!initialized || !validPhase(phase)) {
        return false;
    }

    for (std::size_t index = 0; index < boundFrameCount; ++index) {
        const FrameSlot& slot = *frameSlotPool[index];
        if (slot.phase == phase && slot.state != FrameState::Prepared) {
            return false;
        }
    }
    for (std::size_t index = 0; index < boundFrameCount; ++index) {
        FrameSlot& slot = *frameSlotPool[index];
        if (slot.phase != phase) {
            continue;
        }
        slot.state = FrameState::Ready;
        slot.result.id = slot.metadata.id;
        slot.result.ok = true;
    }
    return true;
}

bool StaticData::beginFrameExecution(const FrameMetadata& metadata) {
    FrameSlot* slot = findFrameSlot(metadata);
    if (slot == nullptr || slot->state != FrameState::Ready) {
        return false;
    }

    slot->state = FrameState::Executing;
    return true;
}

bool StaticData::finishFrameExecution(const FrameMetadata& metadata, bool succeeded, bool cancelled) {
    FrameSlot* slot = findFrameSlot(metadata);
    if (slot == nullptr || slot->state != FrameState::Executing) {
        return false;
    }

    if (cancelled) {
        slot->state = succeeded ? FrameState::Cancelled : FrameState::Failed;
        slot->result.ok = false;
    }
    else if (succeeded) {
        slot->state = FrameState::Completed;
    }
    else {
        slot->state = FrameState::Failed;
        slot->result.ok = false;
    }
    return true;
}

bool StaticData::cancelFrame(const FrameMetadata& metadata, FrameState expectedState) {
    FrameSlot* slot = findFrameSlot(metadata);
    if (slot == nullptr || slot->state != expectedState || (expectedState != FrameState::Prepared && expectedState != FrameState::Ready)) {
        return false;
    }

    slot->state = FrameState::Cancelled;
    slot->result.ok = false;
    return true;
}

const JobResult* StaticData::resultFor(const FrameMetadata& metadata) const {
    const FrameSlot* slot = findFrameSlot(metadata);
    return slot == nullptr ? nullptr : &slot->result;
}

bool StaticData::isInitialized() const {
    return initialized;
}
