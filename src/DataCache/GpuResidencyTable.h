#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "DataCache/GpuDataKey.h"

// Fixed-capacity linear-probing table for currently resident/loading cache
// keys. Storage is allocated once; insert/erase never grows the table.
class GpuResidencyTable {
public:
    GpuResidencyTable() = default;

    bool initialize(std::size_t maxResidentEntries);
    void clear();
    void release();
    void swap(GpuResidencyTable& other) noexcept;

    bool find(const GpuDataKey& key, std::size_t& entryIndex) const;
    bool insert(const GpuDataKey& key, std::size_t entryIndex);
    bool erase(const GpuDataKey& key, std::size_t entryIndex);

    bool isInitialized() const;
    std::size_t entryCount() const;
    std::size_t maxEntries() const;
    std::size_t slotCount() const;

    GpuResidencyTable(const GpuResidencyTable&) = delete;
    GpuResidencyTable& operator=(const GpuResidencyTable&) = delete;

private:
    struct Slot {
        GpuDataKey key;
        std::size_t entryIndex = 0;
        bool occupied = false;
    };

    inline static constexpr std::size_t NO_SLOT = std::numeric_limits<std::size_t>::max();

    std::size_t findSlot(const GpuDataKey& key) const;
    std::size_t homeSlot(const GpuDataKey& key) const;
    std::size_t nextSlot(std::size_t index) const;
    std::size_t probeDistance(std::size_t home, std::size_t index) const;
    void eraseSlot(std::size_t index);

    std::vector<Slot> slots;
    std::size_t residentLimit = 0;
    std::size_t residentCount = 0;
    bool initialized = false;
};
