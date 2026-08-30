#include "GpuResidencyTable.h"

#include <limits>
#include <utility>
#include <vector>

bool GpuResidencyTable::initialize(std::size_t maxResidentEntries) {
    if (initialized || !slots.empty() || residentLimit != 0 || residentCount != 0) {
        return false;
    }

    std::size_t requiredSlots = 0;
    if (maxResidentEntries != 0) {
        if (maxResidentEntries > std::numeric_limits<std::size_t>::max() / 2) {
            return false;
        }
        requiredSlots = maxResidentEntries * 2;
    }

    std::size_t newSlotCount = requiredSlots == 0 ? 0 : 1;
    while (newSlotCount < requiredSlots) {
        if (newSlotCount > std::numeric_limits<std::size_t>::max() / 2) {
            return false;
        }
        newSlotCount *= 2;
    }

    std::vector<Slot> newSlots;
    try {
        newSlots.resize(newSlotCount);
    } catch (...) {
        return false;
    }

    slots.swap(newSlots);
    residentLimit = maxResidentEntries;
    residentCount = 0;
    initialized = true;
    return true;
}

void GpuResidencyTable::clear() {
    for (Slot& slot : slots) {
        slot = Slot();
    }
    residentCount = 0;
}

void GpuResidencyTable::release() {
    std::vector<Slot> emptySlots;
    slots.swap(emptySlots);
    residentLimit = 0;
    residentCount = 0;
    initialized = false;
}

void GpuResidencyTable::swap(GpuResidencyTable& other) noexcept {
    slots.swap(other.slots);
    std::swap(residentLimit, other.residentLimit);
    std::swap(residentCount, other.residentCount);
    std::swap(initialized, other.initialized);
}

bool GpuResidencyTable::find(const GpuDataKey& key, std::size_t& entryIndex) const {
    if (!initialized) {
        return false;
    }

    const std::size_t index = findSlot(key);
    if (index == NO_SLOT) {
        return false;
    }
    entryIndex = slots[index].entryIndex;
    return true;
}

bool GpuResidencyTable::insert(const GpuDataKey& key, std::size_t entryIndex) {
    if (!initialized || residentCount >= residentLimit || slots.empty()) {
        return false;
    }

    std::size_t index = homeSlot(key);
    for (std::size_t probe = 0; probe < slots.size(); ++probe) {
        Slot& slot = slots[index];
        if (!slot.occupied) {
            slot.key = key;
            slot.entryIndex = entryIndex;
            slot.occupied = true;
            ++residentCount;
            return true;
        }
        if (slot.key == key) {
            return false;
        }
        index = nextSlot(index);
    }
    return false;
}

bool GpuResidencyTable::erase(const GpuDataKey& key, std::size_t entryIndex) {
    if (!initialized) {
        return false;
    }

    const std::size_t index = findSlot(key);
    if (index == NO_SLOT || slots[index].entryIndex != entryIndex) {
        return false;
    }
    eraseSlot(index);
    --residentCount;
    return true;
}

bool GpuResidencyTable::isInitialized() const {
    return initialized;
}

std::size_t GpuResidencyTable::entryCount() const {
    return residentCount;
}

std::size_t GpuResidencyTable::maxEntries() const {
    return residentLimit;
}

std::size_t GpuResidencyTable::slotCount() const {
    return slots.size();
}

std::size_t GpuResidencyTable::findSlot(const GpuDataKey& key) const {
    if (slots.empty()) {
        return NO_SLOT;
    }

    std::size_t index = homeSlot(key);
    for (std::size_t probe = 0; probe < slots.size(); ++probe) {
        const Slot& slot = slots[index];
        if (!slot.occupied) {
            return NO_SLOT;
        }
        if (slot.key == key) {
            return index;
        }
        index = nextSlot(index);
    }
    return NO_SLOT;
}

std::size_t GpuResidencyTable::homeSlot(const GpuDataKey& key) const {
    return GpuDataKeyHash{}(key) & (slots.size() - 1);
}

std::size_t GpuResidencyTable::nextSlot(std::size_t index) const {
    return (index + 1) & (slots.size() - 1);
}

std::size_t GpuResidencyTable::probeDistance(std::size_t home, std::size_t index) const {
    return index >= home ? index - home : slots.size() - (home - index);
}

void GpuResidencyTable::eraseSlot(std::size_t index) {
    std::size_t emptyIndex = index;
    std::size_t scanIndex = nextSlot(index);
    while (slots[scanIndex].occupied) {
        const std::size_t homeIndex = homeSlot(slots[scanIndex].key);
        if (probeDistance(homeIndex, emptyIndex) < probeDistance(homeIndex, scanIndex)) {
            slots[emptyIndex] = slots[scanIndex];
            emptyIndex = scanIndex;
        }
        scanIndex = nextSlot(scanIndex);
    }
    slots[emptyIndex] = Slot();
}
