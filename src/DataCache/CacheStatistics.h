#pragma once

#include <chrono>
#include <cstdint>

enum class CacheFallbackReason {
    CapacityZero,
    LoadingTimeout,
    FullTimeout,
    LoadingNoWait,
    FullNoWait,
    ReplicaUnavailable
};

struct CacheFallbackStatistics {
    std::uint64_t capacityZero = 0;
    std::uint64_t loadingTimeout = 0;
    std::uint64_t fullTimeout = 0;
    std::uint64_t loadingNoWait = 0;
    std::uint64_t fullNoWait = 0;
    std::uint64_t replicaUnavailable = 0;
};

// Manager-local counters. Snapshot is a copy, protected by the manager mutex.
// Fill counts reservations; completion counters can lag while leases are live.
struct CacheStatistics {
    std::uint64_t hit = 0;
    std::uint64_t fill = 0;
    std::uint64_t fallback = 0;
    std::uint64_t invalid = 0;
    std::uint64_t fillSucceeded = 0;
    std::uint64_t fillFailed = 0;
    std::uint64_t eviction = 0;
    std::uint64_t firstBlockedLoading = 0;
    std::uint64_t firstBlockedFull = 0;
    CacheFallbackStatistics fallbackReasons;
    std::uint64_t waitedRequests = 0;
    std::chrono::nanoseconds totalWaitTime{0};
    std::chrono::nanoseconds maxWaitTime{0};
};
