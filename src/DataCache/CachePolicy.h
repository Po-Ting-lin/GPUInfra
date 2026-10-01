#pragma once

// Selected once at cache initialization. Both policies skip live leases.
enum class CacheEvictionPolicy {
    LRU,
    FIFO
};

// Per-access completion choice, independent of the eviction policy.
enum class CacheRetention {
    Keep,
    Discard
};
