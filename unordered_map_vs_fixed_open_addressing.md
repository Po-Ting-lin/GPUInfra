# `std::unordered_map` vs. Fixed Open Addressing

## Scope

This comparison uses:

```text
N = 1024 possible incoming frame keys
S = 300 requested CPU residency-index slots
K = 150 persistent GPU cache entries
```

The incoming frame IDs are approximately continuous but may arrive slightly
out of order:

```text
1, 4, 2, 3, 5, 9, 6, 7, 8, 10, 11, 14, 12, ...
```

Both options change only the resident-key index. They retain the same:

```text
persistent GpuCacheEntry[K]
global LRU
Empty / Loading / Valid states
active GpuDataAccess leases
TaskFallback path
resetCache() lifetime
```

Therefore, absent an allocation failure, both options should make the same
cache-hit, cache-fill, fallback, and victim decisions. Both can use all 150 GPU
cache entries.

## Option A: `std::unordered_map`

```cpp
std::vector<std::unique_ptr<GpuCacheEntry>> entries;  // K persistent GPU allocations
std::unordered_map<GpuDataKey, std::size_t, GpuDataKeyHash> residency;
```

Cold-path initialization would set the target load factor before reserving
capacity:

```cpp
residency.max_load_factor(0.5F);
residency.reserve(K);
```

For `K = 150`, this requests enough buckets for approximately 300 slots. The
C++ standard library chooses the actual bucket count, which may be greater than
300 and differs between implementations.

`reserve()` prevents rehashing while the resident count remains at or below K,
but it does not normally preallocate the individual map nodes. A new resident
key can still allocate a CPU node, and erasing an evicted key can free one.

The map cannot replace the persistent entry pool or LRU by itself:

- The map does not choose an inactive LRU victim.
- It does not prevent eviction of Loading or actively leased entries.
- It does not own the graph-lifetime GPU buffers safely.
- Directly erasing a map-owned `GpuCacheEntry` could destroy the entry and call
  `cudaFree()` on the hot path.

The safe role of the map is only:

```text
GpuDataKey -> persistent entryIndex
```

## Option B: Current Fixed Open Addressing

The current [`GpuResidencyTable`](src/DataCache/GpuResidencyTable.cpp) owns a
fixed contiguous slot vector. It allocates the vector during initialization and
does not grow, rehash, allocate, or free CPU nodes during `find()`, `insert()`,
or `erase()`.

The implementation requires a power-of-two slot count for bit-mask indexing:

```text
required slots = 2 x K = 300
actual slots   = nextPowerOfTwo(300) = 512
actual load    = 150 / 512 = approximately 29.3%
```

Colliding keys use linear probing. Erasing a key performs backward-shift
deletion until the end of its occupied collision cluster. This avoids
tombstones but requires custom code and tests.

## Direct Comparison

| Property | `std::unordered_map` index | Fixed open-addressing index |
| --- | --- | --- |
| Average lookup | O(1) | O(1) |
| Worst-case lookup | O(K) | O(K) |
| GPU entries usable | All 150 | All 150 |
| Collision handling | Usually bucket chaining | Linear probing |
| Collision effect | Extra CPU lookup work only | Extra CPU probing only |
| Insert | Usually allocates a CPU node | Writes a preallocated slot |
| Erase | Usually frees a CPU node | Shifts the occupied cluster |
| Rehash | Avoidable with correct reserve | Impossible after initialization |
| Hot-path CPU allocation | Normally yes on cache fill/replacement | No |
| Hot-path CUDA allocation | No, with a separate persistent entry pool | No |
| Memory layout | Scattered nodes and bucket array | Contiguous slot array |
| Cache locality | Usually lower due to pointer chasing | Usually higher |
| Latency predictability | Depends on STL and allocator | More predictable |
| Memory usage | Implementation-dependent | Fixed and directly measurable |
| Implementation effort | Lower | Higher |
| Portability of performance | STL implementation-dependent | Controlled by GPUInfra |
| LRU and lease handling | Still required separately | Still required separately |

## Behavior for N = 1024

Only currently resident or Loading keys appear in either index, so neither
index grows to N. Its logical size remains at most K:

```text
maximum resident mappings = 150
possible keys over the run = 1024
```

If all 1024 keys are first-time arrivals:

```text
first 150 keys -> consume empty cache entries
next 874 keys -> reuse inactive global-LRU entries
```

Typical `std::unordered_map` node activity is then:

```text
150 initial node allocations
874 erase/free plus emplace/allocation replacements
```

The fixed table performs the same logical replacements without CPU memory
allocation or deallocation. Cache hits do not allocate in either option.

The slightly out-of-order but approximately continuous frame IDs do not alter
this comparison. Both indexes hash the complete `(frameId, cameraId)` key.
Continuity would matter only if the cache policy were changed to a direct
`frameId % K` ring, which would introduce conflict behavior that neither of
these globally usable indexes has.

## Collision Difference from the Old Two-Way Cache

For both options here, a hash collision affects only CPU lookup work. A key can
still map to any of the 150 persistent GPU entries.

The old two-way cache was different: multiple distinct keys that mapped to the
same set could use only that set's two ways, even when entries in other sets
were free. Neither `unordered_map + global LRU` nor fixed open addressing has
that capacity-conflict limitation.

## Recommendation

Choose `std::unordered_map` when code simplicity and standard-library
maintainability are more important than deterministic CPU hot-path behavior.
For `K = 150` and approximately 300 incoming frames per second, its allocation
rate may be acceptable, but this should be confirmed by profiling under real
concurrency.

Choose fixed open addressing when the cache must provide predictable latency,
contiguous metadata, and no CPU allocation, free, or rehash on the hot path.

Adding a custom fixed allocator or node-recycling scheme to
`std::unordered_map` can reduce its allocation cost, but it also removes much
of the simplicity advantage over the current fixed table.

The `GpuCacheManager` public API should remain independent of the chosen index,
allowing the implementation to be replaced without changing `DummyTask` or the
graph scheduler.
