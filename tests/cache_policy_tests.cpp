#include "test_support.h"

#include <utility>

namespace gpuinfra_tests {
namespace {

constexpr std::size_t POLICY_PAYLOAD_BYTES = 64;

FrameMetadata makePolicyMetadata(std::uint64_t dataId) {
    FrameMetadata metadata;
    metadata.key.frameId = dataId;
    metadata.width = 8;
    metadata.height = 8;
    metadata.dtype = 1;
    metadata.bytes = POLICY_PAYLOAD_BYTES;
    return metadata;
}

class CachePolicyResources {
public:
    CachePolicyResources(TestContext& testContext, const GpuLocation& location, std::size_t capacity, CacheEvictionPolicy policy, std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) : test(testContext) {
        initialized = cache.initialize({location.gpuId}, POLICY_PAYLOAD_BYTES, capacity, timeout, policy);
        for (TaskGpuResources& resource : resources) {
            initialized = initializeAccessResources(resource, location, POLICY_PAYLOAD_BYTES) && initialized;
        }
        test.expect(initialized, "initialize cache policy and separate caller streams");
    }

    ~CachePolicyResources() {
        test.expect(cache.release(), "release cache policy after all leases finish");
        for (TaskGpuResources& resource : resources) {
            test.expect(releaseAccessResources(resource), "release cache policy caller resources");
        }
    }

    GpuDataCache cache;
    std::array<TaskGpuResources, 2> resources;
    bool initialized = false;

private:
    TestContext& test;
};

bool fillPolicyPayload(TestContext& test, GpuDataCache& cache, const TaskGpuResources& resources, std::uint64_t dataId, CacheRetention retention = CacheRetention::Keep) {
    GpuDataAccess access = cache.getCacheData(makePolicyMetadata(dataId), makeCacheRequest(resources));
    test.expect(access.status() == CacheStatus::CacheFill, "new policy payload reserves a fill");
    if (access.status() != CacheStatus::CacheFill) {
        return false;
    }
    const bool submitted = cudaMemsetAsync(access.writableData(), static_cast<int>(dataId & 0xffU), POLICY_PAYLOAD_BYTES, access.getStream()) == cudaSuccess;
    const bool finished = access.freeCacheData(submitted, retention);
    test.expect(submitted && finished, "complete policy fill with caller-selected retention");
    return submitted && finished;
}

bool verifyPolicyHit(TestContext& test, GpuDataCache& cache, const TaskGpuResources& resources, std::uint64_t dataId) {
    GpuDataAccess access = cache.getCacheData(makePolicyMetadata(dataId), makeCacheRequest(resources));
    test.expect(access.status() == CacheStatus::CacheHit, "expected survivor is still cached");
    if (access.status() != CacheStatus::CacheHit) {
        return false;
    }
    std::array<unsigned char, POLICY_PAYLOAD_BYTES> output{};
    const bool copied = cudaMemcpyAsync(output.data(), access.data(), POLICY_PAYLOAD_BYTES, cudaMemcpyDeviceToHost, access.getStream()) == cudaSuccess;
    const bool finished = access.freeCacheData(copied);
    const unsigned char expected = static_cast<unsigned char>(dataId & 0xffU);
    const bool matches = std::all_of(output.begin(), output.end(), [expected](unsigned char value) { return value == expected; });
    test.expect(copied && finished && matches, "surviving immutable payload retains every byte");
    return copied && finished && matches;
}

void verifyPolicyMiss(TestContext& test, GpuDataCache& cache, const TaskGpuResources& resources, std::uint64_t dataId) {
    GpuDataAccess access = cache.getCacheData(makePolicyMetadata(dataId), makeCacheRequest(resources));
    test.expect(access.status() == CacheStatus::CacheFill, "expected eviction victim is a miss");
    test.expect(!access.freeCacheData(false), "roll back unpopulated victim probe");
}

}  // namespace

void testCacheEvictionPolicies(TestContext& test, const GpuLocation& location) {
    GpuDataCache invalidPolicy;
    test.expect(!invalidPolicy.initialize({location.gpuId}, POLICY_PAYLOAD_BYTES, 2, std::chrono::milliseconds(0), static_cast<CacheEvictionPolicy>(99)), "reject unknown eviction policy before allocating entries");
    for (CacheEvictionPolicy policy : {CacheEvictionPolicy::LRU, CacheEvictionPolicy::FIFO}) {
        CachePolicyResources fixture(test, location, 2, policy);
        if (!fixture.initialized) {
            continue;
        }
        GpuDataCache& cache = fixture.cache;
        const TaskGpuResources& resources = fixture.resources[0];

        fillPolicyPayload(test, cache, resources, 1);
        fillPolicyPayload(test, cache, resources, 2);
        verifyPolicyHit(test, cache, resources, 1);
        fillPolicyPayload(test, cache, resources, 3);
        test.expect(cache.statisticsSnapshot().eviction == 1, "full cache replaces exactly one eligible entry");
        const std::uint64_t survivor = policy == CacheEvictionPolicy::FIFO ? 2 : 1;
        const std::uint64_t victim = policy == CacheEvictionPolicy::FIFO ? 1 : 2;
        verifyPolicyHit(test, cache, resources, survivor);
        verifyPolicyHit(test, cache, resources, 3);
        verifyPolicyMiss(test, cache, resources, victim);

        test.expect(cache.resetCache(), "reset ordering before busy-entry scenario");
        fillPolicyPayload(test, cache, resources, 1);
        fillPolicyPayload(test, cache, resources, 2);
        {
            GpuDataAccess oldestReader = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
            GpuDataAccess anotherReader = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(fixture.resources[1]));
            test.expect(oldestReader.status() == CacheStatus::CacheHit && anotherReader.status() == CacheStatus::CacheHit, "two readers protect the oldest entry");
            fillPolicyPayload(test, cache, resources, 3);
            test.expect(oldestReader.freeCacheData(true), "release one oldest-entry reader");
            GpuDataAccess thirdReader = cache.getCacheData(makePolicyMetadata(3), makeCacheRequest(resources));
            GpuDataAccess full = cache.getCacheData(makePolicyMetadata(4), makeCacheRequest(resources));
            test.expect(full.status() == CacheStatus::TaskFallback && full.freeCacheData(true), "all live readers remain protected under either policy");
            test.expect(thirdReader.freeCacheData(true) && anotherReader.freeCacheData(true), "release all readers without losing FIFO age");
        }
        fillPolicyPayload(test, cache, resources, 4);
        const std::uint64_t retained = policy == CacheEvictionPolicy::FIFO ? 3 : 1;
        const std::uint64_t removed = policy == CacheEvictionPolicy::FIFO ? 1 : 3;
        verifyPolicyHit(test, cache, resources, retained);
        verifyPolicyHit(test, cache, resources, 4);
        verifyPolicyMiss(test, cache, resources, removed);

        test.expect(cache.resetCache(), "reset before testing publication order");
        {
            GpuDataAccess reservedFirst = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
            test.expect(reservedFirst.status() == CacheStatus::CacheFill, "reserve a producer without publishing it");
            fillPolicyPayload(test, cache, fixture.resources[1], 2);
            const bool submitted = cudaMemsetAsync(reservedFirst.writableData(), 1, POLICY_PAYLOAD_BYTES, resources.stream) == cudaSuccess;
            test.expect(reservedFirst.freeCacheData(submitted), "publish the earlier reservation after the later one");
        }
        fillPolicyPayload(test, cache, resources, 3);
        verifyPolicyHit(test, cache, resources, 1);
        verifyPolicyMiss(test, cache, resources, 2);

        test.expect(cache.resetCache(), "reset removes FIFO links and retains the configured policy");
        fillPolicyPayload(test, cache, resources, 1);
        {
            GpuDataAccess abandoned = cache.getCacheData(makePolicyMetadata(2), makeCacheRequest(resources));
            test.expect(abandoned.status() == CacheStatus::CacheFill, "abandoned reservation never enters completed-fill order");
        }
        fillPolicyPayload(test, cache, resources, 3);
        fillPolicyPayload(test, cache, resources, 4);
        verifyPolicyHit(test, cache, resources, 3);
        verifyPolicyMiss(test, cache, resources, 1);
        test.expect(cache.resetCache() && cache.release(), "clear ordering before reinitialization");
        test.expect(cache.initialize({location.gpuId}, POLICY_PAYLOAD_BYTES, 2, std::chrono::milliseconds(0), policy), "released manager can initialize its policy again");
        fillPolicyPayload(test, cache, resources, 5);
        verifyPolicyHit(test, cache, resources, 5);
    }
}

void testFifoDiscardPositions(TestContext& test, const GpuLocation& location) {
    CachePolicyResources fixture(test, location, 4, CacheEvictionPolicy::FIFO);
    if (!fixture.initialized) {
        return;
    }
    GpuDataCache& cache = fixture.cache;
    const TaskGpuResources& resources = fixture.resources[0];
    for (std::uint64_t dataId : {1U, 2U, 3U, 4U}) {
        fillPolicyPayload(test, cache, resources, dataId);
    }

    // Remove a middle entry with two readers. Keep the following entry alive
    // so that corruption of either FIFO link cannot hide behind eviction.
    {
        GpuDataAccess following = cache.getCacheData(makePolicyMetadata(3), makeCacheRequest(resources));
        GpuDataAccess terminal = cache.getCacheData(makePolicyMetadata(2), makeCacheRequest(resources));
        GpuDataAccess remaining = cache.getCacheData(makePolicyMetadata(2), makeCacheRequest(fixture.resources[1]));
        const void* discardedBuffer = remaining.data();
        test.expect(following.status() == CacheStatus::CacheHit && terminal.status() == CacheStatus::CacheHit && remaining.status() == CacheStatus::CacheHit, "hold readers around a middle FIFO discard");
        test.expect(terminal.freeCacheData(true, CacheRetention::Discard), "mark middle FIFO entry for discard");
        test.expect(cache.statisticsSnapshot().discard == 0, "middle entry remains linked until its last reader completes");
        test.expect(remaining.freeCacheData(true), "last middle reader completes the deferred discard");
        GpuDataAccess replacement = cache.getCacheData(makePolicyMetadata(5), makeCacheRequest(resources));
        test.expect(replacement.status() == CacheStatus::CacheFill && replacement.writableData() == discardedBuffer, "new FIFO tail reuses the discarded middle allocation");
        const bool submitted = replacement.writableData() != nullptr && cudaMemsetAsync(replacement.writableData(), 5, POLICY_PAYLOAD_BYTES, replacement.getStream()) == cudaSuccess;
        test.expect(replacement.freeCacheData(submitted), "publish replacement at FIFO tail");
        test.expect(following.freeCacheData(true), "following entry survives removal of its predecessor");
    }
    for (std::uint64_t dataId : {1U, 3U, 4U, 5U}) {
        verifyPolicyHit(test, cache, resources, dataId);
    }

    // FIFO is now 1,3,4,5. Remove its tail, append a new tail, then remove
    // the head. Hits used to verify survivors must not change these positions.
    {
        GpuDataAccess tail = cache.getCacheData(makePolicyMetadata(5), makeCacheRequest(resources));
        test.expect(tail.status() == CacheStatus::CacheHit && tail.freeCacheData(true, CacheRetention::Discard), "discard FIFO tail while older entries remain");
    }
    fillPolicyPayload(test, cache, resources, 6);
    {
        GpuDataAccess head = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
        test.expect(head.status() == CacheStatus::CacheHit && head.freeCacheData(true, CacheRetention::Discard), "discard FIFO head while newer entries remain");
    }
    fillPolicyPayload(test, cache, resources, 7);
    test.expect(cache.statisticsSnapshot().discard == 3 && cache.statisticsSnapshot().eviction == 0, "middle, tail and head discards create empty slots without eviction");
    for (std::uint64_t dataId : {3U, 4U, 6U, 7U}) {
        verifyPolicyHit(test, cache, resources, dataId);
    }

    // FIFO is 3,4,6,7. Force two consecutive replacements. Survivors catch a
    // reversed/broken link, while probes establish which keys were evicted.
    fillPolicyPayload(test, cache, resources, 8);
    fillPolicyPayload(test, cache, resources, 9);
    test.expect(cache.statisticsSnapshot().eviction == 2, "post-discard FIFO replaces the two oldest surviving entries");
    for (std::uint64_t dataId : {6U, 7U, 8U, 9U}) {
        verifyPolicyHit(test, cache, resources, dataId);
    }
    verifyPolicyMiss(test, cache, resources, 3);
    verifyPolicyMiss(test, cache, resources, 4);
    verifyPolicyMiss(test, cache, resources, 1);
    verifyPolicyMiss(test, cache, resources, 2);
    verifyPolicyMiss(test, cache, resources, 5);
}

void testCacheDiscard(TestContext& test, const GpuLocation& location) {
    for (CacheEvictionPolicy policy : {CacheEvictionPolicy::LRU, CacheEvictionPolicy::FIFO}) {
        CachePolicyResources fixture(test, location, 1, policy);
        if (!fixture.initialized) {
            continue;
        }
        GpuDataCache& cache = fixture.cache;
        const TaskGpuResources& resources = fixture.resources[0];
        fillPolicyPayload(test, cache, resources, 1);
        const void* originalBuffer = nullptr;
        {
            GpuDataAccess terminal = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
            GpuDataAccess outstanding = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(fixture.resources[1]));
            originalBuffer = outstanding.data();
            test.expect(terminal.freeCacheData(true, CacheRetention::Discard), "mark a cached result for discard");
            test.expect(cache.statisticsSnapshot().discard == 0, "deferred discard is counted only after the last reader");
            test.expect(!cache.resetCache() && !cache.release(), "pending discard still protects active readers from reset and release");
            GpuDataAccess fallback = cache.getCacheData(makePolicyMetadata(2), makeCacheRequest(resources));
            test.expect(fallback.status() == CacheStatus::TaskFallback, "pending discard cannot expose its allocation to another key");
            const bool overwritten = cudaMemsetAsync(fallback.writableData(), 0x7f, POLICY_PAYLOAD_BYTES, resources.stream) == cudaSuccess;
            test.expect(fallback.freeCacheData(overwritten, CacheRetention::Discard), "fallback discard only completes caller-owned work");
            test.expect(cache.statisticsSnapshot().discard == 0, "fallback has no entry to count as discarded");
            GpuDataAccess moved = std::move(outstanding);
            std::array<unsigned char, POLICY_PAYLOAD_BYTES> output{};
            const bool copied = cudaMemcpyAsync(output.data(), moved.data(), POLICY_PAYLOAD_BYTES, cudaMemcpyDeviceToHost, moved.getStream()) == cudaSuccess;
            test.expect(moved.freeCacheData(copied, CacheRetention::Keep), "a later Keep completion cannot cancel pending discard");
            test.expect(std::all_of(output.begin(), output.end(), [](unsigned char value) { return value == 1; }), "other stream reads the original bytes until its lease finishes");
        }
        test.expect(cache.statisticsSnapshot().discard == 1, "multiple readers cause exactly one actual discard");
        {
            GpuDataAccess reused = cache.getCacheData(makePolicyMetadata(2), makeCacheRequest(resources));
            test.expect(reused.status() == CacheStatus::CacheFill && reused.writableData() == originalBuffer, "discard keeps the allocation available for a new key");
            const bool submitted = cudaMemsetAsync(reused.writableData(), 2, POLICY_PAYLOAD_BYTES, resources.stream) == cudaSuccess;
            test.expect(reused.freeCacheData(submitted), "publish into the discarded slot");
            GpuDataAccess failed = cache.getCacheData(makePolicyMetadata(2), makeCacheRequest(resources));
            test.expect(!failed.freeCacheData(false, CacheRetention::Discard), "failed consumer reports failure and still discards its final-use data");
        }
        const CacheStatistics discarded = cache.statisticsSnapshot();
        test.expect(discarded.discard == 2 && discarded.eviction == 0, "explicit discards never become capacity evictions");
        CacheStatistics completed;
        test.expect(cache.resetCache(&completed) && completed.discard == 2 && cache.statisticsSnapshot().discard == 0, "reset snapshots and clears the discard interval");

        fillPolicyPayload(test, cache, resources, 1);
        {
            GpuDataAccess outstanding = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
            GpuDataAccess terminal = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(fixture.resources[1]));
            test.expect(!terminal.freeCacheData(false, CacheRetention::Discard), "failed terminal reader sets pending discard");
            test.expect(cache.statisticsSnapshot().discard == 0, "remaining reader defers a failed terminal discard");
            // The remaining reader deliberately exits through RAII cleanup.
        }
        test.expect(cache.statisticsSnapshot().discard == 1, "RAII cleanup of the last reader honors pending discard");
        test.expect(cache.resetCache(), "reset after deferred RAII discard");

        fillPolicyPayload(test, cache, resources, 1, CacheRetention::Discard);
        test.expect(cache.statisticsSnapshot().fillSucceeded == 1 && cache.statisticsSnapshot().discard == 1, "successful fill may finish without retaining its key");
        {
            GpuDataAccess failedFill = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
            test.expect(failedFill.status() == CacheStatus::CacheFill && !failedFill.freeCacheData(false, CacheRetention::Discard), "a discarded fill key is absent and a failed refill rolls back");
        }
        test.expect(cache.statisticsSnapshot().fillFailed == 1 && cache.statisticsSnapshot().discard == 1, "failed fill rollback is not another explicit discard");
        {
            GpuDataAccess producer = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
            GpuDataAccess duplicate = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(fixture.resources[1]));
            test.expect(producer.status() == CacheStatus::CacheFill && duplicate.status() == CacheStatus::TaskFallback, "same-key loading request still uses bounded-wait fallback policy");
            const bool fallbackSubmitted = cudaMemsetAsync(duplicate.writableData(), 1, POLICY_PAYLOAD_BYTES, duplicate.getStream()) == cudaSuccess;
            test.expect(duplicate.freeCacheData(fallbackSubmitted, CacheRetention::Discard), "fallback discard cannot remove another producer's reservation");
            const bool submitted = cudaMemsetAsync(producer.writableData(), 1, POLICY_PAYLOAD_BYTES, producer.getStream()) == cudaSuccess;
            test.expect(producer.freeCacheData(submitted), "original producer publishes after duplicate fallback completes");
        }
        verifyPolicyHit(test, cache, resources, 1);
        {
            GpuDataAccess access = cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
            test.expect(!access.freeCacheData(true, static_cast<CacheRetention>(99)) && access.status() == CacheStatus::CacheHit, "unknown retention is rejected while leaving the lease available for cleanup");
            test.expect(access.freeCacheData(true, CacheRetention::Discard), "valid completion cleans up a rejected retention choice");
        }
        test.expect(cache.statisticsSnapshot().discard == 2 && cache.statisticsSnapshot().eviction == 0, "successful Fill and Hit discards count independently of failed fills and fallback");
    }
}

void testCacheDiscardWaiter(TestContext& test, const GpuLocation& location) {
    for (CacheEvictionPolicy policy : {CacheEvictionPolicy::LRU, CacheEvictionPolicy::FIFO}) {
        CachePolicyResources fixture(test, location, 1, policy, std::chrono::milliseconds(1000));
        if (!fixture.initialized) {
            continue;
        }
        fillPolicyPayload(test, fixture.cache, fixture.resources[0], 1);
        GpuDataAccess reader = fixture.cache.getCacheData(makePolicyMetadata(1), makeCacheRequest(fixture.resources[0]));
        std::promise<CacheStatus> resultPromise;
        std::future<CacheStatus> result = resultPromise.get_future();
        std::thread waiter = NumaExecutor(location.numaNode).start([&fixture, &location, &resultPromise](bool numaReady) {
            if (!numaReady || cudaSetDevice(location.gpuId) != cudaSuccess) {
                resultPromise.set_value(CacheStatus::Invalid);
                return;
            }
            GpuDataAccess access = fixture.cache.getCacheData(makePolicyMetadata(2), makeCacheRequest(fixture.resources[1]));
            const CacheStatus status = access.status();
            const bool submitted = access.writableData() != nullptr && cudaMemsetAsync(access.writableData(), 2, POLICY_PAYLOAD_BYTES, access.getStream()) == cudaSuccess;
            const bool finished = access.freeCacheData(submitted);
            resultPromise.set_value(finished ? status : CacheStatus::Invalid);
        });
        const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (fixture.cache.statisticsSnapshot().waitedRequests == 0 && result.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        test.expect(fixture.cache.statisticsSnapshot().waitedRequests == 1, "new key waits while the sole entry is in use");
        test.expect(reader.freeCacheData(true, CacheRetention::Discard), "terminal completion releases and signals the empty slot");
        test.expect(result.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready, "discard wakes full-cache waiter before its one-second deadline");
        test.expect(result.get() == CacheStatus::CacheFill, "woken waiter receives a fill in the discarded allocation");
        waiter.join();
        const CacheStatistics statistics = fixture.cache.statisticsSnapshot();
        test.expect(statistics.discard == 1 && statistics.eviction == 0 && statistics.fallback == 0 && statistics.fillSucceeded == 2, "discard wakeup avoids both eviction and fallback");
    }
}

void testStaticDataCachePolicy(TestContext& test, const GpuLocation& location) {
    StaticDataConfig config;
    config.runtime.frameW = 8;
    config.runtime.frameH = 8;
    config.runtime.frameDtype = 1;
    config.runtime.inBytes = POLICY_PAYLOAD_BYTES;
    config.gpuCacheEntries = 2;
    config.gpuCacheWaitTimeout = std::chrono::milliseconds(0);
    config.gpuCacheEvictionPolicy = CacheEvictionPolicy::FIFO;
    StaticData owner;
    TaskGpuResources resources;
    const bool initialized = owner.init(config) && initializeAccessResources(resources, location, POLICY_PAYLOAD_BYTES);
    test.expect(initialized, "StaticData forwards the configured FIFO policy");
    if (initialized) {
        for (std::uint64_t dataId : {1U, 2U}) {
            GpuDataAccess access = owner.getCacheData(makePolicyMetadata(dataId), makeCacheRequest(resources));
            const bool submitted = access.status() == CacheStatus::CacheFill && cudaMemsetAsync(access.writableData(), static_cast<int>(dataId), POLICY_PAYLOAD_BYTES, access.getStream()) == cudaSuccess;
            test.expect(access.freeCacheData(submitted), "fill StaticData FIFO entries");
        }
        GpuDataAccess first = owner.getCacheData(makePolicyMetadata(1), makeCacheRequest(resources));
        test.expect(first.status() == CacheStatus::CacheHit && first.freeCacheData(true), "StaticData first-key hit retains FIFO age");
        GpuDataAccess third = owner.getCacheData(makePolicyMetadata(3), makeCacheRequest(resources));
        const bool submitted = third.status() == CacheStatus::CacheFill && cudaMemsetAsync(third.writableData(), 3, POLICY_PAYLOAD_BYTES, third.getStream()) == cudaSuccess;
        test.expect(third.freeCacheData(submitted), "StaticData replaces FIFO victim");
        GpuDataAccess second = owner.getCacheData(makePolicyMetadata(2), makeCacheRequest(resources));
        test.expect(second.status() == CacheStatus::CacheHit && second.freeCacheData(true, CacheRetention::Discard), "StaticData second key survives FIFO and supports Discard");
        test.expect(owner.cacheStatisticsSnapshot().discard == 1, "StaticData forwards discard statistics");
    }
    test.expect(owner.release() && releaseAccessResources(resources), "release StaticData policy test resources");
}

}  // namespace gpuinfra_tests
