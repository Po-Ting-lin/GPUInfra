#include "test_support.h"

#include <type_traits>
#include <utility>

namespace gpuinfra_tests {

void testGpuResidencyTable(TestContext& test) {
    GpuResidencyTable table;
    test.expect(table.initialize(3), "initialize fixed residency table");
    test.expect(table.maxEntries() == 3 && table.slotCount() >= 6 && table.entryCount() == 0, "fixed residency table reserves at most half load");

    std::vector<GpuDataKey> collidingKeys;
    for (std::uint64_t frameId = 0; frameId < 10000 && collidingKeys.size() < 4; ++frameId) {
        const GpuDataKey key{frameId, 7};
        if ((GpuDataKeyHash{}(key) & (table.slotCount() - 1)) == table.slotCount() - 1) {
            collidingKeys.push_back(key);
        }
    }
    test.expect(collidingKeys.size() == 4, "find colliding keys for backward-shift coverage");
    if (collidingKeys.size() != 4) {
        return;
    }

    test.expect(table.insert(collidingKeys[0], 10) && table.insert(collidingKeys[1], 11) && table.insert(collidingKeys[2], 12), "insert colliding resident keys without growth");
    std::size_t entryIndex = 0;
    test.expect(table.find(collidingKeys[2], entryIndex) && entryIndex == 12, "find the tail of a probe cluster");
    test.expect(table.erase(collidingKeys[0], 10), "erase the head of a probe cluster");
    test.expect(table.find(collidingKeys[1], entryIndex) && entryIndex == 11 && table.find(collidingKeys[2], entryIndex) && entryIndex == 12, "backward-shift deletion preserves later collided keys");
    test.expect(table.insert(collidingKeys[3], 13) && table.entryCount() == 3, "reuse fixed hash storage after erase");
    test.expect(!table.insert(GpuDataKey{10001, 7}, 14), "reject residency beyond cache capacity");
    test.expect(!table.erase(collidingKeys[1], 99), "reject erase with a mismatched cache entry index");
    table.clear();
    test.expect(table.entryCount() == 0 && !table.find(collidingKeys[2], entryIndex), "clear fixed residency without releasing its storage");
    table.release();
    test.expect(!table.isInitialized() && table.slotCount() == 0, "release fixed residency table storage");
}

void testGpuDataAccessState(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    const FrameMetadata firstMetadata = makeFrameMetadata(0, runtime);
    const FrameMetadata secondMetadata = makeFrameMetadata(1, runtime);
    GpuCacheManager cache;
    TaskGpuResources resources;
    const bool initialized = initializeAccessResources(resources, location, runtime.inBytes) && cache.initialize({location.gpuId}, runtime.inBytes, 1);
    test.expect(initialized, "initialize bounded frame GPU cache and fallback buffer");
    if (!initialized) {
        cache.release();
        releaseAccessResources(resources);
        return;
    }

    test.expect(cache.entryCount() == 1 && cache.bytes() == runtime.inBytes, "cache owns one correctly sized preallocated entry");
    void* cacheDeviceData = nullptr;
    {
        GpuDataAccess firstFill = cache.getCacheData(firstMetadata, makeCacheRequest(resources));
        test.expect(static_cast<bool>(firstFill) && firstFill.status() == CacheStatus::CacheFill, "first miss reserves a cache fill");
        cacheDeviceData = firstFill.writableData();
        test.expect(cacheDeviceData != nullptr && firstFill.data() == cacheDeviceData, "cache fill exposes the preallocated device pointer");

        GpuDataAccess loadingFallback = cache.getCacheData(firstMetadata, makeCacheRequest(resources));
        test.expect(static_cast<bool>(loadingFallback) && loadingFallback.status() == CacheStatus::TaskFallback, "same frame loading uses task fallback without waiting");
        test.expect(loadingFallback.writableData() == resources.d_input, "loading fallback uses the task-private input buffer");
        const bool fallbackMemset = cudaMemsetAsync(loadingFallback.writableData(), 0x19, loadingFallback.bytes(), resources.stream) == cudaSuccess;
        test.expect(loadingFallback.freeCacheData(fallbackMemset), "complete loading fallback without publishing cache state");

        const bool fillMemset = cudaMemsetAsync(firstFill.writableData(), 0x2a, firstFill.bytes(), resources.stream) == cudaSuccess;
        test.expect(firstFill.freeCacheData(fillMemset), "publish cache fill after stream synchronization");
    }

    FrameMetadata mismatchedMetadata = firstMetadata;
    ++mismatchedMetadata.width;
    GpuDataAccess mismatchedAccess = cache.getCacheData(mismatchedMetadata, makeCacheRequest(resources));
    test.expect(!mismatchedAccess, "reject the same frame ID with different metadata");

    GpuDataAccess failedHit = cache.getCacheData(firstMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(failedHit) && failedHit.status() == CacheStatus::CacheHit && !failedHit.freeCacheData(false), "failed cache reader releases its lease without publishing state");
    GpuDataAccess hitAfterFailure = cache.getCacheData(firstMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(hitAfterFailure) && hitAfterFailure.status() == CacheStatus::CacheHit && hitAfterFailure.freeCacheData(true), "failed cache reader preserves the immutable cached payload");

    GpuDataAccess activeHit = cache.getCacheData(firstMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(activeHit) && activeHit.status() == CacheStatus::CacheHit, "valid matching entry returns a cache hit");
    test.expect(activeHit.data() == cacheDeviceData && activeHit.writableData() == nullptr, "cache hit is immutable and reuses the cached pointer");
    GpuDataAccess secondReader = cache.getCacheData(firstMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(secondReader) && secondReader.status() == CacheStatus::CacheHit && secondReader.freeCacheData(true), "matching immutable cache readers may coexist");

    GpuDataAccess busyFallback = cache.getCacheData(secondMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(busyFallback) && busyFallback.status() == CacheStatus::TaskFallback, "all active cache entries force task fallback");
    test.expect(busyFallback.freeCacheData(true), "complete busy-cache fallback");
    test.expect(!cache.release(), "cache release rejects an active reader");
    test.expect(activeHit.freeCacheData(true), "release active cache reader");

    {
        GpuDataAccess abandonedFill = cache.getCacheData(secondMetadata, makeCacheRequest(resources));
        test.expect(static_cast<bool>(abandonedFill) && abandonedFill.status() == CacheStatus::CacheFill, "inactive LRU entry can be reused for another frame");
    }
    GpuDataAccess retryAfterAbort = cache.getCacheData(secondMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(retryAfterAbort) && retryAfterAbort.status() == CacheStatus::CacheFill && retryAfterAbort.writableData() == cacheDeviceData, "aborted fill returns the same allocation to the cache");
    test.expect(!retryAfterAbort.freeCacheData(false), "failed fill is not published");

    GpuDataAccess successfulRetry = cache.getCacheData(secondMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(successfulRetry) && successfulRetry.status() == CacheStatus::CacheFill, "failed fill leaves an empty reusable entry");
    if (successfulRetry) {
        const bool memsetSucceeded = cudaMemsetAsync(successfulRetry.writableData(), 0x17, successfulRetry.bytes(), resources.stream) == cudaSuccess;
        test.expect(successfulRetry.freeCacheData(memsetSucceeded), "successful retry publishes the new frame");
    }

    TaskGpuResources wrongGpuResources = resources;
    wrongGpuResources.gpuId = location.gpuId + 1;
    GpuDataAccess wrongGpuRead = cache.getCacheData(secondMetadata, makeCacheRequest(wrongGpuResources));
    test.expect(!wrongGpuRead, "reject access from a GPU without a replica");

    test.expect(cache.release(), "release bounded frame GPU cache");
    test.expect(!cache.isInitialized() && cache.entryCount() == 0, "cache release resets allocation state");
    test.expect(cache.release(), "repeated cache release is harmless");

    GpuCacheManager zeroCapacityCache;
    test.expect(zeroCapacityCache.initialize({location.gpuId}, runtime.inBytes, 0), "initialize a zero-capacity cache");
    GpuDataAccess zeroCapacityAccess = zeroCapacityCache.getCacheData(firstMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(zeroCapacityAccess) && zeroCapacityAccess.status() == CacheStatus::TaskFallback && zeroCapacityAccess.writableData() == resources.d_input, "zero cache capacity always uses task fallback");
    test.expect(!zeroCapacityCache.resetCache(), "cache reset rejects an active fallback lease");
    test.expect(!zeroCapacityCache.release(), "cache release rejects an active fallback lease");
    test.expect(zeroCapacityAccess.freeCacheData(true), "complete zero-capacity fallback");
    {
        GpuDataAccess abandonedFallback = zeroCapacityCache.getCacheData(firstMetadata, makeCacheRequest(resources));
        test.expect(static_cast<bool>(abandonedFallback) && abandonedFallback.status() == CacheStatus::TaskFallback, "acquire fallback used for RAII abort");
    }
    test.expect(zeroCapacityCache.release(), "release zero-capacity cache");
    test.expect(releaseAccessResources(resources), "release frame cache test resources");
}

void testGpuDataAccessMoves(TestContext& test, const GpuLocation& location) {
    static_assert(!std::is_copy_constructible<GpuDataAccess>::value, "leases must not be copied");
    static_assert(!std::is_copy_assignable<GpuDataAccess>::value, "leases must not be copy assigned");
    static_assert(std::is_nothrow_move_constructible<GpuDataAccess>::value, "lease transfer must not throw");
    static_assert(std::is_nothrow_move_assignable<GpuDataAccess>::value, "lease assignment must not throw");
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    const FrameMetadata metadata = makeFrameMetadata(70, runtime);
    const FrameMetadata otherMetadata = makeFrameMetadata(71, runtime);
    TaskGpuResources resources;
    GpuCacheManager cache;
    GpuCacheManager fallbackCache;
    const bool initialized = initializeAccessResources(resources, location, runtime.inBytes) && cache.initialize({location.gpuId}, runtime.inBytes, 1, std::chrono::milliseconds(0)) && fallbackCache.initialize({location.gpuId}, runtime.inBytes, 0);
    test.expect(initialized, "initialize move test resources");
    if (!initialized) {
        cache.release();
        fallbackCache.release();
        releaseAccessResources(resources);
        return;
    }
    const GpuCacheRequest request = makeCacheRequest(resources);
    std::array<GpuDataAccess, 2> saved;
    const void* originalData = nullptr;
    {
        GpuDataAccess source = cache.getCacheData(metadata, request);
        originalData = source.data();
        const bool submitted = cudaMemsetAsync(source.writableData(), 0x35, source.bytes(), source.getStream()) == cudaSuccess;
        test.expect(submitted, "enqueue fill before moving lease");
        GpuDataAccess transferred(std::move(source));
        test.expect(!source && source.status() == CacheStatus::Invalid && source.data() == nullptr && source.writableData() == nullptr && source.getStream() == nullptr && source.bytes() == 0 && source.gpuId() == -1, "move construction fully invalidates source");
        test.expect(!source.freeCacheData(true), "moved-from lease cannot complete transferred fill");
        saved[0] = std::move(transferred);
        test.expect(!transferred && saved[0].status() == CacheStatus::CacheFill && saved[0].data() == originalData && saved[0].getStream() == request.stream && saved[0].bytes() == runtime.inBytes && saved[0].gpuId() == location.gpuId, "store transferred fill in a preallocated member slot");
    }
    test.expect(!cache.resetCache() && !cache.release(), "source destructors do not release the saved lease");
    {
        GpuDataAccess sameKey = cache.getCacheData(metadata, request);
        GpuDataAccess otherKey = cache.getCacheData(otherMetadata, request);
        test.expect(sameKey.status() == CacheStatus::TaskFallback && otherKey.status() == CacheStatus::TaskFallback, "move neither publishes nor makes the fill evictable");
        test.expect(sameKey.freeCacheData(true) && otherKey.freeCacheData(true), "finish move test fallback requests");
    }
    GpuDataAccess& self = saved[0];
    saved[0] = std::move(self);
    test.expect(saved[0].data() == originalData && saved[0].status() == CacheStatus::CacheFill, "self-move preserves the live lease");
    test.expect(saved[0].freeCacheData(true), "saved fill publishes exactly once");
    test.expect(!saved[0].freeCacheData(true), "completed moved lease rejects duplicate completion");
    saved[0] = cache.getCacheData(metadata, request);
    saved[1] = std::move(saved[0]);
    test.expect(saved[1].status() == CacheStatus::CacheHit && saved[1].data() == originalData && saved[1].writableData() == nullptr, "moved hit remains read-only");
    test.expect(!cache.resetCache(), "moved hit keeps reader protection");
    unsigned char value = 0;
    test.expect(cudaMemcpyAsync(&value, saved[1].data(), 1, cudaMemcpyDeviceToHost, saved[1].getStream()) == cudaSuccess && saved[1].freeCacheData(true) && value == 0x35, "moved hit retains filled payload");

    saved[0] = cache.getCacheData(otherMetadata, request);
    test.expect(saved[0].status() == CacheStatus::CacheFill, "acquire destination fill to be replaced");
    test.expect(cudaMemsetAsync(saved[0].writableData(), 0x46, saved[0].bytes(), saved[0].getStream()) == cudaSuccess, "submit work on replaced destination");
    saved[0] = fallbackCache.getCacheData(metadata, request);
    test.expect(saved[0].status() == CacheStatus::TaskFallback && saved[0].data() == resources.d_input, "assignment transfers fallback from another manager");
    test.expect(cache.statisticsSnapshot().fillFailed == 1, "overwriting a live destination aborts instead of publishing its fill");
    {
        GpuDataAccess retry = cache.getCacheData(otherMetadata, request);
        test.expect(retry.status() == CacheStatus::CacheFill, "overwritten fill can be retried");
    }
    test.expect(cache.release(), "old manager has no orphaned lease after cross-manager assignment");
    saved[1] = fallbackCache.getCacheData(otherMetadata, request);
    saved[1] = std::move(saved[0]);
    test.expect(!saved[0] && !fallbackCache.resetCache() && !fallbackCache.release(), "replacing a fallback decrements only the old lease");
    saved[1] = GpuDataAccess();
    test.expect(fallbackCache.resetCache(), "assignment from Invalid aborts the final fallback lease");
    {
        GpuDataAccess abandoned;
        {
            GpuDataAccess source = fallbackCache.getCacheData(metadata, request);
            abandoned = std::move(source);
        }
        test.expect(!fallbackCache.release(), "moved fallback remains active beyond source scope");
    }
    test.expect(fallbackCache.release(), "moved destination destructor releases fallback once");
    test.expect(releaseAccessResources(resources), "release move test resources");
}

void testIndependentPayloadCaches(TestContext& test, const GpuLocation& location, std::size_t capacity) {
    constexpr std::size_t FRAME_BYTES = 16;
    constexpr std::size_t RESULT_BYTES = 64;
    TaskGpuResources resources;
    GpuCacheManager frameCache;
    GpuCacheManager resultCache;
    void* d_resultFallback = nullptr;
    const bool initialized = initializeAccessResources(resources, location, FRAME_BYTES) && cudaMalloc(&d_resultFallback, RESULT_BYTES) == cudaSuccess && frameCache.initialize({location.gpuId}, FRAME_BYTES, capacity) && resultCache.initialize({location.gpuId}, RESULT_BYTES, capacity);
    test.expect(initialized, "initialize independent frame/result caches with different payload sizes");
    if (!initialized) {
        frameCache.release();
        resultCache.release();
        if (d_resultFallback != nullptr) {
            cudaFree(d_resultFallback);
        }
        releaseAccessResources(resources);
        return;
    }

    FrameMetadata frameMetadata;
    frameMetadata.key = {42, 7};
    frameMetadata.width = 4;
    frameMetadata.height = 4;
    frameMetadata.dtype = 1;
    frameMetadata.bytes = FRAME_BYTES;
    FrameMetadata resultMetadata = frameMetadata;
    resultMetadata.dtype = 4;
    resultMetadata.bytes = RESULT_BYTES;
    const GpuCacheRequest frameRequest = makeCacheRequest(resources);
    GpuCacheRequest resultRequest = frameRequest;
    resultRequest.d_fallback = d_resultFallback;
    resultRequest.fallbackBytes = RESULT_BYTES;

    GpuCacheRequest invalidRequest = resultRequest;
    invalidRequest.fallbackBytes = RESULT_BYTES - 1;
    GpuDataAccess tooSmall = resultCache.getCacheData(resultMetadata, invalidRequest);
    test.expect(tooSmall.status() == CacheStatus::Invalid && tooSmall.data() == nullptr && tooSmall.getStream() == nullptr && !tooSmall.freeCacheData(true), "reject undersized fallback without exposing a usable access or stream");
    invalidRequest = resultRequest;
    invalidRequest.stream = nullptr;
    GpuDataAccess noStream = resultCache.getCacheData(resultMetadata, invalidRequest);
    test.expect(noStream.status() == CacheStatus::Invalid, "reject a request without an explicit stream");
    invalidRequest = resultRequest;
    invalidRequest.d_fallback = nullptr;
    GpuDataAccess noFallback = resultCache.getCacheData(resultMetadata, invalidRequest);
    test.expect(noFallback.status() == CacheStatus::Invalid, "reject a request without caller-owned fallback storage");

    std::array<unsigned char, FRAME_BYTES> h_frame{};
    std::array<unsigned char, RESULT_BYTES> h_result{};
    {
        GpuDataAccess frame = frameCache.getCacheData(frameMetadata, frameRequest);
        GpuDataAccess result = resultCache.getCacheData(resultMetadata, resultRequest);
        const CacheStatus expectedStatus = capacity == 0 ? CacheStatus::TaskFallback : CacheStatus::CacheFill;
        test.expect(frame.status() == expectedStatus && result.status() == expectedStatus, "same key independently reserves a frame and a result payload");
        test.expect(frame.bytes() == FRAME_BYTES && result.bytes() == RESULT_BYTES && frame.data() != result.data(), "different payload sizes use independent storage");
        test.expect(frame.getStream() == resources.stream && result.getStream() == resources.stream, "both accesses expose the borrowed caller stream");
        if (capacity == 0) {
            test.expect(frame.data() == resources.d_input && result.data() == d_resultFallback, "simultaneous frame/result fallbacks do not alias");
        }
        bool submitted = static_cast<bool>(frame) && static_cast<bool>(result);
        if (submitted) {
            submitted = cudaMemsetAsync(frame.writableData(), 0x19, frame.bytes(), frame.getStream()) == cudaSuccess && cudaMemsetAsync(result.writableData(), 0x37, result.bytes(), result.getStream()) == cudaSuccess;
        }
        if (submitted) {
            submitted = cudaMemcpyAsync(h_frame.data(), frame.data(), frame.bytes(), cudaMemcpyDeviceToHost, frame.getStream()) == cudaSuccess && cudaMemcpyAsync(h_result.data(), result.data(), result.bytes(), cudaMemcpyDeviceToHost, result.getStream()) == cudaSuccess;
        }
        test.expect(frame.freeCacheData(submitted), "finish the frame access after caller-submitted work");
        test.expect(frame.status() == CacheStatus::Invalid && frame.getStream() == nullptr && frame.data() == nullptr && frame.writableData() == nullptr && !frame.freeCacheData(true), "finishing invalidates status, stream, pointers, and repeated finish");
        test.expect(!resultCache.resetCache() && !resultCache.release(), "the other cache retains its live lease despite synchronization of the shared stream");
        if (capacity != 0) {
            GpuDataAccess pendingResult = resultCache.getCacheData(resultMetadata, resultRequest);
            test.expect(pendingResult.status() == CacheStatus::TaskFallback && pendingResult.freeCacheData(true), "a result is not published until its own freeCacheData call");
        }
        test.expect(result.freeCacheData(submitted), "finish the independent result access");
    }
    test.expect(std::all_of(h_frame.begin(), h_frame.end(), [](unsigned char value) { return value == 0x19; }) && std::all_of(h_result.begin(), h_result.end(), [](unsigned char value) { return value == 0x37; }), "caller-generated frame/result bytes survive simultaneous access without overwriting each other");
    if (capacity != 0) {
        // A larger fallback allocation is valid even when the requested payload is smaller.
        GpuDataAccess frameHit = frameCache.getCacheData(frameMetadata, resultRequest);
        GpuDataAccess resultHit = resultCache.getCacheData(resultMetadata, resultRequest);
        test.expect(frameHit.status() == CacheStatus::CacheHit && resultHit.status() == CacheStatus::CacheHit && frameHit.writableData() == nullptr && resultHit.writableData() == nullptr, "published frame and result keys hit independently as read-only data");
        test.expect(frameHit.freeCacheData(true) && resultHit.freeCacheData(true), "release both independent readers");
        test.expect(frameCache.resetCache(), "reset only the frame cache");
        GpuDataAccess frameRefill = frameCache.getCacheData(frameMetadata, frameRequest);
        GpuDataAccess preservedResult = resultCache.getCacheData(resultMetadata, resultRequest);
        test.expect(frameRefill.status() == CacheStatus::CacheFill && preservedResult.status() == CacheStatus::CacheHit, "resetting one cache preserves the other cache's key");
        test.expect(!frameRefill.freeCacheData(false) && preservedResult.freeCacheData(true), "rollback of a frame fill leaves the result cache intact");
    }
    test.expect(frameCache.release() && resultCache.release(), "release independent payload caches");
    test.expect(cudaFree(d_resultFallback) == cudaSuccess && releaseAccessResources(resources), "caller releases fallback allocations and stream after all leases finish");
}

void testConcurrentCacheRequests(TestContext& test, const GpuLocation& location) {
    constexpr std::size_t PAYLOAD_BYTES = 64;
    std::array<TaskGpuResources, 3> resources;
    GpuCacheManager cache;
    bool initialized = cache.initialize({location.gpuId}, PAYLOAD_BYTES, 1);
    for (TaskGpuResources& taskResources : resources) {
        initialized = initializeAccessResources(taskResources, location, PAYLOAD_BYTES) && initialized;
    }
    test.expect(initialized, "initialize concurrent callers with independent streams and fallbacks");
    if (!initialized) {
        cache.release();
        for (TaskGpuResources& taskResources : resources) {
            releaseAccessResources(taskResources);
        }
        return;
    }
    FrameMetadata metadata;
    metadata.key = {73, 2};
    metadata.width = 8;
    metadata.height = 8;
    metadata.dtype = 1;
    metadata.bytes = PAYLOAD_BYTES;
    {
        GpuDataAccess fill = cache.getCacheData(metadata, makeCacheRequest(resources[0]));
        const bool submitted = fill.status() == CacheStatus::CacheFill && cudaMemsetAsync(fill.writableData(), 0x5a, fill.bytes(), fill.getStream()) == cudaSuccess;
        test.expect(fill.freeCacheData(submitted), "publish payload before concurrent readers start");
    }
    std::mutex readerLock;
    std::condition_variable readerCondition;
    std::size_t readyReaders = 0;
    bool finishReaders = false;
    std::array<bool, 2> succeeded{};
    std::array<std::array<unsigned char, PAYLOAD_BYTES>, 2> h_outputs{};
    std::array<std::thread, 2> readers;
    for (std::size_t index = 0; index < readers.size(); ++index) {
        readers[index] = NumaExecutor(location.numaNode).start([&, index](bool numaReady) {
            const bool deviceReady = numaReady && cudaSetDevice(location.gpuId) == cudaSuccess;
            GpuDataAccess access = deviceReady ? cache.getCacheData(metadata, makeCacheRequest(resources[index + 1])) : GpuDataAccess();
            const bool hit = access.status() == CacheStatus::CacheHit && access.writableData() == nullptr && access.getStream() == resources[index + 1].stream;
            {
                std::unique_lock<std::mutex> guard(readerLock);
                ++readyReaders;
                readerCondition.notify_all();
                readerCondition.wait(guard, [&finishReaders] { return finishReaders; });
            }
            const bool submitted = hit && cudaMemcpyAsync(h_outputs[index].data(), access.data(), access.bytes(), cudaMemcpyDeviceToHost, access.getStream()) == cudaSuccess;
            succeeded[index] = access.freeCacheData(submitted);
        });
    }
    {
        std::unique_lock<std::mutex> guard(readerLock);
        readerCondition.wait(guard, [&readyReaders] { return readyReaders == 2; });
    }
    test.expect(!cache.resetCache() && !cache.release(), "concurrent reader leases block reset and release");
    FrameMetadata anotherKey = metadata;
    ++anotherKey.key.frameId;
    {
        GpuDataAccess busy = cache.getCacheData(anotherKey, makeCacheRequest(resources[0]));
        test.expect(busy.status() == CacheStatus::TaskFallback && busy.freeCacheData(true), "a different key cannot evict the entry held by concurrent readers");
    }
    {
        std::lock_guard<std::mutex> guard(readerLock);
        finishReaders = true;
    }
    readerCondition.notify_all();
    for (std::thread& reader : readers) {
        reader.join();
    }
    for (std::size_t index = 0; index < succeeded.size(); ++index) {
        test.expect(succeeded[index] && std::all_of(h_outputs[index].begin(), h_outputs[index].end(), [](unsigned char value) { return value == 0x5a; }), "each concurrent caller reads the same immutable payload on its own stream");
    }
    test.expect(cache.resetCache() && cache.release(), "the cache can reset and release after the last concurrent reader finishes");
    for (TaskGpuResources& taskResources : resources) {
        test.expect(releaseAccessResources(taskResources), "release concurrent caller stream and fallback");
    }
}

void testCacheWaitWakeup(TestContext& test, const GpuLocation& location, int scenario) {
    constexpr std::size_t PAYLOAD_BYTES = 64;
    GpuCacheManager cache;
    std::array<TaskGpuResources, 2> resources;
    bool initialized = cache.initialize({location.gpuId}, PAYLOAD_BYTES, 1, std::chrono::milliseconds(1000));
    for (TaskGpuResources& taskResources : resources) {
        initialized = initializeAccessResources(taskResources, location, PAYLOAD_BYTES) && initialized;
    }
    test.expect(initialized, "initialize cache wait wakeup scenario");
    if (!initialized) {
        cache.release();
        for (TaskGpuResources& taskResources : resources) {
            releaseAccessResources(taskResources);
        }
        return;
    }

    FrameMetadata metadata;
    metadata.key = {81, 3};
    metadata.width = 8;
    metadata.height = 8;
    metadata.dtype = 1;
    metadata.bytes = PAYLOAD_BYTES;
    FrameMetadata requestedMetadata = metadata;
    if (scenario == 3) {
        ++requestedMetadata.key.frameId;
    }
    std::promise<void> startedPromise;
    std::future<void> started = startedPromise.get_future();
    std::promise<CacheStatus> resultPromise;
    std::future<CacheStatus> result = resultPromise.get_future();
    std::thread waiter;
    {
        GpuDataAccess producer = cache.getCacheData(metadata, makeCacheRequest(resources[0]));
        test.expect(producer.status() == CacheStatus::CacheFill, "reserve original fill before waiter");
        const bool submitted = cudaMemsetAsync(producer.writableData(), 0x5a, PAYLOAD_BYTES, producer.getStream()) == cudaSuccess;
        if (scenario == 3) {
            test.expect(producer.freeCacheData(submitted), "publish before holding a reader in full-cache scenario");
        }
        GpuDataAccess reader = scenario == 3 ? cache.getCacheData(metadata, makeCacheRequest(resources[0])) : GpuDataAccess();
        waiter = NumaExecutor(location.numaNode).start([&](bool numaReady) {
            const bool ready = numaReady && cudaSetDevice(location.gpuId) == cudaSuccess;
            startedPromise.set_value();
            GpuDataAccess access = ready ? cache.getCacheData(requestedMetadata, makeCacheRequest(resources[1])) : GpuDataAccess();
            const CacheStatus status = access.status();
            bool succeeded = status == (scenario == 0 ? CacheStatus::CacheHit : CacheStatus::CacheFill);
            if (status == CacheStatus::CacheHit) {
                std::array<unsigned char, PAYLOAD_BYTES> h_output{};
                succeeded = cudaMemcpyAsync(h_output.data(), access.data(), PAYLOAD_BYTES, cudaMemcpyDeviceToHost, access.getStream()) == cudaSuccess && succeeded;
                succeeded = access.freeCacheData(succeeded) && succeeded;
                succeeded = std::all_of(h_output.begin(), h_output.end(), [](unsigned char value) { return value == 0x5a; }) && succeeded;
            }
            else {
                succeeded = access.freeCacheData(succeeded) && succeeded;
            }
            resultPromise.set_value(succeeded ? status : CacheStatus::Invalid);
        });
        started.wait();
        test.expect(result.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout, "request waits while matching fill or full-cache reader is active");
        test.expect(!cache.resetCache() && !cache.release(), "live producer and waiting request block reset and release");
        if (scenario == 0) {
            test.expect(producer.freeCacheData(submitted), "successful fill wakes same-key waiter");
        }
        else if (scenario == 1) {
            test.expect(!producer.freeCacheData(false), "failed fill wakes a waiter to retry CacheFill");
        }
        else if (scenario == 3) {
            test.expect(reader.freeCacheData(true), "last reader release wakes a full-cache waiter");
        }
        // Scenario 2 deliberately leaves the fill for RAII rollback.
    }
    test.expect(result.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready, "state change wakes the request before its one-second deadline");
    const CacheStatus status = result.get();
    waiter.join();
    test.expect(status == (scenario == 0 ? CacheStatus::CacheHit : CacheStatus::CacheFill), "waiter receives the correct cache lease after wakeup");
    const CacheStatistics stats = cache.statisticsSnapshot();
    test.expect(stats.waitedRequests == 1 && stats.totalWaitTime.count() > 0 && stats.maxWaitTime == stats.totalWaitTime, "one waiting request accumulates one total and maximum sample");
    test.expect(stats.fallback == 0 && stats.invalid == 0 && stats.firstBlockedLoading == (scenario == 3 ? 0U : 1U) && stats.firstBlockedFull == (scenario == 3 ? 1U : 0U), "wakeup counts first obstruction without a fallback");
    test.expect(stats.fill == (scenario == 0 ? 1U : 2U) && stats.hit == (scenario == 0 || scenario == 3 ? 1U : 0U), "wakeup lookup counts only final hit or fill");
    test.expect(stats.fillFailed == (scenario == 1 || scenario == 2 ? 1U : 0U) && stats.fillSucceeded + stats.fillFailed == stats.fill, "completion counters reconcile after all fills finish");
    test.expect(cache.resetCache() && cache.release(), "reset and release succeed after waiters and leases finish");
    for (TaskGpuResources& taskResources : resources) {
        test.expect(releaseAccessResources(taskResources), "release waiter resources");
    }
}

void testCacheWaitSharedDeadline(TestContext& test, const GpuLocation& location) {
    constexpr std::size_t PAYLOAD_BYTES = 64;
    GpuCacheManager cache;
    std::array<TaskGpuResources, 2> resources;
    bool initialized = cache.initialize({location.gpuId}, PAYLOAD_BYTES, 2, std::chrono::milliseconds(80));
    for (TaskGpuResources& taskResources : resources) {
        initialized = initializeAccessResources(taskResources, location, PAYLOAD_BYTES) && initialized;
    }
    test.expect(initialized, "initialize deadline test with an unrelated entry");
    if (!initialized) {
        cache.release();
        for (TaskGpuResources& taskResources : resources) {
            releaseAccessResources(taskResources);
        }
        return;
    }
    FrameMetadata metadata;
    metadata.key = {99, 3};
    metadata.width = 8;
    metadata.height = 8;
    metadata.dtype = 1;
    metadata.bytes = PAYLOAD_BYTES;
    FrameMetadata unrelatedMetadata = metadata;
    ++unrelatedMetadata.key.frameId;
    {
        GpuDataAccess unrelated = cache.getCacheData(unrelatedMetadata, makeCacheRequest(resources[1]));
        test.expect(unrelated.freeCacheData(true), "publish unrelated entry for repeated notifications");
    }
    {
        GpuDataAccess producer = cache.getCacheData(metadata, makeCacheRequest(resources[0]));
        std::promise<void> startedPromise;
        std::future<void> started = startedPromise.get_future();
        bool notifierSucceeded = true;
        std::thread notifier = NumaExecutor(location.numaNode).start([&](bool numaReady) {
            notifierSucceeded = numaReady && cudaSetDevice(location.gpuId) == cudaSuccess;
            startedPromise.set_value();
            const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
            while (notifierSucceeded && std::chrono::steady_clock::now() < end) {
                GpuDataAccess unrelated = cache.getCacheData(unrelatedMetadata, makeCacheRequest(resources[1]));
                notifierSucceeded = unrelated.status() == CacheStatus::CacheHit && unrelated.freeCacheData(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
        started.wait();
        const std::chrono::steady_clock::time_point startedWaiting = std::chrono::steady_clock::now();
        GpuDataAccess waiting = cache.getCacheData(metadata, makeCacheRequest(resources[0]));
        const std::chrono::steady_clock::duration elapsed = std::chrono::steady_clock::now() - startedWaiting;
        test.expect(waiting.status() == CacheStatus::TaskFallback && elapsed >= std::chrono::milliseconds(80) && elapsed < std::chrono::milliseconds(250), "unrelated wakeups do not restart the request deadline");
        test.expect(waiting.freeCacheData(true) && producer.freeCacheData(true), "finish deadline test accesses");
        notifier.join();
        test.expect(notifierSucceeded, "unrelated entry repeatedly becomes evictable during wait");
    }
    const CacheStatistics stats = cache.statisticsSnapshot();
    test.expect(stats.waitedRequests == 1 && stats.firstBlockedLoading == 1 && stats.fallbackReasons.loadingTimeout == 1 && stats.fallback == 1 && stats.fill == 2, "repeated unrelated wakeups do not multiply request counters");
    test.expect(stats.maxWaitTime == stats.totalWaitTime && stats.totalWaitTime.count() > 0, "sum wait episodes per request before calculating maximum");
    test.expect(cache.release(), "release shared deadline cache");
    for (TaskGpuResources& taskResources : resources) {
        test.expect(releaseAccessResources(taskResources), "release shared deadline resources");
    }
}

void testCacheWaitTimeout(TestContext& test, const GpuLocation& location) {
    constexpr std::size_t PAYLOAD_BYTES = 64;
    TaskGpuResources resources;
    test.expect(initializeAccessResources(resources, location, PAYLOAD_BYTES), "initialize timeout resources");
    FrameMetadata metadata;
    metadata.key = {91, 3};
    metadata.width = 8;
    metadata.height = 8;
    metadata.dtype = 1;
    metadata.bytes = PAYLOAD_BYTES;
    GpuCacheManager invalidCache;
    test.expect(!invalidCache.initialize({location.gpuId}, PAYLOAD_BYTES, 1, std::chrono::milliseconds(-1)), "reject a negative wait timeout");
    for (int scenario = 0; scenario < 4; ++scenario) {
        GpuCacheManager cache;
        const bool initialized = scenario == 0 || scenario == 1 ? cache.initialize({location.gpuId}, PAYLOAD_BYTES, 1) : cache.initialize({location.gpuId}, PAYLOAD_BYTES, scenario == 2 ? 1 : 0, std::chrono::milliseconds(scenario == 2 ? 0 : 1000));
        test.expect(initialized, "initialize default, disabled or zero-capacity wait");
        {
            GpuDataAccess producer = cache.getCacheData(metadata, makeCacheRequest(resources));
            FrameMetadata requestedMetadata = metadata;
            if (scenario == 1) {
                ++requestedMetadata.key.frameId;
            }
            const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
            GpuDataAccess blocked = cache.getCacheData(requestedMetadata, makeCacheRequest(resources));
            const std::chrono::steady_clock::duration elapsed = std::chrono::steady_clock::now() - started;
            test.expect(blocked.status() == CacheStatus::TaskFallback, "unavailable entry falls back at the deadline");
            if (scenario < 2) {
                test.expect(elapsed >= std::chrono::milliseconds(50) && elapsed < std::chrono::milliseconds(500), "default Loading and full-cache requests wait for 50 ms");
            }
            else {
                test.expect(elapsed < std::chrono::milliseconds(250), "zero timeout or zero capacity bypasses waiting");
            }
            test.expect(blocked.freeCacheData(true), "finish timeout fallback");
            test.expect(producer.freeCacheData(true), "finish original access after timeout");
        }
        const CacheStatistics stats = cache.statisticsSnapshot();
        test.expect(stats.waitedRequests == (scenario < 2 ? 1U : 0U) && stats.fallback == (scenario == 3 ? 2U : 1U), "timeout outcomes and waited-request counts");
        test.expect(stats.fallbackReasons.loadingTimeout == (scenario == 0 ? 1U : 0U) && stats.fallbackReasons.fullTimeout == (scenario == 1 ? 1U : 0U) && stats.fallbackReasons.loadingNoWait == (scenario == 2 ? 1U : 0U) && stats.fallbackReasons.capacityZero == (scenario == 3 ? 2U : 0U), "fallback reasons distinguish timeout, no wait and zero capacity");
        test.expect(stats.firstBlockedLoading + stats.firstBlockedFull == (scenario == 3 ? 0U : 1U), "zero capacity is not a Loading or Full obstruction");
        test.expect(cache.release(), "release cache after timeout");
    }
    test.expect(releaseAccessResources(resources), "release timeout resources");
}

void testCacheStatistics(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    TaskGpuResources resources;
    GpuCacheManager cache;
    GpuCacheManager independent;
    const bool initialized = initializeAccessResources(resources, location, runtime.inBytes) && cache.initialize({location.gpuId}, runtime.inBytes, 1, std::chrono::milliseconds(0)) && independent.initialize({location.gpuId}, runtime.inBytes, 0);
    test.expect(initialized, "initialize isolated statistics managers");
    if (!initialized) {
        cache.release();
        independent.release();
        releaseAccessResources(resources);
        return;
    }
    const FrameMetadata first = makeFrameMetadata(777, runtime);
    const FrameMetadata second = makeFrameMetadata(778, runtime);
    CacheStatistics completed;
    completed.hit = 999;
    {
        GpuDataAccess fill = cache.getCacheData(first, makeCacheRequest(resources));
        GpuDataAccess loading = cache.getCacheData(first, makeCacheRequest(resources));
        GpuDataAccess full = cache.getCacheData(second, makeCacheRequest(resources));
        GpuCacheRequest invalidRequest = makeCacheRequest(resources);
        invalidRequest.stream = nullptr;
        GpuDataAccess invalid = cache.getCacheData(first, invalidRequest);
        const CacheStatistics live = cache.statisticsSnapshot();
        test.expect(live.fill == 1 && live.fillSucceeded == 0 && live.fallback == 2 && live.invalid == 1, "count final outcomes while fill remains outstanding");
        test.expect(live.firstBlockedLoading == 1 && live.firstBlockedFull == 1 && live.fallbackReasons.loadingNoWait == 1 && live.fallbackReasons.fullNoWait == 1, "classify immediate fallback and first obstruction");
        test.expect(live.waitedRequests == 0 && live.totalWaitTime.count() == 0 && live.maxWaitTime.count() == 0, "disabled waiting does not contribute wait samples");
        test.expect(!cache.resetCache(&completed) && completed.hit == 999 && cache.statisticsSnapshot().fill == 1, "rejected reset leaves output and counters unchanged");
        test.expect(loading.freeCacheData(true) && full.freeCacheData(true) && fill.freeCacheData(true), "complete initial fill and fallbacks");
    }
    {
        GpuDataAccess hit = cache.getCacheData(first, makeCacheRequest(resources));
        test.expect(hit.status() == CacheStatus::CacheHit && hit.freeCacheData(true), "record hit after publication");
        GpuDataAccess replacement = cache.getCacheData(second, makeCacheRequest(resources));
        test.expect(!replacement.freeCacheData(false), "failed replacement still counts eviction");
    }
    {
        GpuDataAccess abandoned = cache.getCacheData(first, makeCacheRequest(resources));
        test.expect(abandoned.status() == CacheStatus::CacheFill, "reserve fill for RAII failure statistics");
    }
    const CacheStatistics snapshot = cache.statisticsSnapshot();
    test.expect(snapshot.hit == 1 && snapshot.fill == 3 && snapshot.fillSucceeded == 1 && snapshot.fillFailed == 2 && snapshot.eviction == 1, "count successes, explicit failure, RAII failure and eviction independently");
    test.expect(cache.statisticsSnapshot().fill == snapshot.fill && independent.statisticsSnapshot().fill == 0, "snapshot is nondestructive and managers are independent");
    test.expect(cache.resetCache(&completed) && completed.hit == 1 && completed.fill == 3 && completed.fillFailed == 2 && completed.fallback == 2 && completed.invalid == 1, "successful reset returns the previous interval");
    const CacheStatistics cleared = cache.statisticsSnapshot();
    test.expect(cleared.hit == 0 && cleared.fill == 0 && cleared.fallback == 0 && cleared.invalid == 0 && cleared.eviction == 0 && cleared.firstBlockedFull == 0 && cleared.fillFailed == 0, "successful reset clears interval statistics");
    test.expect(cache.release() && independent.release() && releaseAccessResources(resources), "release statistics test resources");
}

void testGpuCacheResetBoundaries(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    const FrameMetadata firstCamera = makeFrameMetadata(47, runtime, 3);
    const FrameMetadata secondCamera = makeFrameMetadata(47, runtime, 4);
    const FrameMetadata arbitraryIncoming = makeFrameMetadata(700003, runtime, 91);
    StaticData staticData;
    TaskGpuResources resources;
    const bool staticDataReady = initializeStaticData(staticData, location, runtime, 1);
    const bool resourcesReady = initializeAccessResources(resources, location, runtime.inBytes);
    test.expect(staticDataReady && resourcesReady, "initialize cache run-boundary test resources");
    if (!staticDataReady || !resourcesReady) {
        if (staticDataReady) {
            releaseStaticData(staticData, location);
        }
        if (resourcesReady) {
            releaseAccessResources(resources);
        }
        return;
    }

    void* persistentDeviceData = nullptr;
    {
        GpuDataAccess firstFill = staticData.getCacheData(firstCamera, makeCacheRequest(resources));
        test.expect(static_cast<bool>(firstFill) && firstFill.status() == CacheStatus::CacheFill, "first composite key reserves the cache entry");
        persistentDeviceData = firstFill.writableData();
        const bool submitted = persistentDeviceData != nullptr && cudaMemsetAsync(persistentDeviceData, 0x2f, firstFill.bytes(), resources.stream) == cudaSuccess;
        test.expect(firstFill.freeCacheData(submitted), "publish the first composite-key fill");
    }

    {
        GpuDataAccess activeHit = staticData.getCacheData(firstCamera, makeCacheRequest(resources));
        test.expect(static_cast<bool>(activeHit) && activeHit.status() == CacheStatus::CacheHit, "reacquire the first composite key as a hit");
        if (activeHit) {
            test.expect(!staticData.resetCache(), "reject cache reset while a lease is active");
            test.expect(activeHit.freeCacheData(true), "release the active reset-boundary lease");
        }
    }

    {
        GpuDataAccess secondCameraFill = staticData.getCacheData(secondCamera, makeCacheRequest(resources));
        test.expect(static_cast<bool>(secondCameraFill) && secondCameraFill.status() == CacheStatus::CacheFill, "camera ID participates in the cache key");
        test.expect(secondCameraFill.writableData() == persistentDeviceData, "composite-key eviction reuses the persistent cache allocation");
        const bool submitted = secondCameraFill.writableData() != nullptr && cudaMemsetAsync(secondCameraFill.writableData(), 0x30, secondCameraFill.bytes(), resources.stream) == cudaSuccess;
        test.expect(secondCameraFill.freeCacheData(submitted), "publish the second camera fill");
    }

    test.expect(staticData.resetCache(), "reset residency at a safe run boundary");
    {
        GpuDataAccess fillAfterReset = staticData.getCacheData(secondCamera, makeCacheRequest(resources));
        test.expect(static_cast<bool>(fillAfterReset) && fillAfterReset.status() == CacheStatus::CacheFill, "the same frame identity requires a new upload after reset");
        test.expect(fillAfterReset.writableData() == persistentDeviceData, "reset retains the device allocation");
        const bool submitted = fillAfterReset.writableData() != nullptr && cudaMemsetAsync(fillAfterReset.writableData(), 0x31, fillAfterReset.bytes(), resources.stream) == cudaSuccess;
        test.expect(fillAfterReset.freeCacheData(submitted), "publish the post-reset fill");
    }

    {
        GpuDataAccess unseenFill = staticData.getCacheData(arbitraryIncoming, makeCacheRequest(resources));
        test.expect(static_cast<bool>(unseenFill) && unseenFill.status() == CacheStatus::CacheFill, "cache an unregistered frame arriving after the reset boundary");
        test.expect(unseenFill.writableData() == persistentDeviceData, "unregistered-frame eviction reuses the persistent device allocation");
        const bool submitted = unseenFill.writableData() != nullptr && cudaMemsetAsync(unseenFill.writableData(), 0x33, unseenFill.bytes(), resources.stream) == cudaSuccess;
        test.expect(unseenFill.freeCacheData(submitted), "publish the unregistered incoming frame");
    }

    test.expect(releaseStaticData(staticData, location), "release run-boundary StaticData");
    test.expect(releaseAccessResources(resources), "release run-boundary fallback resources");
}

void testGpuCacheManagerLru(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    const FrameMetadata firstMetadata = makeFrameMetadata(10, runtime);
    const FrameMetadata secondMetadata = makeFrameMetadata(11, runtime);
    const FrameMetadata thirdMetadata = makeFrameMetadata(12, runtime);
    GpuCacheManager cache;
    TaskGpuResources resources;
    const bool initialized = initializeAccessResources(resources, location, runtime.inBytes) && cache.initialize({location.gpuId}, runtime.inBytes, 2);
    test.expect(initialized, "initialize two-entry cache for LRU test");
    if (!initialized) {
        cache.release();
        releaseAccessResources(resources);
        return;
    }

    auto fillFrame = [&cache, &resources](const FrameMetadata& metadata) {
        GpuDataAccess access = cache.getCacheData(metadata, makeCacheRequest(resources));
        if (!access || access.status() != CacheStatus::CacheFill) {
            return false;
        }
        const bool submitted = cudaMemsetAsync(access.writableData(), static_cast<int>(metadata.key.frameId), access.bytes(), resources.stream) == cudaSuccess;
        return access.freeCacheData(submitted);
    };

    test.expect(fillFrame(firstMetadata) && fillFrame(secondMetadata), "fill both cache entries");
    GpuDataAccess firstHit = cache.getCacheData(firstMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(firstHit) && firstHit.status() == CacheStatus::CacheHit && firstHit.freeCacheData(true), "touch first frame so second frame becomes LRU");
    test.expect(fillFrame(thirdMetadata), "third frame evicts one inactive cache entry");
    GpuDataAccess secondAgain = cache.getCacheData(secondMetadata, makeCacheRequest(resources));
    test.expect(static_cast<bool>(secondAgain) && secondAgain.status() == CacheStatus::CacheFill, "least-recently-used second frame was evicted");
    if (secondAgain) {
        test.expect(secondAgain.freeCacheData(true), "complete refill after LRU eviction");
    }

    bool churnSucceeded = true;
    FrameMetadata lastIncomingMetadata;
    for (std::uint64_t frameId = 1000; frameId < 1221; ++frameId) {
        lastIncomingMetadata = makeFrameMetadata(frameId, runtime, static_cast<std::uint32_t>(frameId % 13U));
        GpuDataAccess incomingFill = cache.getCacheData(lastIncomingMetadata, makeCacheRequest(resources));
        if (!incomingFill || incomingFill.status() != CacheStatus::CacheFill) {
            churnSucceeded = false;
            break;
        }
        const bool submitted = cudaMemsetAsync(incomingFill.writableData(), static_cast<int>(frameId & 0xffU), incomingFill.bytes(), resources.stream) == cudaSuccess;
        if (!incomingFill.freeCacheData(submitted)) {
            churnSucceeded = false;
            break;
        }
    }
    test.expect(churnSucceeded, "cache more than 220 unregistered incoming frames through two fixed entries");
    if (churnSucceeded) {
        GpuDataAccess lastIncomingHit = cache.getCacheData(lastIncomingMetadata, makeCacheRequest(resources));
        test.expect(static_cast<bool>(lastIncomingHit) && lastIncomingHit.status() == CacheStatus::CacheHit && lastIncomingHit.freeCacheData(true), "fixed residency table preserves the newest incoming frame after churn");
    }

    test.expect(cache.release(), "release LRU test cache");
    test.expect(releaseAccessResources(resources), "release LRU test resources");
}

}  // namespace gpuinfra_tests
