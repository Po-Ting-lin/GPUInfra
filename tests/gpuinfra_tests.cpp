#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "DataCache/GpuCacheManager.h"
#include "DataCache/GpuDataAccess.h"
#include "DataCache/GpuResidencyTable.h"
#include "DummyGraph.h"
#include "DummyTask.h"
#include "FrameCpuAtom.h"
#include "GpuContextManager.h"
#include "ImageSizing.h"
#include "NumaExecutor.h"
#include "ParameterRegistry.h"
#include "StaticData.h"
#include "TaskGpuResources.h"

namespace {

class TestContext {
public:
    void expect(bool condition, const std::string& description) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << description << '\n';
        }
    }

    int failureCount() const {
        return failures;
    }

private:
    int failures = 0;
};

AlgoRuntimeInfo makeRuntime(int factor) {
    AlgoRuntimeInfo runtime;
    runtime.sizeFactor = factor;
    runtime.frameW = ImageSizing::scaledDimension(factor, ImageSizing::INPUT_MULTIPLIER);
    runtime.frameH = runtime.frameW;
    runtime.inBytes = ImageSizing::squareBytes(runtime.frameW, sizeof(std::uint8_t));
    runtime.frameDtype = 1;
    return runtime;
}

FrameMetadata makeFrameMetadata(std::uint64_t frameId, const AlgoRuntimeInfo& runtime, std::uint32_t cameraId = 0) {
    FrameMetadata metadata;
    metadata.key.frameId = frameId;
    metadata.key.cameraId = cameraId;
    metadata.bytes = runtime.inBytes;
    metadata.width = runtime.frameW;
    metadata.height = runtime.frameH;
    metadata.dtype = runtime.frameDtype;
    return metadata;
}

bool loadTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.load(); });
}

// Test bodies run inside NumaExecutor, including registration and error cases.
bool configureTaskParameters(DummyTask& task, ParameterRegistry& registry) {
    return task.registerParameters(registry) && registry.setString(DummyTask::NAME_PARAMETER, "test") && registry.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && registry.seal();
}

bool startTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.start(); });
}

bool stopTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.stop(); });
}

bool notifyTask(DummyTask& task, const ParameterSnapshot& parameters, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task, &parameters] { return task.notifyParameters(parameters); });
}

bool unloadTask(DummyTask& task, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task] { return task.unload(); });
}

bool initializeStaticData(StaticData& staticData, const GpuLocation& location, const AlgoRuntimeInfo& runtime, std::size_t gpuCacheEntries = 4) {
    return NumaExecutor(location.numaNode).run([&staticData, &runtime, gpuCacheEntries] {
        StaticDataConfig config;
        config.runtime = runtime;
        config.gpuCacheEntries = gpuCacheEntries;
        return staticData.init(config);
    });
}

bool releaseStaticData(StaticData& staticData, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&staticData] { return staticData.release(); });
}

GpuCacheRequest makeCacheRequest(const TaskGpuResources& resources) {
    GpuCacheRequest request;
    request.gpuId = resources.gpuId;
    request.stream = resources.stream;
    request.d_fallback = resources.d_input;
    request.fallbackBytes = resources.inBytes;
    return request;
}

bool initializeAccessResources(TaskGpuResources& resources, const GpuLocation& location, std::size_t bytes) {
    resources.gpuId = location.gpuId;
    resources.inBytes = bytes;
    if (cudaSetDevice(location.gpuId) != cudaSuccess || cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&resources.d_input, bytes) != cudaSuccess) {
        cudaStreamDestroy(resources.stream);
        resources.stream = nullptr;
        return false;
    }
    return true;
}

bool releaseAccessResources(TaskGpuResources& resources) {
    bool ok = true;
    if (resources.stream != nullptr && cudaStreamSynchronize(resources.stream) != cudaSuccess) {
        ok = false;
    }
    if (resources.d_input != nullptr && cudaFree(resources.d_input) != cudaSuccess) {
        ok = false;
    }
    resources.d_input = nullptr;
    if (resources.stream != nullptr && cudaStreamDestroy(resources.stream) != cudaSuccess) {
        ok = false;
    }
    resources.stream = nullptr;
    resources.inBytes = 0;
    return ok;
}

bool executeTask(DummyTask& task, FrameCpuAtom& atom, StaticData& staticData, const GpuLocation& location) {
    return NumaExecutor(location.numaNode).run([&task, &atom, &staticData] { return staticData.execute() && task.execute(atom, staticData); });
}

std::uint32_t referenceValue(const FrameCpuAtom& atom, int dimension, int x, int y, int inputStride) {
    std::uint32_t sum = 0;
    for (int k = 0; k < dimension; ++k) {
        const std::uint32_t a = atom.data[static_cast<std::size_t>(y) * inputStride + k];
        const std::uint32_t b = atom.data[static_cast<std::size_t>(k) * inputStride + x];
        sum += a * b;
    }
    return sum;
}

bool verifyOutput(const FrameCpuAtom& atom, const AlgoOutput& output, int inputStride) {
    for (int y = 0; y < output.height; ++y) {
        for (int x = 0; x < output.width; ++x) {
            std::uint32_t actual = 0;
            const std::size_t offset = (static_cast<std::size_t>(y) * output.width + x) * sizeof(actual);
            std::memcpy(&actual, output.data.data() + offset, sizeof(actual));
            if (actual != referenceValue(atom, output.width, x, y, inputStride)) {
                return false;
            }
        }
    }
    return true;
}

void testParameterRegistry(TestContext& test) {
    ParameterRegistry registry;
    ParameterSnapshot snapshot;
    test.expect(registry.registerParameter("value", ParameterType::String), "register string parameter");
    test.expect(registry.registerParameter("value", ParameterType::String), "repeat matching schema registration");
    test.expect(!registry.registerParameter("value", ParameterType::Bytes), "reject schema type mismatch");
    test.expect(!registry.setBytes("value", {1}), "reject value type mismatch");
    test.expect(registry.setString("value", "ready"), "set registered string value");
    const std::uint64_t initialRevision = registry.revision();
    test.expect(registry.seal(), "seal complete registry");
    test.expect(registry.isSealed(), "registry exposes sealed schema state");
    test.expect(!registry.registerParameter("late", ParameterType::String), "reject schema registration after seal");
    test.expect(registry.snapshot(snapshot) && snapshot.revision() == initialRevision, "snapshot captures the initial parameter revision");
    std::string value;
    test.expect(snapshot.getString("value", value) && value == "ready", "read typed snapshot value");
    std::vector<std::uint8_t> bytes;
    test.expect(!snapshot.getBytes("value", bytes), "reject snapshot type mismatch");
    test.expect(registry.setString("value", "ready") && registry.revision() == initialRevision, "same value preserves parameter revision");
    test.expect(registry.setString("value", "changed") && registry.revision() == initialRevision + 1, "changed value advances parameter revision after schema seal");
    test.expect(registry.snapshot(snapshot) && snapshot.revision() == initialRevision + 1 && snapshot.getString("value", value) && value == "changed", "updated snapshot exposes changed parameter value");
}

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

void testStaticDataValidation(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    const GraphConfig defaultGraphConfig;
    test.expect(defaultGraphConfig.gpuCacheEntries == 4, "graph cache capacity defaults to four entries");

    const FrameMetadata arbitraryFrame = makeFrameMetadata(900001, runtime, 27);
    const FrameMetadata anotherArbitraryFrame = makeFrameMetadata(17, runtime, 81);
    StaticData staticData;
    test.expect(initializeStaticData(staticData, location, runtime, 4), "initialize StaticData without a frame registration list");
    test.expect(staticData.isInitialized() && staticData.gpuCacheEntryCount() == 4, "cache capacity is independent from incoming frame count");
    test.expect(staticData.validateFrame(arbitraryFrame) && staticData.validateFrame(anotherArbitraryFrame), "accept arbitrary incoming keys with the fixed layout");

    FrameMetadata wrongLayout = arbitraryFrame;
    ++wrongLayout.width;
    test.expect(!staticData.validateFrame(wrongLayout), "reject an arbitrary frame with the wrong fixed layout");
    test.expect(releaseStaticData(staticData, location), "release registry-free StaticData cache");
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

void testFrameDataAcrossTaskInstances(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask firstTask(90, ExecutionModel::Batched, runtime);
    DummyTask secondTask(91, ExecutionModel::Batched, runtime);
    ParameterRegistry parameters;
    const bool parametersReady = firstTask.registerParameters(parameters) && secondTask.registerParameters(parameters) && parameters.setString(DummyTask::NAME_PARAMETER, "test") && parameters.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && parameters.seal();
    const bool tasksReady = parametersReady && loadTask(firstTask, location) && loadTask(secondTask, location) && startTask(firstTask, location) && startTask(secondTask, location);
    test.expect(tasksReady, "prepare two task instances for frame GPU continuity");

    const FrameMetadata metadata = makeFrameMetadata(29, runtime);
    FrameCpuAtom atom(metadata, runtime);
    StaticData staticData;
    const bool frameReady = initializeStaticData(staticData, location, runtime, 1);
    const bool frameAccepted = staticData.validateFrame(metadata);
    test.expect(frameReady && frameAccepted, "initialize shared StaticData frame GPU data");
    test.expect(staticData.isInitialized() && staticData.gpuCacheEntryCount() == 1, "StaticData accepts incoming frames independently from cache capacity");

    bool firstSucceeded = false;
    std::vector<AlgoOutput> firstOutputs;
    if (tasksReady && frameReady && frameAccepted) {
        firstSucceeded = executeTask(firstTask, atom, staticData, location);
        firstOutputs = atom.result.outputs;
    }
    test.expect(firstSucceeded, "first task instance uploads and processes the frame");

    std::fill(atom.data.begin(), atom.data.end(), 0);
    const bool secondSucceeded = tasksReady && frameReady && frameAccepted && executeTask(secondTask, atom, staticData, location);
    test.expect(secondSucceeded, "second task instance consumes existing frame GPU data");

    bool outputsMatch = firstSucceeded && firstOutputs.size() == atom.result.outputs.size();
    for (std::size_t index = 0; outputsMatch && index < firstOutputs.size(); ++index) {
        const AlgoOutput& expected = firstOutputs[index];
        const AlgoOutput& actual = atom.result.outputs[index];
        outputsMatch = expected.algoName == actual.algoName && expected.width == actual.width && expected.height == actual.height && expected.data == actual.data;
    }
    test.expect(outputsMatch, "second task reads frame-owned GPU data instead of modified host input");

    const bool firstTaskStopped = stopTask(firstTask, location);
    const bool secondTaskStopped = stopTask(secondTask, location);
    test.expect(firstTaskStopped && secondTaskStopped, "stop both frame continuity task instances");
    test.expect(releaseStaticData(staticData, location), "release shared StaticData frame GPU data");
    const bool firstTaskUnloaded = unloadTask(firstTask, location);
    const bool secondTaskUnloaded = unloadTask(secondTask, location);
    test.expect(firstTaskUnloaded && secondTaskUnloaded, "unload both frame continuity task instances");
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

void testTaskFallbackExecution(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask task(92, ExecutionModel::Batched, runtime);
    ParameterRegistry parameters;
    const bool taskReady = configureTaskParameters(task, parameters) && loadTask(task, location) && startTask(task, location);
    test.expect(taskReady, "prepare task for zero-capacity fallback execution");

    const FrameMetadata metadata = makeFrameMetadata(30, runtime);
    FrameCpuAtom atom(metadata, runtime);
    StaticData staticData;
    const bool frameReady = initializeStaticData(staticData, location, runtime, 0);
    const bool frameAccepted = staticData.validateFrame(metadata);
    test.expect(frameReady && frameAccepted && staticData.gpuCacheEntryCount() == 0, "initialize arbitrary-frame input with GPU cache disabled");

    const bool succeeded = taskReady && frameReady && executeTask(task, atom, staticData, location);
    test.expect(succeeded, "task executes correctly through its fallback input buffer");
    if (frameAccepted) {
        for (const AlgoOutput& output : atom.result.outputs) {
            test.expect(verifyOutput(atom, output, runtime.frameW), "fallback execution matches CPU reference");
        }
    }

    test.expect(stopTask(task, location), "stop fallback execution task");
    test.expect(releaseStaticData(staticData, location), "release zero-capacity StaticData");
    test.expect(unloadTask(task, location), "unload fallback execution task");
}

void testLifecycleAndResults(TestContext& test, const GpuLocation& location, ExecutionModel model, int taskId) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask task(taskId, model, runtime);
    ParameterRegistry parameters;
    ParameterSnapshot parameterSnapshot;
    const FrameMetadata prematureMetadata = makeFrameMetadata(1, runtime);
    const FrameMetadata firstMetadata = makeFrameMetadata(7, runtime);
    const FrameMetadata secondMetadata = makeFrameMetadata(19, runtime);
    const FrameMetadata malformedMetadata = makeFrameMetadata(23, runtime);
    FrameMetadata mismatchedAtomMetadata = makeFrameMetadata(24, runtime);
    ++mismatchedAtomMetadata.dtype;
    const FrameMetadata mismatchedRecordMetadata = makeFrameMetadata(25, runtime);
    FrameCpuAtom prematureAtom(prematureMetadata, runtime);
    test.expect(prematureAtom.result.id == prematureMetadata.key.frameId && prematureAtom.result.ok && prematureAtom.result.outputs.size() == 3, "FrameCpuAtom owns a preallocated result");
    test.expect(!loadTask(task, location), "reject load before parameter registration");
    test.expect(!task.notifyParameters(parameterSnapshot), "reject notifyParameters before registration");
    test.expect(task.registerParameters(parameters), "register parameters immediately after construction");
    test.expect(task.lifecycle() == TaskLifecycle::Registered, "task reaches registered lifecycle state before load");
    test.expect(!task.registerParameters(parameters), "reject repeated parameter registration");
    test.expect(!loadTask(task, location), "reject load before initial parameters are complete");
    test.expect(parameters.setString(DummyTask::NAME_PARAMETER, "test") && parameters.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && parameters.seal() && parameters.snapshot(parameterSnapshot), "define initial parameters before load");
    test.expect(task.gpuId() == -1, "task has no GPU binding before load");
    test.expect(loadTask(task, location), "load task resources");
    test.expect(task.gpuId() == location.gpuId, "load derives GPU binding from the framework NUMA environment");
    test.expect(task.lifecycle() == TaskLifecycle::Loaded, "load applies initial parameters without a notify callback");

    StaticData staticData;
    test.expect(initializeStaticData(staticData, location, runtime), "initialize task StaticData cache");
    FrameMetadata wrongLayoutMetadata = firstMetadata;
    ++wrongLayoutMetadata.width;
    test.expect(staticData.validateFrame(firstMetadata), "accept an arbitrary frame with the configured layout");
    test.expect(!staticData.validateFrame(wrongLayoutMetadata), "reject an arbitrary frame with mismatched metadata");
    test.expect(staticData.execute() && !task.execute(prematureAtom, staticData), "reject execute before start");
    test.expect(!task.load(), "reject repeated load");
    test.expect(!stopTask(task, location), "reject stop before start");
    test.expect(startTask(task, location), "start task execution cycle");
    test.expect(task.lifecycle() == TaskLifecycle::Started, "task reaches started lifecycle state");
    test.expect(!startTask(task, location), "reject repeated task start");

    const std::uint64_t initialParameterRevision = parameters.revision();
    test.expect(parameters.setString(DummyTask::NAME_PARAMETER, "test") && parameters.revision() == initialParameterRevision && parameters.snapshot(parameterSnapshot) && notifyTask(task, parameterSnapshot, location), "same parameter values preserve the applied revision");
    test.expect(parameters.setString(DummyTask::NAME_PARAMETER, "changed") && parameters.revision() == initialParameterRevision + 1 && parameters.snapshot(parameterSnapshot) && notifyTask(task, parameterSnapshot, location), "changed parameters notify a started task at a quiescent boundary");

    FrameCpuAtom firstAtom(firstMetadata, runtime);
    FrameCpuAtom secondAtom(secondMetadata, runtime);
    const bool firstFrameAccepted = staticData.validateFrame(firstMetadata);
    const bool secondFrameAccepted = staticData.validateFrame(secondMetadata);
    test.expect(firstFrameAccepted && secondFrameAccepted, "accept unrelated incoming task frames without registration");
    bool firstSucceeded = false;
    bool secondSucceeded = false;
    std::thread::id firstThreadId;
    std::thread::id secondThreadId;
    std::mutex sequenceLock;
    std::condition_variable sequenceCondition;
    int turn = 0;
    std::thread firstWorker = NumaExecutor(location.numaNode).start([&](bool numaReady) {
        firstThreadId = std::this_thread::get_id();
        if (numaReady) {
            firstSucceeded = staticData.execute() && task.execute(firstAtom, staticData);
        }
        {
            std::lock_guard<std::mutex> guard(sequenceLock);
            turn = 1;
        }
        sequenceCondition.notify_one();
    });
    std::thread secondWorker = NumaExecutor(location.numaNode).start([&](bool numaReady) {
        secondThreadId = std::this_thread::get_id();
        {
            std::unique_lock<std::mutex> guard(sequenceLock);
            sequenceCondition.wait(guard, [&turn] { return turn == 1; });
        }
        if (numaReady) {
            secondSucceeded = staticData.execute() && task.execute(secondAtom, staticData);
        }
    });
    firstWorker.join();
    secondWorker.join();

    test.expect(firstThreadId != secondThreadId, "mobility test uses distinct live host threads");
    test.expect(firstSucceeded && secondSucceeded, "one task executes sequential frames on different threads");
    if (firstFrameAccepted) {
        for (const AlgoOutput& output : firstAtom.result.outputs) {
            test.expect(verifyOutput(firstAtom, output, runtime.frameW), "first frame matches CPU reference");
        }
    }
    if (secondFrameAccepted) {
        for (const AlgoOutput& output : secondAtom.result.outputs) {
            test.expect(verifyOutput(secondAtom, output, runtime.frameW), "second frame matches CPU reference");
        }
    }

    FrameCpuAtom malformedAtom(malformedMetadata, runtime);
    const bool malformedFrameAccepted = staticData.validateFrame(malformedMetadata);
    test.expect(malformedFrameAccepted, "accept metadata before the CPU input is malformed");
    malformedAtom.data.pop_back();
    test.expect(staticData.execute() && !task.execute(malformedAtom, staticData), "reject malformed frame input");
    test.expect(malformedFrameAccepted && !malformedAtom.result.ok, "malformed atom records failed task status");

    FrameCpuAtom mismatchedAtom(mismatchedAtomMetadata, runtime);
    const bool unrelatedFrameAccepted = staticData.validateFrame(mismatchedRecordMetadata);
    test.expect(unrelatedFrameAccepted, "accept unrelated metadata with the configured layout");
    test.expect(staticData.execute() && !task.execute(mismatchedAtom, staticData), "reject an atom with a mismatched dtype");
    test.expect(!mismatchedAtom.result.ok && staticData.validateFrame(mismatchedRecordMetadata), "wrong-layout frame fails without affecting other incoming frames");

    test.expect(stopTask(task, location), "stop task execution cycle");
    test.expect(task.lifecycle() == TaskLifecycle::Stopped, "task reaches stopped lifecycle state");
    test.expect(!task.execute(prematureAtom, staticData), "reject execute after stop");
    test.expect(releaseStaticData(staticData, location), "release task StaticData pool");
    test.expect(unloadTask(task, location), "unload task resources");
    test.expect(task.unload(), "repeated unload is harmless");
    test.expect(task.lifecycle() == TaskLifecycle::Unloaded && task.gpuId() == -1, "unload clears the task GPU binding");
}

bool runGraphPhase(DummyGraph& graph, FramePhase phase) {
    PhaseGate gate;
    if (!graph.startPhase(phase, gate)) {
        gate.release();
        return false;
    }
    gate.release();
    return graph.waitForPhase();
}

GraphConfig makeGraphConfig(std::size_t tasks, std::size_t workers, ExecutionModel model) {
    GraphConfig config;
    config.taskInstancesPerGpu = tasks;
    config.graphThreads = workers;
    config.warmupFramesPerGpu = 0;
    config.timedFramesPerGpu = 8;
    config.executionModel = model;
    config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    config.parameters.name = "graph-test";
    return config;
}

void testGpuTopologySelection(TestContext& test) {
    const std::vector<GpuLocation> locations = {{7, 2}, {3, 0}, {8, 2}, {5, 1}};
    std::vector<int> gpuIds = {99};
    test.expect(!GpuTopology::resolveGpuIds(3, locations, gpuIds) && gpuIds.empty(), "reject a NUMA node with zero GPUs and clear stale results");
    test.expect(!GpuTopology::resolveGpuIds(2, locations, gpuIds) && gpuIds.empty(), "reject multiple GPUs instead of selecting the first");
    test.expect(GpuTopology::resolveGpuIds(1, locations, gpuIds) && gpuIds == std::vector<int>{5}, "resolve the sole local GPU without equating GPU and NUMA IDs");
    test.expect(GpuTopology::resolveGpuIds(0, locations, gpuIds) && gpuIds == std::vector<int>{3}, "ignore GPUs belonging to other NUMA nodes");
    test.expect(!GpuTopology::resolveGpuIds(-1, locations, gpuIds) && gpuIds.empty(), "reject failed NUMA detection");
    test.expect(!GpuTopology::resolveGpuIds(0, {}, gpuIds) && gpuIds.empty(), "reject an empty GPU topology");
    test.expect(!GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds) && gpuIds.empty(), "reject lookup before infrastructure initialization");
}

void testNumaExecutionBoundary(TestContext& test, const GpuLocation& location) {
    const NumaExecutor executor(location.numaNode);
    test.expect(executor.run([&location] {
        std::vector<int> gpuIds;
        return GpuTopology::currentNumaNode() == location.numaNode && GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds) && gpuIds == std::vector<int>{location.gpuId};
    }), "framework affinity establishes the NUMA identity used for GPU lookup");

    bool callbackCalled = false;
    test.expect(!NumaExecutor(-1).run([&callbackCalled] { callbackCalled = true; return true; }) && !callbackCalled, "failed framework affinity never invokes a task callback");
    GraphSink sink;
    std::atomic<bool> cancellation{false};
    GraphConfig config = makeGraphConfig(1, 1, ExecutionModel::Batched);
    DummyGraph graph(NumaExecutor(-1), config, sink, cancellation);
    test.expect(!graph.initialize(), "reject a graph without a valid NUMA execution environment");
    test.expect(graph.taskCount() == 0 && !cancellation.load(std::memory_order_acquire) && sink.count() == 0 && graph.shutdown(), "NUMA rejection occurs before task allocation or global cancellation");

    GraphConfig defaultWorkers = makeGraphConfig(3, 0, ExecutionModel::Batched);
    DummyGraph sizedGraph(executor, defaultWorkers, sink, cancellation);
    test.expect(sizedGraph.initialize() && sizedGraph.taskCount() == 3 && sizedGraph.workerCount() == 3, "derive task and default worker counts from resolved GPU count");
    test.expect(sizedGraph.shutdown(), "release graph with derived task and worker counts");
}

void testIndependentPools(TestContext& test, const GpuLocation& location) {
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        GraphConfig config = makeGraphConfig(1, 2, ExecutionModel::Batched);
        config.warmupFramesPerGpu = 1;
        DummyGraph graph(NumaExecutor(location.numaNode), config, sink, cancellation);
        test.expect(graph.initialize(), "initialize one-task two-worker graph");
        PhaseGate beforeStartGate;
        test.expect(!graph.startPhase(FramePhase::Warmup, beforeStartGate), "reject frame submission before graph start");
        beforeStartGate.release();
        AlgoParams loadedParameters = config.parameters;
        loadedParameters.blob = {4, 5, 6};
        test.expect(graph.changeParameters(loadedParameters), "notify changed parameters while graph tasks are loaded and quiescent");
        test.expect(graph.start(), "start one graph execution cycle");
        test.expect(graph.changeParameters(loadedParameters), "same graph parameters require no task notification");
        PhaseGate invalidPhaseGate;
        test.expect(!graph.startPhase(static_cast<FramePhase>(99), invalidPhaseGate), "reject an invalid graph-owned frame phase");
        invalidPhaseGate.release();

        PhaseGate warmupGate;
        test.expect(graph.startPhase(FramePhase::Warmup, warmupGate), "submit warmup inside the started execution cycle");
        AlgoParams changedParameters = loadedParameters;
        changedParameters.name = "graph-test-updated";
        test.expect(!graph.changeParameters(changedParameters), "reject parameter changes while a phase is active");
        test.expect(!graph.stop(), "reject graph stop while a phase is active");
        warmupGate.release();
        test.expect(graph.waitForPhase(), "finish warmup before changing parameters");
        test.expect(graph.changeParameters(changedParameters), "notify changed parameters at a quiescent phase boundary");

        test.expect(runGraphPhase(graph, FramePhase::Timed), "run one-task two-worker graph");
        PhaseGate repeatedPhaseGate;
        test.expect(!graph.startPhase(FramePhase::Timed, repeatedPhaseGate), "graph-owned phase state rejects repeated submission");
        repeatedPhaseGate.release();
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "DummyGraph free-task pool prevents concurrent reuse of its sole task instance");
        test.expect(sink.count() == 9 && sink.failureCount() == 0, "one-task graph completes every frame once across both phases");
        test.expect(graph.stop(), "stop graph after all frame executions finish");
        AlgoParams stoppedParameters = changedParameters;
        stoppedParameters.name = "graph-test-stopped";
        test.expect(graph.changeParameters(stoppedParameters), "notify changed parameters after the execution cycle stops and before unload");
        test.expect(!graph.stop(), "reject repeated graph stop");
        test.expect(graph.shutdown(), "shutdown one-task graph");
    }
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        DummyGraph graph(NumaExecutor(location.numaNode), makeGraphConfig(2, 1, ExecutionModel::Interleaved), sink, cancellation);
        test.expect(graph.initialize(), "initialize two-task one-worker graph");
        test.expect(graph.start(), "start two-task one-worker graph");
        test.expect(runGraphPhase(graph, FramePhase::Timed), "run two-task one-worker graph");
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "worker pool limits concurrency independently");
        test.expect(sink.count() == 8 && sink.failureCount() == 0, "one-worker graph completes every frame once");
        test.expect(graph.stop(), "stop two-task one-worker graph");
        test.expect(graph.shutdown(), "shutdown one-worker graph");
    }
}

void testConditionalNumaGraphs(TestContext& test, const std::vector<GpuLocation>& locations) {
    std::map<int, std::vector<int>> grouped;
    for (const GpuLocation& location : locations) {
        grouped[location.numaNode].push_back(location.gpuId);
    }
    if (grouped.size() < 2) {
        std::cout << "SKIP: multi-NUMA graph test requires GPUs on at least two NUMA nodes\n";
        return;
    }
    for (const auto& entry : grouped) {
        if (entry.second.size() != 1) {
            std::cout << "SKIP: temporary topology supports exactly one GPU in each NUMA graph copy\n";
            return;
        }
    }

    GraphSink sink;
    std::atomic<bool> cancellation{false};
    std::vector<std::unique_ptr<DummyGraph>> graphs;
    for (const auto& entry : grouped) {
        GraphConfig config;
        config.taskInstancesPerGpu = 1;
        config.graphThreads = entry.second.size();
        config.timedFramesPerGpu = 1;
        config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
        config.parameters.name = "numa-test";
        graphs.push_back(std::make_unique<DummyGraph>(NumaExecutor(entry.first), config, sink, cancellation));
    }
    bool ok = true;
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->initialize() && ok;
    }
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->start() && ok;
    }
    PhaseGate gate;
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->startPhase(FramePhase::Timed, gate) && ok;
    }
    gate.release();
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->waitForPhase() && ok;
    }
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->stop() && ok;
    }
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->shutdown() && ok;
    }
    test.expect(ok && sink.count() == locations.size(), "one graph copy runs on each GPU-bearing NUMA node");
}

void testGraphCancellation(TestContext& test, const GpuLocation& location) {
    GraphSink sink;
    std::atomic<bool> cancellation{false};
    GraphConfig config = makeGraphConfig(1, 2, static_cast<ExecutionModel>(99));
    config.warmupFramesPerGpu = 2;
    config.timedFramesPerGpu = 4;
    DummyGraph graph(NumaExecutor(location.numaNode), config, sink, cancellation);
    test.expect(graph.initialize(), "initialize graph used for failure propagation");
    test.expect(graph.start(), "start graph used for failure propagation");
    test.expect(!runGraphPhase(graph, FramePhase::Warmup), "execution failure fails graph phase");
    test.expect(cancellation.load(std::memory_order_acquire), "execution failure raises global cancellation");
    test.expect(sink.count() == 2 && sink.failureCount() == 2, "active failed phase gives every frame one terminal result");
    test.expect(graph.stop(), "failed graph still stops after in-flight execution ends");
    test.expect(graph.shutdown(), "failed graph still unloads cleanly");
    test.expect(sink.count() == 6 && sink.failureCount() == 6, "shutdown cancels every preallocated future frame exactly once");
}

}  // namespace

int main() {
    TestContext test;
    testParameterRegistry(test);
    testGpuResidencyTable(test);
    testGpuTopologySelection(test);

    GpuInfraConfig config;
    config.requireNuma = true;
    if (!GpuContextManager::init(config)) {
        std::cerr << "FAIL: CUDA/NUMA infrastructure initialization\n";
        return 1;
    }
    const std::vector<GpuLocation> locations = GpuContextManager::gpuLocations();
    if (locations.empty()) {
        std::cerr << "FAIL: no CUDA GPU locations\n";
        GpuContextManager::shutdown();
        return 1;
    }

    const bool callbacksRan = NumaExecutor(locations.front().numaNode).run([&test, &locations] {
        testLifecycleAndResults(test, locations.front(), ExecutionModel::Batched, 1);
        testLifecycleAndResults(test, locations.front(), ExecutionModel::Interleaved, 2);
        testStaticDataValidation(test, locations.front());
        testGpuDataAccessState(test, locations.front());
        testIndependentPayloadCaches(test, locations.front(), 0);
        testIndependentPayloadCaches(test, locations.front(), 1);
        testConcurrentCacheRequests(test, locations.front());
        testGpuCacheResetBoundaries(test, locations.front());
        testGpuCacheManagerLru(test, locations.front());
        testFrameDataAcrossTaskInstances(test, locations.front());
        testTaskFallbackExecution(test, locations.front());
        return true;
    });
    test.expect(callbacksRan, "all direct task callbacks execute inside the graph NUMA environment");
    testNumaExecutionBoundary(test, locations.front());
    testIndependentPools(test, locations.front());
    testConditionalNumaGraphs(test, locations);
    testGraphCancellation(test, locations.front());

    GpuContextManager::shutdown();
    if (test.failureCount() != 0) {
        std::cerr << test.failureCount() << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All GPUInfra tests passed\n";
    return 0;
}
