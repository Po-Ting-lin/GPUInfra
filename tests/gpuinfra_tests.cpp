#include <algorithm>
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

#include "DummyGraph.h"
#include "DummyTask.h"
#include "FrameCpuAtom.h"
#include "GpuCacheManager.h"
#include "GpuContextManager.h"
#include "GpuDataAccess.h"
#include "ImageSizing.h"
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

FrameMetadata makeFrameMetadata(std::uint64_t frameId, const AlgoRuntimeInfo& runtime) {
    FrameMetadata metadata;
    metadata.id = frameId;
    metadata.bytes = runtime.inBytes;
    metadata.width = runtime.frameW;
    metadata.height = runtime.frameH;
    metadata.dtype = runtime.frameDtype;
    return metadata;
}

bool loadTask(DummyTask& task, const GpuLocation& location) {
    bool loaded = false;
    std::thread setup([&task, &location, &loaded] {
        loaded = GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode) && task.load();
    });
    setup.join();
    return loaded;
}

bool notifyTask(DummyTask& task) {
    ParameterRegistry registry;
    ParameterSnapshot snapshot;
    return task.registerParameters(registry) && registry.setString(DummyTask::NAME_PARAMETER, "test") && registry.setBytes(DummyTask::BLOB_PARAMETER, {1, 2, 3}) && registry.seal() && registry.snapshot(snapshot) && task.notifyParameters(snapshot);
}

bool unloadTask(DummyTask& task, const GpuLocation& location) {
    bool unloaded = false;
    std::thread teardown([&task, &location, &unloaded] {
        unloaded = GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode) && task.unload();
    });
    teardown.join();
    return unloaded;
}

bool initializeStaticData(StaticData& staticData, const GpuLocation& location, const AlgoRuntimeInfo& runtime, const std::vector<FrameMetadata>& frames, std::size_t gpuCacheEntries = 4) {
    bool initialized = false;
    std::thread setup([&staticData, &location, &runtime, &frames, gpuCacheEntries, &initialized] {
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            StaticDataConfig config;
            config.gpuIds = {location.gpuId};
            config.runtime = runtime;
            config.frames = frames;
            config.gpuCacheEntries = gpuCacheEntries;
            initialized = staticData.init(config);
        }
    });
    setup.join();
    return initialized;
}

bool releaseStaticData(StaticData& staticData, const GpuLocation& location) {
    bool released = false;
    std::thread teardown([&staticData, &location, &released] {
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            released = staticData.release();
        }
    });
    teardown.join();
    return released;
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
    bool succeeded = false;
    std::thread worker([&task, &atom, &staticData, &location, &succeeded] {
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            succeeded = staticData.execute() && task.execute(atom, staticData);
        }
    });
    worker.join();
    return succeeded;
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
    test.expect(registry.seal(), "seal complete registry");
    test.expect(registry.isSealed(), "registry exposes sealed state");
    test.expect(!registry.setString("value", "changed"), "reject update after seal");
    test.expect(!registry.registerParameter("late", ParameterType::String), "reject registration after seal");
    test.expect(registry.snapshot(snapshot), "create immutable snapshot");
    std::string value;
    test.expect(snapshot.getString("value", value) && value == "ready", "read typed snapshot value");
    std::vector<std::uint8_t> bytes;
    test.expect(!snapshot.getBytes("value", bytes), "reject snapshot type mismatch");
}

void testStaticDataValidation(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    const GraphConfig defaultGraphConfig;
    test.expect(defaultGraphConfig.gpuCacheEntries == 4, "graph cache capacity defaults to four entries");

    std::vector<FrameMetadata> manyFrames;
    manyFrames.reserve(221);
    for (std::size_t index = 0; index < 221; ++index) {
        manyFrames.push_back(makeFrameMetadata(static_cast<std::uint64_t>(index), runtime));
    }

    StaticData manyFrameData;
    test.expect(initializeStaticData(manyFrameData, location, runtime, manyFrames, 2), "accept more than 220 registered frames with a bounded cache");
    test.expect(manyFrameData.frameCount() == manyFrames.size() && manyFrameData.gpuCacheEntryCount() == 2, "registered frame count is independent from cache capacity");
    test.expect(manyFrameData.validateFrame(manyFrames.back()), "validate a registered sparse frame through the immutable index");
    test.expect(releaseStaticData(manyFrameData, location), "release large registered frame set and bounded cache");

    const FrameMetadata duplicateMetadata = makeFrameMetadata(41, runtime);
    StaticData duplicateData;
    test.expect(!initializeStaticData(duplicateData, location, runtime, {duplicateMetadata, duplicateMetadata}), "reject duplicate StaticData frame IDs");
    test.expect(duplicateData.frameCount() == 0 && duplicateData.gpuCacheEntryCount() == 0 && duplicateData.release(), "duplicate rejection leaves StaticData empty");

    StaticData smallData;
    test.expect(initializeStaticData(smallData, location, runtime, {duplicateMetadata}, 4), "initialize fewer frames than requested cache entries");
    test.expect(smallData.frameCount() == 1 && smallData.gpuCacheEntryCount() == 1, "effective cache capacity is limited by configured frame count");
    test.expect(releaseStaticData(smallData, location), "release capacity-limited cache");
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
        GpuDataAccess firstFill = cache.acquire(firstMetadata, resources);
        test.expect(static_cast<bool>(firstFill) && firstFill.source() == GpuDataAccessSource::CacheFill && firstFill.needsUpload(), "first miss reserves a cache fill");
        cacheDeviceData = firstFill.writableData();
        test.expect(cacheDeviceData != nullptr && firstFill.data() == cacheDeviceData, "cache fill exposes the preallocated device pointer");

        GpuDataAccess loadingFallback = cache.acquire(firstMetadata, resources);
        test.expect(static_cast<bool>(loadingFallback) && loadingFallback.source() == GpuDataAccessSource::TaskFallback && loadingFallback.needsUpload(), "same frame loading uses task fallback without waiting");
        test.expect(loadingFallback.writableData() == resources.d_input, "loading fallback uses the task-private input buffer");
        const bool fallbackMemset = cudaMemsetAsync(loadingFallback.writableData(), 0x19, loadingFallback.bytes(), resources.stream) == cudaSuccess;
        test.expect(loadingFallback.complete(fallbackMemset), "complete loading fallback without publishing cache state");

        const bool fillMemset = cudaMemsetAsync(firstFill.writableData(), 0x2a, firstFill.bytes(), resources.stream) == cudaSuccess;
        test.expect(firstFill.complete(fillMemset), "publish cache fill after stream synchronization");
    }

    FrameMetadata mismatchedMetadata = firstMetadata;
    ++mismatchedMetadata.width;
    GpuDataAccess mismatchedAccess = cache.acquire(mismatchedMetadata, resources);
    test.expect(!mismatchedAccess, "reject the same frame ID with different metadata");

    GpuDataAccess failedHit = cache.acquire(firstMetadata, resources);
    test.expect(static_cast<bool>(failedHit) && failedHit.source() == GpuDataAccessSource::CacheHit && !failedHit.complete(false), "failed cache reader releases its lease without publishing state");
    GpuDataAccess hitAfterFailure = cache.acquire(firstMetadata, resources);
    test.expect(static_cast<bool>(hitAfterFailure) && hitAfterFailure.source() == GpuDataAccessSource::CacheHit && hitAfterFailure.complete(true), "failed cache reader preserves the immutable cached payload");

    GpuDataAccess activeHit = cache.acquire(firstMetadata, resources);
    test.expect(static_cast<bool>(activeHit) && activeHit.source() == GpuDataAccessSource::CacheHit && !activeHit.needsUpload(), "valid matching entry returns a cache hit");
    test.expect(activeHit.data() == cacheDeviceData && activeHit.writableData() == nullptr, "cache hit is immutable and reuses the cached pointer");
    GpuDataAccess secondReader = cache.acquire(firstMetadata, resources);
    test.expect(static_cast<bool>(secondReader) && secondReader.source() == GpuDataAccessSource::CacheHit && secondReader.complete(true), "matching immutable cache readers may coexist");

    GpuDataAccess busyFallback = cache.acquire(secondMetadata, resources);
    test.expect(static_cast<bool>(busyFallback) && busyFallback.source() == GpuDataAccessSource::TaskFallback, "all active cache entries force task fallback");
    test.expect(busyFallback.complete(true), "complete busy-cache fallback");
    test.expect(!cache.release(), "cache release rejects an active reader");
    test.expect(activeHit.complete(true), "release active cache reader");

    {
        GpuDataAccess abandonedFill = cache.acquire(secondMetadata, resources);
        test.expect(static_cast<bool>(abandonedFill) && abandonedFill.source() == GpuDataAccessSource::CacheFill, "inactive LRU entry can be reused for another frame");
    }
    GpuDataAccess retryAfterAbort = cache.acquire(secondMetadata, resources);
    test.expect(static_cast<bool>(retryAfterAbort) && retryAfterAbort.source() == GpuDataAccessSource::CacheFill && retryAfterAbort.writableData() == cacheDeviceData, "aborted fill returns the same allocation to the cache");
    test.expect(!retryAfterAbort.complete(false), "failed fill is not published");

    GpuDataAccess successfulRetry = cache.acquire(secondMetadata, resources);
    test.expect(static_cast<bool>(successfulRetry) && successfulRetry.source() == GpuDataAccessSource::CacheFill, "failed fill leaves an empty reusable entry");
    if (successfulRetry) {
        const bool memsetSucceeded = cudaMemsetAsync(successfulRetry.writableData(), 0x17, successfulRetry.bytes(), resources.stream) == cudaSuccess;
        test.expect(successfulRetry.complete(memsetSucceeded), "successful retry publishes the new frame");
    }

    TaskGpuResources wrongGpuResources = resources;
    wrongGpuResources.gpuId = location.gpuId + 1;
    GpuDataAccess wrongGpuRead = cache.acquire(secondMetadata, wrongGpuResources);
    test.expect(!wrongGpuRead, "reject access from a GPU without a replica");

    test.expect(cache.release(), "release bounded frame GPU cache");
    test.expect(!cache.isInitialized() && cache.entryCount() == 0, "cache release resets allocation state");
    test.expect(cache.release(), "repeated cache release is harmless");

    GpuCacheManager zeroCapacityCache;
    test.expect(zeroCapacityCache.initialize({location.gpuId}, runtime.inBytes, 0), "initialize a zero-capacity cache");
    GpuDataAccess zeroCapacityAccess = zeroCapacityCache.acquire(firstMetadata, resources);
    test.expect(static_cast<bool>(zeroCapacityAccess) && zeroCapacityAccess.source() == GpuDataAccessSource::TaskFallback && zeroCapacityAccess.writableData() == resources.d_input, "zero cache capacity always uses task fallback");
    test.expect(!zeroCapacityCache.release(), "cache release rejects an active fallback lease");
    test.expect(zeroCapacityAccess.complete(true), "complete zero-capacity fallback");
    {
        GpuDataAccess abandonedFallback = zeroCapacityCache.acquire(firstMetadata, resources);
        test.expect(static_cast<bool>(abandonedFallback) && abandonedFallback.source() == GpuDataAccessSource::TaskFallback, "acquire fallback used for RAII abort");
    }
    test.expect(zeroCapacityCache.release(), "release zero-capacity cache");
    test.expect(releaseAccessResources(resources), "release frame cache test resources");
}

void testFrameDataAcrossTaskInstances(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask firstTask(90, location.gpuId, ExecutionModel::Batched, runtime);
    DummyTask secondTask(91, location.gpuId, ExecutionModel::Batched, runtime);
    const bool tasksReady = loadTask(firstTask, location) && loadTask(secondTask, location) && notifyTask(firstTask) && notifyTask(secondTask);
    test.expect(tasksReady, "prepare two task instances for frame GPU continuity");

    const FrameMetadata metadata = makeFrameMetadata(29, runtime);
    FrameCpuAtom atom(metadata, runtime);
    StaticData staticData;
    const bool frameReady = initializeStaticData(staticData, location, runtime, {metadata}, 1);
    const bool frameRegistered = staticData.validateFrame(metadata);
    test.expect(frameReady && frameRegistered, "initialize shared StaticData frame GPU data");
    test.expect(staticData.frameCount() == 1 && staticData.gpuCacheEntryCount() == 1, "StaticData separates one registered frame from one cache entry");

    bool firstSucceeded = false;
    std::vector<AlgoOutput> firstOutputs;
    if (tasksReady && frameReady && frameRegistered) {
        firstSucceeded = executeTask(firstTask, atom, staticData, location);
        firstOutputs = atom.result.outputs;
    }
    test.expect(firstSucceeded, "first task instance uploads and processes the frame");

    std::fill(atom.data.begin(), atom.data.end(), 0);
    const bool secondSucceeded = tasksReady && frameReady && frameRegistered && executeTask(secondTask, atom, staticData, location);
    test.expect(secondSucceeded, "second task instance consumes existing frame GPU data");

    bool outputsMatch = firstSucceeded && firstOutputs.size() == atom.result.outputs.size();
    for (std::size_t index = 0; outputsMatch && index < firstOutputs.size(); ++index) {
        const AlgoOutput& expected = firstOutputs[index];
        const AlgoOutput& actual = atom.result.outputs[index];
        outputsMatch = expected.algoName == actual.algoName && expected.width == actual.width && expected.height == actual.height && expected.data == actual.data;
    }
    test.expect(outputsMatch, "second task reads frame-owned GPU data instead of modified host input");

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
        GpuDataAccess access = cache.acquire(metadata, resources);
        if (!access || access.source() != GpuDataAccessSource::CacheFill) {
            return false;
        }
        const bool submitted = cudaMemsetAsync(access.writableData(), static_cast<int>(metadata.id), access.bytes(), resources.stream) == cudaSuccess;
        return access.complete(submitted);
    };

    test.expect(fillFrame(firstMetadata) && fillFrame(secondMetadata), "fill both cache entries");
    GpuDataAccess firstHit = cache.acquire(firstMetadata, resources);
    test.expect(static_cast<bool>(firstHit) && firstHit.source() == GpuDataAccessSource::CacheHit && firstHit.complete(true), "touch first frame so second frame becomes LRU");
    test.expect(fillFrame(thirdMetadata), "third frame evicts one inactive cache entry");
    GpuDataAccess secondAgain = cache.acquire(secondMetadata, resources);
    test.expect(static_cast<bool>(secondAgain) && secondAgain.source() == GpuDataAccessSource::CacheFill, "least-recently-used second frame was evicted");
    if (secondAgain) {
        test.expect(secondAgain.complete(true), "complete refill after LRU eviction");
    }

    test.expect(cache.release(), "release LRU test cache");
    test.expect(releaseAccessResources(resources), "release LRU test resources");
}

void testTaskFallbackExecution(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask task(92, location.gpuId, ExecutionModel::Batched, runtime);
    const bool taskReady = loadTask(task, location) && notifyTask(task);
    test.expect(taskReady, "prepare task for zero-capacity fallback execution");

    const FrameMetadata metadata = makeFrameMetadata(30, runtime);
    FrameCpuAtom atom(metadata, runtime);
    StaticData staticData;
    const bool frameReady = initializeStaticData(staticData, location, runtime, {metadata}, 0);
    const bool frameRegistered = staticData.validateFrame(metadata);
    test.expect(frameReady && frameRegistered && staticData.gpuCacheEntryCount() == 0, "initialize registered frame with GPU cache disabled");

    const bool succeeded = taskReady && frameReady && executeTask(task, atom, staticData, location);
    test.expect(succeeded, "task executes correctly through its fallback input buffer");
    if (frameRegistered) {
        for (const AlgoOutput& output : atom.result.outputs) {
            test.expect(verifyOutput(atom, output, runtime.frameW), "fallback execution matches CPU reference");
        }
    }

    test.expect(releaseStaticData(staticData, location), "release zero-capacity StaticData");
    test.expect(unloadTask(task, location), "unload fallback execution task");
}

void testLifecycleAndResults(TestContext& test, const GpuLocation& location, ExecutionModel model, int taskId) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    DummyTask task(taskId, location.gpuId, model, runtime);
    ParameterRegistry prematureRegistry;
    ParameterSnapshot prematureSnapshot;
    const FrameMetadata prematureMetadata = makeFrameMetadata(1, runtime);
    const FrameMetadata firstMetadata = makeFrameMetadata(7, runtime);
    const FrameMetadata secondMetadata = makeFrameMetadata(19, runtime);
    const FrameMetadata malformedMetadata = makeFrameMetadata(23, runtime);
    const FrameMetadata mismatchedAtomMetadata = makeFrameMetadata(24, runtime);
    const FrameMetadata mismatchedRecordMetadata = makeFrameMetadata(25, runtime);
    FrameCpuAtom prematureAtom(prematureMetadata, runtime);
    test.expect(prematureAtom.result.id == prematureMetadata.id && prematureAtom.result.ok && prematureAtom.result.outputs.size() == 3, "FrameCpuAtom owns a preallocated result");
    test.expect(!task.registerParameters(prematureRegistry), "reject registerParameters before load");
    test.expect(!task.notifyParameters(prematureSnapshot), "reject notifyParameters before registration");
    test.expect(loadTask(task, location), "load task resources");

    StaticData staticData;
    const std::vector<FrameMetadata> frameConfigs = {
        prematureMetadata,
        firstMetadata,
        secondMetadata,
        malformedMetadata,
        mismatchedRecordMetadata,
    };
    test.expect(initializeStaticData(staticData, location, runtime, frameConfigs), "initialize task StaticData pool");
    FrameMetadata wrongLayoutMetadata = firstMetadata;
    ++wrongLayoutMetadata.width;
    test.expect(staticData.validateFrame(firstMetadata), "find sparse frame ID through StaticData index");
    test.expect(!staticData.validateFrame(wrongLayoutMetadata), "reject indexed frame ID with mismatched metadata");
    test.expect(staticData.execute() && !task.execute(prematureAtom, staticData), "reject execute before notification");
    test.expect(!task.load(), "reject repeated load");
    test.expect(notifyTask(task), "register and notify shared parameters");

    FrameCpuAtom firstAtom(firstMetadata, runtime);
    FrameCpuAtom secondAtom(secondMetadata, runtime);
    const bool firstFrameRegistered = staticData.validateFrame(firstMetadata);
    const bool secondFrameRegistered = staticData.validateFrame(secondMetadata);
    test.expect(firstFrameRegistered && secondFrameRegistered, "find registered task frames in StaticData");
    bool firstSucceeded = false;
    bool secondSucceeded = false;
    std::thread::id firstThreadId;
    std::thread::id secondThreadId;
    std::mutex sequenceLock;
    std::condition_variable sequenceCondition;
    int turn = 0;
    std::thread firstWorker([&] {
        firstThreadId = std::this_thread::get_id();
        if (GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode)) {
            firstSucceeded = staticData.execute() && task.execute(firstAtom, staticData);
        }
        {
            std::lock_guard<std::mutex> guard(sequenceLock);
            turn = 1;
        }
        sequenceCondition.notify_one();
    });
    std::thread secondWorker([&] {
        secondThreadId = std::this_thread::get_id();
        const bool pinned = GpuContextManager::pinCurrentThreadToNumaNode(location.numaNode);
        {
            std::unique_lock<std::mutex> guard(sequenceLock);
            sequenceCondition.wait(guard, [&turn] { return turn == 1; });
        }
        if (pinned) {
            secondSucceeded = staticData.execute() && task.execute(secondAtom, staticData);
        }
    });
    firstWorker.join();
    secondWorker.join();

    test.expect(firstThreadId != secondThreadId, "mobility test uses distinct live host threads");
    test.expect(firstSucceeded && secondSucceeded, "one task executes sequential frames on different threads");
    if (firstFrameRegistered) {
        for (const AlgoOutput& output : firstAtom.result.outputs) {
            test.expect(verifyOutput(firstAtom, output, runtime.frameW), "first frame matches CPU reference");
        }
    }
    if (secondFrameRegistered) {
        for (const AlgoOutput& output : secondAtom.result.outputs) {
            test.expect(verifyOutput(secondAtom, output, runtime.frameW), "second frame matches CPU reference");
        }
    }

    FrameCpuAtom malformedAtom(malformedMetadata, runtime);
    const bool malformedFrameRegistered = staticData.validateFrame(malformedMetadata);
    test.expect(malformedFrameRegistered, "find malformed-input frame in StaticData");
    malformedAtom.data.pop_back();
    test.expect(staticData.execute() && !task.execute(malformedAtom, staticData), "reject malformed frame input");
    test.expect(malformedFrameRegistered && !malformedAtom.result.ok, "malformed atom records failed task status");

    FrameCpuAtom mismatchedAtom(mismatchedAtomMetadata, runtime);
    const bool unrelatedFrameRegistered = staticData.validateFrame(mismatchedRecordMetadata);
    test.expect(unrelatedFrameRegistered, "find intentionally unrelated metadata in StaticData");
    test.expect(staticData.execute() && !task.execute(mismatchedAtom, staticData), "reject atom without matching StaticData registry metadata");
    test.expect(!mismatchedAtom.result.ok && staticData.validateFrame(mismatchedRecordMetadata), "missing frame fails its atom result without affecting unrelated registered metadata");

    test.expect(releaseStaticData(staticData, location), "release task StaticData pool");
    test.expect(unloadTask(task, location), "unload task resources");
    test.expect(task.unload(), "repeated unload is harmless");
    test.expect(task.lifecycle() == TaskLifecycle::Unloaded, "task reaches unloaded lifecycle state");
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

GraphConfig makeGraphConfig(const GpuLocation& location, std::size_t tasks, std::size_t workers, ExecutionModel model) {
    GraphConfig config;
    config.numaNode = location.numaNode;
    config.gpuIds = {location.gpuId};
    config.taskInstancesPerGpu = tasks;
    config.graphThreads = workers;
    config.warmupFramesPerGpu = 0;
    config.timedFramesPerGpu = 8;
    config.executionModel = model;
    config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    config.parameters.name = "graph-test";
    return config;
}

void testTemporaryTopologyGuard(TestContext& test, const GpuLocation& location) {
    GraphSink sink;
    std::atomic<bool> cancellation{false};
    GraphConfig config = makeGraphConfig(location, 1, 1, ExecutionModel::Batched);
    config.gpuIds = {location.gpuId, location.gpuId};
    DummyGraph graph(config, sink, cancellation);
    test.expect(!graph.initialize(), "reject more than one GPU per NUMA graph copy in temporary scope");
    test.expect(!cancellation.load(std::memory_order_acquire) && sink.count() == 0, "topology rejection occurs before graph execution");
    test.expect(graph.shutdown(), "unsupported topology has no resources to clean up");

    GraphConfig wrongNumaConfig = makeGraphConfig(location, 1, 1, ExecutionModel::Batched);
    wrongNumaConfig.numaNode = location.numaNode + 1;
    DummyGraph wrongNumaGraph(wrongNumaConfig, sink, cancellation);
    test.expect(!wrongNumaGraph.initialize(), "reject a graph GPU outside the configured NUMA node");
    test.expect(!cancellation.load(std::memory_order_acquire) && wrongNumaGraph.shutdown(), "GPU/NUMA mismatch fails before graph resources are created");
}

void testIndependentPools(TestContext& test, const GpuLocation& location) {
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        DummyGraph graph(makeGraphConfig(location, 1, 2, ExecutionModel::Batched), sink, cancellation);
        test.expect(graph.initialize(), "initialize one-task two-worker graph");
        PhaseGate invalidPhaseGate;
        test.expect(!graph.startPhase(static_cast<FramePhase>(99), invalidPhaseGate), "reject an invalid graph-owned frame phase");
        invalidPhaseGate.release();
        test.expect(runGraphPhase(graph, FramePhase::Timed), "run one-task two-worker graph");
        PhaseGate repeatedPhaseGate;
        test.expect(!graph.startPhase(FramePhase::Timed, repeatedPhaseGate), "graph-owned phase state rejects repeated submission");
        repeatedPhaseGate.release();
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "DummyGraph free-task pool prevents concurrent reuse of its sole task instance");
        test.expect(sink.count() == 8 && sink.failureCount() == 0, "one-task graph completes every frame once");
        test.expect(graph.shutdown(), "shutdown one-task graph");
    }
    {
        GraphSink sink;
        std::atomic<bool> cancellation{false};
        DummyGraph graph(makeGraphConfig(location, 2, 1, ExecutionModel::Interleaved), sink, cancellation);
        test.expect(graph.initialize(), "initialize two-task one-worker graph");
        test.expect(runGraphPhase(graph, FramePhase::Timed), "run two-task one-worker graph");
        test.expect(graph.lastMaxConcurrentExecutions() == 1, "worker pool limits concurrency independently");
        test.expect(sink.count() == 8 && sink.failureCount() == 0, "one-worker graph completes every frame once");
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
        config.numaNode = entry.first;
        config.gpuIds = entry.second;
        config.taskInstancesPerGpu = 1;
        config.graphThreads = entry.second.size();
        config.timedFramesPerGpu = 1;
        config.runtime = makeRuntime(ImageSizing::MIN_FACTOR);
        config.parameters.name = "numa-test";
        graphs.push_back(std::make_unique<DummyGraph>(config, sink, cancellation));
    }
    bool ok = true;
    for (const std::unique_ptr<DummyGraph>& graph : graphs) {
        ok = graph->initialize() && ok;
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
        ok = graph->shutdown() && ok;
    }
    test.expect(ok && sink.count() == locations.size(), "one graph copy runs on each GPU-bearing NUMA node");
}

void testGraphCancellation(TestContext& test, const GpuLocation& location) {
    GraphSink sink;
    std::atomic<bool> cancellation{false};
    GraphConfig config = makeGraphConfig(location, 1, 2, static_cast<ExecutionModel>(99));
    config.warmupFramesPerGpu = 2;
    config.timedFramesPerGpu = 4;
    DummyGraph graph(config, sink, cancellation);
    test.expect(graph.initialize(), "initialize graph used for failure propagation");
    test.expect(!runGraphPhase(graph, FramePhase::Warmup), "execution failure fails graph phase");
    test.expect(cancellation.load(std::memory_order_acquire), "execution failure raises global cancellation");
    test.expect(sink.count() == 2 && sink.failureCount() == 2, "active failed phase gives every frame one terminal result");
    test.expect(graph.shutdown(), "failed graph still unloads cleanly");
    test.expect(sink.count() == 6 && sink.failureCount() == 6, "shutdown cancels every preallocated future frame exactly once");
}

}  // namespace

int main() {
    TestContext test;
    testParameterRegistry(test);

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

    testLifecycleAndResults(test, locations.front(), ExecutionModel::Batched, 1);
    testLifecycleAndResults(test, locations.front(), ExecutionModel::Interleaved, 2);
    testStaticDataValidation(test, locations.front());
    testGpuDataAccessState(test, locations.front());
    testGpuCacheManagerLru(test, locations.front());
    testFrameDataAcrossTaskInstances(test, locations.front());
    testTaskFallbackExecution(test, locations.front());
    testTemporaryTopologyGuard(test, locations.front());
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
