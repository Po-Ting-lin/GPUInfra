#include "test_support.h"

namespace gpuinfra_tests {

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

void testStaticGpuData(TestContext& test, const GpuLocation& location) {
    StaticGpuData storage;
    test.expect(!storage.finalize() && storage.data(StaticGpuDataType::DetectionZones) == nullptr, "empty static storage is not readable");
    test.expect(storage.initialize(location.gpuId, 128, 32), "allocate independent static buffers");
    test.expect(storage.state() == StaticGpuDataState::Uploading && storage.data(StaticGpuDataType::DetectionZones) == nullptr, "hide pointers before finalize");
    test.expect(storage.bytes(StaticGpuDataType::DetectionZones) == 128 && storage.bytes(StaticGpuDataType::DistortionTables) == 32, "report independent byte capacities");
    test.expect(!storage.initialize(location.gpuId, 128, 32), "reject reinitialization without release");

    unsigned char* h_staging = nullptr;
    const bool stagingReady = cudaMallocHost(reinterpret_cast<void**>(&h_staging), 32) == cudaSuccess;
    test.expect(stagingReady, "allocate reusable pinned upload source");
    if (!stagingReady) {
        storage.release();
        return;
    }
    std::memset(h_staging, 0x31, 32);
    test.expect(storage.upload(StaticGpuDataType::DetectionZones, h_staging, 32, 0), "upload first detection zone segment");
    std::memset(h_staging, 0x52, 32);
    test.expect(storage.upload(StaticGpuDataType::DetectionZones, h_staging, 32, 96), "reuse staging source and upload exactly to capacity boundary");
    std::memset(h_staging, 0x73, 32);
    test.expect(storage.upload(StaticGpuDataType::DistortionTables, h_staging, 32, 0), "upload independent distortion buffer");
    std::memset(h_staging, 0, 32);
    test.expect(cudaFreeHost(h_staging) == cudaSuccess, "source can be freed before finalize");
    test.expect(storage.finalize() && storage.state() == StaticGpuDataState::Ready, "finalize enables read-only data");

    std::array<unsigned char, 128> h_zones{};
    std::array<unsigned char, 32> h_distortion{};
    test.expect(cudaMemcpy(h_zones.data(), storage.data(StaticGpuDataType::DetectionZones), h_zones.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "read detection zone contents");
    test.expect(cudaMemcpy(h_distortion.data(), storage.data(StaticGpuDataType::DistortionTables), h_distortion.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "read distortion contents");
    bool contentsMatch = true;
    for (std::size_t index = 0; index < h_zones.size(); ++index) {
        const unsigned char expected = index < 32 ? 0x31 : index >= 96 ? 0x52 : 0;
        contentsMatch = contentsMatch && h_zones[index] == expected;
    }
    test.expect(contentsMatch && std::all_of(h_distortion.begin(), h_distortion.end(), [](unsigned char value) { return value == 0x73; }), "segments survive staging reuse and unwritten gaps stay zero");
    test.expect(!storage.upload(StaticGpuDataType::DetectionZones, h_zones.data(), 1, 0) && storage.state() == StaticGpuDataState::Ready, "post-finalize writes are rejected without changing published data");
    test.expect(!storage.finalize(), "reject duplicate finalize");
    test.expect(storage.release() && storage.release(), "static release is idempotent");
    test.expect(storage.state() == StaticGpuDataState::Empty && storage.bytes(StaticGpuDataType::DetectionZones) == 0, "release clears capacities and state");

    test.expect(storage.initialize(location.gpuId, 0, 32) && storage.finalize(), "support distortion-only configuration");
    test.expect(storage.data(StaticGpuDataType::DetectionZones) == nullptr && storage.data(StaticGpuDataType::DistortionTables) != nullptr, "disabled region has no pointer");
    h_distortion.fill(0xff);
    test.expect(cudaMemcpy(h_distortion.data(), storage.data(StaticGpuDataType::DistortionTables), h_distortion.size(), cudaMemcpyDeviceToHost) == cudaSuccess && std::all_of(h_distortion.begin(), h_distortion.end(), [](unsigned char value) { return value == 0; }), "reinitialization zeroes new allocation without an upload");
    test.expect(storage.release(), "release reinitialized storage");
    test.expect(storage.initialize(location.gpuId, 0, 0) && storage.finalize() && storage.release(), "support fully disabled static storage");

    for (int scenario = 0; scenario < 6; ++scenario) {
        test.expect(storage.initialize(location.gpuId, 16, 0), "initialize upload validation scenario");
        bool uploaded = false;
        switch (scenario) {
            case 0:
                uploaded = storage.upload(StaticGpuDataType::DetectionZones, h_zones.data(), 2, 15);
                break;
            case 1:
                uploaded = storage.upload(StaticGpuDataType::DetectionZones, h_zones.data(), 1, std::numeric_limits<std::size_t>::max());
                break;
            case 2:
                uploaded = storage.upload(StaticGpuDataType::DetectionZones, nullptr, 1, 0);
                break;
            case 3:
                uploaded = storage.upload(StaticGpuDataType::DistortionTables, h_zones.data(), 1, 0);
                break;
            case 4:
                uploaded = storage.upload(static_cast<StaticGpuDataType>(99), h_zones.data(), 1, 0);
                break;
            case 5:
                uploaded = storage.upload(StaticGpuDataType::DetectionZones, h_zones.data(), 0, 0);
                break;
        }
        test.expect(!uploaded && storage.state() == StaticGpuDataState::Failed && !storage.finalize(), "invalid upload poisons initialization and prevents publication");
        test.expect(!storage.upload(StaticGpuDataType::DetectionZones, h_zones.data(), 1, 0) && storage.data(StaticGpuDataType::DetectionZones) == nullptr, "failed upload cannot be retried without reinitialization");
        test.expect(storage.release(), "release failed upload resources");
    }
    test.expect(!storage.initialize(-1, 16, 0) && storage.state() == StaticGpuDataState::Failed && storage.release(), "invalid GPU fails initialization");
    test.expect(!storage.initialize(location.gpuId, 16, std::numeric_limits<std::size_t>::max()), "allocation failure aborts after first buffer allocation");
    cudaGetLastError();
    test.expect(storage.state() == StaticGpuDataState::Failed && storage.data(StaticGpuDataType::DetectionZones) == nullptr && storage.release(), "partial initialization can be safely cleaned");
    test.expect(storage.initialize(location.gpuId, 16, 16) && storage.finalize() && storage.release(), "initialize successfully after allocation failure cleanup");
}

void testStaticDataRegions(TestContext& test, const GpuLocation& location) {
    const AlgoRuntimeInfo runtime = makeRuntime(ImageSizing::MIN_FACTOR);
    const FrameMetadata metadata = makeFrameMetadata(812, runtime);
    StaticDataConfig config;
    config.runtime = runtime;
    config.detectionZoneBytes = 64;
    config.distortionBytes = 16;
    config.gpuCacheEntries = 0;
    StaticData staticData;
    test.expect(staticData.init(config), "initialize static regions with frame cache disabled");
    test.expect(staticData.isInitialized() && !staticData.execute() && !staticData.validateFrame(metadata), "allocated static data is not yet executable");
    TaskGpuResources resources;
    const bool resourcesReady = initializeAccessResources(resources, location, runtime.inBytes);
    test.expect(resourcesReady, "allocate caller for static integration test");
    if (!resourcesReady) {
        staticData.release();
        return;
    }
    {
        GpuDataAccess premature = staticData.getCacheData(metadata, makeCacheRequest(resources));
        test.expect(premature.status() == CacheStatus::Invalid, "direct cache request cannot bypass static readiness");
    }
    std::array<unsigned char, 16> h_payload;
    h_payload.fill(0x65);
    test.expect(staticData.uploadStaticData(StaticGpuDataType::DetectionZones, h_payload.data(), h_payload.size(), 16), "graph owner uploads detection zones");
    test.expect(staticData.uploadStaticData(StaticGpuDataType::DistortionTables, h_payload.data(), h_payload.size(), 0) && staticData.finalizeStaticData(), "graph owner finishes both static regions");
    test.expect(staticData.execute() && staticData.validateFrame(metadata), "finalized static data enables execution");
    const void* d_zones = staticData.staticGpuData(StaticGpuDataType::DetectionZones);
    test.expect(staticData.staticGpuDataBytes(StaticGpuDataType::DetectionZones) == 64, "wrapper exposes byte capacity");
    {
        GpuDataAccess fallback = staticData.getCacheData(metadata, makeCacheRequest(resources));
        test.expect(fallback.status() == CacheStatus::TaskFallback, "zero-capacity frame cache works with static data");
        test.expect(!staticData.release() && staticData.execute() && staticData.staticGpuData(StaticGpuDataType::DetectionZones) == d_zones, "rejected release preserves static buffers and readiness");
        test.expect(fallback.freeCacheData(true), "finish frame fallback before reset");
    }
    CacheStatistics completed;
    test.expect(staticData.cacheStatisticsSnapshot().fallback == 1, "StaticData exposes manager snapshot");
    test.expect(staticData.resetCache(&completed) && completed.fallback == 1 && staticData.cacheStatisticsSnapshot().fallback == 0 && staticData.staticGpuData(StaticGpuDataType::DetectionZones) == d_zones, "StaticData reset returns statistics and retains static allocation");
    std::array<unsigned char, 64> h_zones{};
    test.expect(cudaMemcpy(h_zones.data(), d_zones, h_zones.size(), cudaMemcpyDeviceToHost) == cudaSuccess && h_zones[16] == 0x65 && h_zones[0] == 0, "frame reset preserves static bytes");
    test.expect(staticData.release() && !staticData.execute(), "release disables execution");
    test.expect(staticData.init(config), "reinitialize static owner at a quiescent boundary");
    test.expect(!staticData.uploadStaticData(StaticGpuDataType::DetectionZones, h_payload.data(), 16, 63), "reject invalid upload through StaticData");
    test.expect(!staticData.finalizeStaticData() && !staticData.execute() && staticData.release(), "failed static preparation blocks execution until release");
    config.detectionZoneBytes = std::numeric_limits<std::size_t>::max();
    test.expect(!staticData.init(config) && !staticData.execute() && staticData.release(), "reject total GPU capacity overflow without degraded initialization");
    test.expect(releaseAccessResources(resources), "release static integration caller");
}

}  // namespace gpuinfra_tests
