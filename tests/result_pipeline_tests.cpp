#include "test_support.h"
#include "ResultPipeline/ResultPipeline.h"

namespace gpuinfra_tests {
namespace {

void testOrderedPipeline(TestContext& test, const GpuLocation& location, CacheEvictionPolicy policy, CacheRetention retention) {
    using namespace result_pipeline;
    constexpr std::size_t BYTES = 1024;
    GpuCacheManager cache;
    TaskA producer;
    TaskB consumer;
    DummyCPUTask cpu;
    const bool initialized = cache.initialize({location.gpuId}, BYTES, 2, std::chrono::milliseconds(0), policy) && producer.load(BYTES) && consumer.load(BYTES);
    test.expect(initialized, "initialize ordered A/CPU/B pipeline");
    if (!initialized) return;
    ResultAtom first = makeAtom(1, BYTES);
    ResultAtom second = makeAtom(2, BYTES);
    ResultAtom third = makeAtom(3, BYTES);
    // A1, A2, CPU1, B1, A3, CPU2, B2: final-use B1 makes LRU
    // retain dead data, whereas FIFO or explicit Discard preserves result 2.
    test.expect(producer.execute(cache, first) && producer.execute(cache, second), "interleave two producer frames before either consumer");
    test.expect(!consumer.execute(cache, first, 2, retention), "B rejects a frame before its CPU stages finish");
    test.expect(cpu.execute(first) && cpu.execute(first) && consumer.execute(cache, first, 2, retention), "B1 reads A1 after two CPU stages");
    test.expect(consumer.hits == 1 && consumer.uploadedBytes == 0, "B1 hit avoids result H2D");
    test.expect(producer.execute(cache, third), "A3 reuses producer resources while A2 awaits CPU work");
    test.expect(cpu.execute(second) && cpu.execute(second) && consumer.execute(cache, second, 2, retention), "B2 verifies original A2 bytes after interleaving");
    const bool secondHits = retention == CacheRetention::Discard || policy == CacheEvictionPolicy::FIFO;
    test.expect(consumer.hits == (secondHits ? 2U : 1U) && consumer.uploadedBytes == (secondHits ? 0U : BYTES), "FIFO/Discard preserve A2; LRU Keep recovers evicted A2 from CPU atom");
    const CacheStatistics statistics = cache.statisticsSnapshot();
    test.expect(statistics.discard == (retention == CacheRetention::Discard ? 2U : 0U), "only terminal Discard removes entries");
    test.expect(statistics.eviction == (retention == CacheRetention::Discard ? 0U : policy == CacheEvictionPolicy::LRU ? 2U : 1U), "deterministic policy victim order");
    test.expect(cpu.execute(third) && cpu.execute(third) && consumer.execute(cache, third, 2, retention), "B3 verifies final frame");
    test.expect(cache.resetCache(), "all task leases end before graph reset");

    ResultTaskResources heldResources;
    test.expect(heldResources.load(BYTES), "load independent held readers for forced fallback");
    ResultAtom heldFirst = makeAtom(11, BYTES);
    ResultAtom heldSecond = makeAtom(12, BYTES);
    ResultAtom fallback = makeAtom(13, BYTES);
    test.expect(producer.execute(cache, heldFirst) && producer.execute(cache, heldSecond), "prepare full cache");
    {
        GpuDataAccess firstLease = cache.getCacheData(heldFirst.metadata, heldResources.request());
        GpuDataAccess secondLease = cache.getCacheData(heldSecond.metadata, heldResources.request());
        test.expect(firstLease.status() == CacheStatus::CacheHit && secondLease.status() == CacheStatus::CacheHit, "hold both independent entries");
        test.expect(producer.execute(cache, fallback) && cpu.execute(fallback) && consumer.execute(cache, fallback, 1, retention), "A fallback publishes CPU atom and B fallback restores correct GPU bytes");
        test.expect(cache.statisticsSnapshot().fallbackReasons.fullNoWait == 2, "both tasks use explicit full-cache fallback");
        test.expect(firstLease.freeCacheData(true) && secondLease.freeCacheData(true), "release forced readers");
    }
    test.expect(cache.resetCache(), "reset after fallback transaction cleanup");
    fallback.result[0] ^= 1;
    test.expect(!consumer.execute(cache, fallback, 1, retention), "B detects corrupt atom even on refill");
    test.expect(cache.resetCache(), "failed result verification leaves no live lease");
    test.expect(heldResources.unload() && producer.unload() && consumer.unload() && cache.release(), "release ordered simulation resources");
}

}  // namespace

void testResultPipeline(TestContext& test, const GpuLocation& location) {
    using namespace result_pipeline;
    for (CacheEvictionPolicy policy : {CacheEvictionPolicy::LRU, CacheEvictionPolicy::FIFO}) {
        for (CacheRetention retention : {CacheRetention::Keep, CacheRetention::Discard}) {
            testOrderedPipeline(test, location, policy, retention);
            PipelineConfig config;
            config.frames = 48;
            config.entries = 2;
            config.inFlight = 1;
            config.bytes = 1024;
            config.cpuStages = 3;
            config.cpuDelay = std::chrono::microseconds(0);
            config.policy = policy;
            config.retention = retention;
            PipelineMeasurement measurement;
            test.expect(runPipeline(location.numaNode, config, measurement), "three worker pipeline completes serial baseline");
            test.expect(measurement.completed == config.frames && measurement.consumerHits == config.frames && measurement.consumerUploadedBytes == 0 && measurement.peakInFlight == 1, "single in-flight frame always reaches B through cache hit");
            test.expect(measurement.cache.discard == (retention == CacheRetention::Discard ? config.frames : 0), "pipeline applies selected terminal retention");
            config.inFlight = 8;
            config.cpuDelay = std::chrono::microseconds(100);
            test.expect(runPipeline(location.numaNode, config, measurement), "overlapping A/CPU/B pipeline verifies every output");
            test.expect(measurement.completed == config.frames && measurement.peakInFlight <= config.inFlight, "all interleaved frames complete within bounded atom window");
            test.expect(measurement.consumerUploadedBytes == (config.frames - measurement.consumerHits) * config.bytes, "each consumer miss recovers exactly one CPU payload");
            test.expect(measurement.cache.invalid == 0 && measurement.cache.fillFailed == 0 && measurement.cache.hit + measurement.cache.fill + measurement.cache.fallback == 2 * config.frames, "every frame has exactly one successful A and B transaction");
            // No timing or scheduling-dependent hit-rate assertions.
        }
    }
}

}  // namespace gpuinfra_tests
