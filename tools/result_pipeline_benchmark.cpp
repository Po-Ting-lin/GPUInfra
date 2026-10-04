#include <charconv>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "Grape/NumaExecutor.h"
#include "ResultPipeline/ResultPipeline.h"

namespace {

bool parseSize(const char* text, std::size_t& value) {
    const char* end = text + std::strlen(text);
    const std::from_chars_result result = std::from_chars(text, end, value);
    return result.ec == std::errc() && result.ptr == end;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace result_pipeline;
    PipelineConfig config;
    std::size_t delay = 50;
    std::size_t repeats = 3;
    if (argc != 8 || !parseSize(argv[1], config.entries) || !parseSize(argv[2], config.inFlight) || !parseSize(argv[3], config.frames) || !parseSize(argv[4], config.bytes) || !parseSize(argv[5], config.cpuStages) || !parseSize(argv[6], delay) || !parseSize(argv[7], repeats) || delay > 100000 || repeats == 0 || repeats > 20) {
        std::cerr << "Usage: " << argv[0] << " entries in_flight frames bytes cpu_stages cpu_delay_us repeats\n";
        return 2;
    }
    config.cpuDelay = std::chrono::microseconds(delay);
    if (!GpuContextManager::init(GpuInfraConfig{})) return 1;
    const std::vector<GpuLocation> locations = GpuContextManager::gpuLocations();
    bool ok = false;
    if (!locations.empty()) {
        ok = NumaExecutor(locations.front().numaNode).run([&] {
            std::cout << "repeat,policy,retention,entries,in_flight,frames,bytes,cpu_stages,cpu_delay_us,cache_wait_ms,peak_in_flight,frames_per_second,latency_p50_us,latency_p95_us,b_hits,b_h2d_bytes,hit,fill,eviction,discard,fallback,loading_timeout,full_timeout,loading_no_wait,full_no_wait,capacity_zero,replica_unavailable,waited_requests,total_wait_us,max_wait_us\n";
            // Untimed CUDA/worker warmup. Each measured run uses a fresh cache.
            PipelineConfig warmup = config;
            warmup.frames = 16;
            warmup.cpuDelay = std::chrono::microseconds(0);
            PipelineMeasurement measurement;
            if (!runPipeline(locations.front().numaNode, warmup, measurement)) return false;
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                for (std::size_t order = 0; order < 4; ++order) {
                    const std::size_t combination = (repeat + order) % 4;
                    config.policy = combination < 2 ? CacheEvictionPolicy::LRU : CacheEvictionPolicy::FIFO;
                    config.retention = combination % 2 == 0 ? CacheRetention::Keep : CacheRetention::Discard;
                    if (!runPipeline(locations.front().numaNode, config, measurement)) return false;
                    const CacheStatistics& s = measurement.cache;
                    std::cout << repeat + 1 << ',' << (config.policy == CacheEvictionPolicy::LRU ? "lru" : "fifo") << ',' << (config.retention == CacheRetention::Keep ? "keep" : "discard") << ',' << config.entries << ',' << config.inFlight << ',' << config.frames << ',' << config.bytes << ',' << config.cpuStages << ',' << delay << ',' << config.cacheWait.count() << ',' << measurement.peakInFlight << ',' << std::fixed << std::setprecision(3) << measurement.framesPerSecond << ',' << measurement.latencyP50Us << ',' << measurement.latencyP95Us << ',' << measurement.consumerHits << ',' << measurement.consumerUploadedBytes << ',' << s.hit << ',' << s.fill << ',' << s.eviction << ',' << s.discard << ',' << s.fallback << ',' << s.fallbackReasons.loadingTimeout << ',' << s.fallbackReasons.fullTimeout << ',' << s.fallbackReasons.loadingNoWait << ',' << s.fallbackReasons.fullNoWait << ',' << s.fallbackReasons.capacityZero << ',' << s.fallbackReasons.replicaUnavailable << ',' << s.waitedRequests << ',' << std::chrono::duration<double, std::micro>(s.totalWaitTime).count() << ',' << std::chrono::duration<double, std::micro>(s.maxWaitTime).count() << '\n';
                }
            }
            return true;
        });
    }
    GpuContextManager::shutdown();
    return ok ? 0 : 1;
}
