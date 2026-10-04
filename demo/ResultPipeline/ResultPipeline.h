#pragma once

#include "ResultPipeline/ResultTasks.h"

namespace result_pipeline {

struct PipelineConfig {
    std::size_t frames = 1000;
    std::size_t entries = 16;
    std::size_t inFlight = 32;
    std::size_t bytes = 4096;
    std::size_t cpuStages = 3;
    std::chrono::microseconds cpuDelay{50};
    std::chrono::milliseconds cacheWait{50};
    CacheEvictionPolicy policy = CacheEvictionPolicy::LRU;
    CacheRetention retention = CacheRetention::Keep;
};

struct PipelineMeasurement {
    CacheStatistics cache;
    std::uint64_t completed = 0;
    std::uint64_t consumerHits = 0;
    std::uint64_t consumerUploadedBytes = 0;
    std::size_t peakInFlight = 0;
    double framesPerSecond = 0;
    double latencyP50Us = 0;
    double latencyP95Us = 0;
};

// Call under framework NUMA affinity. Three NUMA-local workers overlap A,
// the CPU task chain, and B. One instance per stage, bounded reusable atoms.
// Stops all workers on failure; joins them before task/cache destruction.
bool runPipeline(int numaNode, const PipelineConfig& config, PipelineMeasurement& measurement);

}  // namespace result_pipeline
