#include <algorithm>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "Context/GpuContextManager.h"
#include "CudaCheck.h"
#include "DataCache/GpuDataCache.h"
#include "Grape/NumaExecutor.h"

namespace {

constexpr std::size_t PAYLOAD_BYTES = 64;
constexpr std::size_t WARMUP_REQUESTS = 100;

struct StartGate {
    std::mutex lock;
    std::condition_variable changed;
    std::size_t ready = 0;
    bool start = false;
};

class Caller {
public:
    Caller() = default;
    ~Caller() {
        if (request.stream != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(request.stream), );
        }
        if (request.d_fallback != nullptr) {
            CUDA_CHECK(cudaFree(request.d_fallback), );
        }
        if (request.stream != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(request.stream), );
        }
    }

    bool initialize(int gpuId, std::size_t iterations) {
        request.gpuId = gpuId;
        request.fallbackBytes = PAYLOAD_BYTES;
        lookupUs.resize(iterations);
        CUDA_CHECK(cudaStreamCreateWithFlags(&request.stream, cudaStreamNonBlocking), return false);
        CUDA_CHECK(cudaMalloc(&request.d_fallback, PAYLOAD_BYTES), return false);
        return true;
    }

    Caller(const Caller&) = delete;
    Caller& operator=(const Caller&) = delete;

    GpuCacheRequest request;
    std::vector<double> lookupUs;
    bool succeeded = false;
};

FrameMetadata makeMetadata(std::uint64_t id) {
    FrameMetadata metadata;
    metadata.key.frameId = id;
    metadata.width = 8;
    metadata.height = 8;
    metadata.dtype = 1;
    metadata.bytes = PAYLOAD_BYTES;
    return metadata;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool fill(GpuDataCache& cache, const GpuCacheRequest& request, std::uint64_t id, double* lookupUs = nullptr) {
    const FrameMetadata metadata = makeMetadata(id);
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    GpuDataAccess access = cache.getCacheData(metadata, request);
    const std::chrono::steady_clock::time_point acquired = std::chrono::steady_clock::now();
    if (lookupUs != nullptr) {
        *lookupUs = std::chrono::duration<double, std::micro>(acquired - started).count();
    }
    if (access.status() != CacheStatus::CacheFill) {
        return false;
    }
    bool submitted = true;
    CUDA_CHECK(cudaMemsetAsync(access.writableData(), static_cast<int>(id & 0xffU), PAYLOAD_BYTES, access.getStream()), submitted = false);
    return access.freeCacheData(submitted);
}

void releaseGate(StartGate& gate) {
    {
        std::lock_guard<std::mutex> guard(gate.lock);
        gate.start = true;
    }
    gate.changed.notify_all();
}

double percentile(const std::vector<double>& sorted, std::size_t percent) {
    return sorted[(sorted.size() - 1) * percent / 100];
}

void measure(const GpuLocation& location, CacheEvictionPolicy policy, std::size_t capacity, std::size_t heldCount, std::size_t workerCount, std::size_t iterations, std::size_t repeat) {
    GpuDataCache cache;
    require(cache.initialize({location.gpuId}, PAYLOAD_BYTES, capacity, std::chrono::milliseconds(0), policy), "cache initialization failed");
    std::vector<std::unique_ptr<Caller>> callers;
    for (std::size_t index = 0; index <= workerCount; ++index) {
        callers.push_back(std::make_unique<Caller>());
        require(callers.back()->initialize(location.gpuId, index == 0 ? 0 : iterations), "caller initialization failed");
    }
    for (std::size_t index = 0; index < capacity; ++index) {
        require(fill(cache, callers[0]->request, index + 1), "prefill failed");
    }

    // Hold the oldest prefix for the whole measurement. Only reader lifetime
    // is stressed; these streams have no pending kernels or copy work.
    std::vector<GpuDataAccess> held;
    held.reserve(heldCount);
    for (std::size_t index = 0; index < heldCount; ++index) {
        held.push_back(cache.getCacheData(makeMetadata(index + 1), callers[0]->request));
        require(held.back().status() == CacheStatus::CacheHit, "could not retain oldest reader");
    }

    StartGate gate;
    std::vector<std::thread> threads;
    threads.reserve(workerCount);
    CacheStatistics before;
    std::chrono::steady_clock::time_point started;
    try {
        for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex) {
            const std::function<void(bool)> work = [&, workerIndex](bool numaReady) {
                Caller& caller = *callers[workerIndex + 1];
                bool ready = numaReady;
                CUDA_CHECK(cudaSetDevice(location.gpuId), ready = false);
                const std::uint64_t firstId = capacity + 1 + workerIndex * (WARMUP_REQUESTS + iterations);
                for (std::size_t index = 0; ready && index < WARMUP_REQUESTS; ++index) {
                    ready = fill(cache, caller.request, firstId + index);
                }
                {
                    std::unique_lock<std::mutex> guard(gate.lock);
                    ++gate.ready;
                    gate.changed.notify_all();
                    gate.changed.wait(guard, [&gate] { return gate.start; });
                }
                caller.succeeded = ready;
                for (std::size_t index = 0; caller.succeeded && index < iterations; ++index) {
                    caller.succeeded = fill(cache, caller.request, firstId + WARMUP_REQUESTS + index, &caller.lookupUs[index]);
                }
            };
            threads.push_back(NumaExecutor(location.numaNode).start(work));
        }
        {
            std::unique_lock<std::mutex> guard(gate.lock);
            gate.changed.wait(guard, [&gate, workerCount] { return gate.ready == workerCount; });
            before = cache.statisticsSnapshot();
            started = std::chrono::steady_clock::now();
            gate.start = true;
        }
        gate.changed.notify_all();
        for (std::thread& thread : threads) {
            thread.join();
        }
    } catch (...) {
        releaseGate(gate);
        for (std::thread& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        throw;
    }
    const std::chrono::steady_clock::time_point finished = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(finished - started).count();
    const CacheStatistics after = cache.statisticsSnapshot();
    const std::size_t requests = workerCount * iterations;
    require(after.fill - before.fill == requests && after.fillSucceeded - before.fillSucceeded == requests && after.eviction - before.eviction == requests && after.fallback == 0 && after.invalid == 0, "measurement did not consist exclusively of successful replacement fills");

    std::vector<double> lookupUs;
    lookupUs.reserve(requests);
    for (std::size_t index = 1; index <= workerCount; ++index) {
        require(callers[index]->succeeded, "worker failed");
        lookupUs.insert(lookupUs.end(), callers[index]->lookupUs.begin(), callers[index]->lookupUs.end());
    }
    std::sort(lookupUs.begin(), lookupUs.end());
    const double mean = std::accumulate(lookupUs.begin(), lookupUs.end(), 0.0) / static_cast<double>(requests);
    for (GpuDataAccess& access : held) {
        require(access.freeCacheData(true), "held reader cleanup failed");
    }
    require(cache.release(), "cache release failed");
    std::cout << repeat << ',' << (policy == CacheEvictionPolicy::FIFO ? "fifo" : "lru") << ',' << capacity << ',' << heldCount << ',' << workerCount << ',' << iterations << ',' << PAYLOAD_BYTES << ',' << std::fixed << std::setprecision(3) << mean << ',' << percentile(lookupUs, 50) << ',' << percentile(lookupUs, 95) << ',' << percentile(lookupUs, 99) << ',' << static_cast<double>(requests) / seconds << ',' << seconds * 1e6 / static_cast<double>(requests) << ',' << after.eviction - before.eviction << ',' << after.fallback - before.fallback << '\n';
}

bool parseSize(const char* text, std::size_t& value) {
    const char* end = text + std::strlen(text);
    const std::from_chars_result result = std::from_chars(text, end, value);
    return result.ec == std::errc() && result.ptr == end;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t capacity = 0;
    std::size_t heldCount = 0;
    std::size_t workerCount = 0;
    std::size_t iterations = 0;
    std::size_t repeats = 3;
    if ((argc != 5 && argc != 6) || !parseSize(argv[1], capacity) || !parseSize(argv[2], heldCount) || !parseSize(argv[3], workerCount) || !parseSize(argv[4], iterations) || (argc == 6 && !parseSize(argv[5], repeats)) || capacity == 0 || capacity > 65536 || heldCount >= capacity || workerCount == 0 || workerCount > 32 || workerCount > capacity - heldCount || iterations == 0 || iterations > 1000000 || repeats == 0 || repeats > 20) {
        std::fprintf(stderr, "Usage: %s entries held_readers workers iterations_per_worker [repeats=3]\nRequire: 1..65536 entries, 1..32 workers, entries - held_readers >= workers, 1..1000000 iterations, 1..20 repeats.\n", argv[0]);
        return 2;
    }
    if (!GpuContextManager::init(GpuInfraConfig())) {
        return 1;
    }
    const std::vector<GpuLocation> locations = GpuContextManager::gpuLocations();
    bool succeeded = false;
    try {
        require(!locations.empty(), "no GPU available");
        const GpuLocation location = locations.front();
        const std::function<bool()> run = [&] {
            CUDA_CHECK(cudaSetDevice(location.gpuId), return false);
            cudaDeviceProp properties{};
            int runtimeVersion = 0;
            int driverVersion = 0;
            CUDA_CHECK(cudaGetDeviceProperties(&properties, location.gpuId), return false);
            CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion), return false);
            CUDA_CHECK(cudaDriverGetVersion(&driverVersion), return false);
            std::cerr << "[CachePolicyBenchmark] gpu=" << properties.name << " device=" << location.gpuId << " numa=" << location.numaNode << " runtime=" << runtimeVersion << " driver=" << driverVersion;
#if defined(GPUINFRA_ENABLE_NVTX) && GPUINFRA_ENABLE_NVTX
            std::cerr << " nvtx=on\n";
#else
            std::cerr << " nvtx=off\n";
#endif
            std::cout << "repeat,policy,entries,held_readers,workers,iterations_per_worker,payload_bytes,lookup_mean_us,lookup_p50_us,lookup_p95_us,lookup_p99_us,requests_per_second,wall_us_per_request,evictions,fallback\n";
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                for (std::size_t order = 0; order < 2; ++order) {
                    const CacheEvictionPolicy policy = (repeat + order) % 2 == 0 ? CacheEvictionPolicy::LRU : CacheEvictionPolicy::FIFO;
                    measure(location, policy, capacity, heldCount, workerCount, iterations, repeat + 1);
                }
            }
            return true;
        };
        succeeded = NumaExecutor(location.numaNode).run(run);
    } catch (const std::exception& error) {
        std::cerr << "Cache policy benchmark failed: " << error.what() << '\n';
    }
    GpuContextManager::shutdown();
    return succeeded ? 0 : 1;
}
