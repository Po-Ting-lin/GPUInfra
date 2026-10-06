#include "test_support.h"

namespace gpuinfra_tests {
namespace {

constexpr std::size_t RESULT_BYTES = 64;

// Test-only handoff atom. Until the generic dataId API is implemented, map
// dataId into the existing frameId field of an independent result manager.
struct ResultAtom {
    std::uint64_t dataId = 0;
    FrameMetadata metadata;
    std::array<unsigned char, RESULT_BYTES> data{};
    bool ready = false;
};

ResultAtom makeResultAtom(std::uint64_t dataId) {
    ResultAtom atom;
    atom.dataId = dataId;
    atom.metadata.key.frameId = dataId;
    atom.metadata.bytes = RESULT_BYTES;
    atom.metadata.width = static_cast<int>(RESULT_BYTES);
    atom.metadata.height = 1;
    atom.metadata.dtype = 1;
    return atom;
}

bool produceResult(TestContext& test, GpuDataCache& cache, const TaskGpuResources& producer, void* d_algoOutput, ResultAtom& atom, unsigned char value, CacheStatus expectedStatus) {
    atom.ready = false;
    GpuDataAccess access = cache.getCacheData(atom.metadata, makeCacheRequest(producer));
    test.expect(access.status() == expectedStatus, "producer uses expected result reservation");
    if (access.status() != expectedStatus || access.writableData() == nullptr) {
        return false;
    }

    // Simulate an algorithm's private GPU output without changing algo APIs.
    bool submitted = cudaMemsetAsync(d_algoOutput, value, RESULT_BYTES, producer.stream) == cudaSuccess;
    if (submitted) {
        submitted = cudaMemcpyAsync(access.writableData(), d_algoOutput, RESULT_BYTES, cudaMemcpyDeviceToDevice, producer.stream) == cudaSuccess;
    }
    if (submitted) {
        submitted = cudaMemcpyAsync(atom.data.data(), d_algoOutput, RESULT_BYTES, cudaMemcpyDeviceToHost, producer.stream) == cudaSuccess;
    }
    const bool finished = access.freeCacheData(submitted);
    atom.ready = submitted && finished;
    test.expect(atom.ready && std::all_of(atom.data.begin(), atom.data.end(), [value](unsigned char byte) { return byte == value; }), "producer D2D publication and CPU atom D2H both complete before handoff");
    return atom.ready;
}

bool consumeResult(TestContext& test, GpuDataCache& cache, const TaskGpuResources& consumer, void* d_consumerOutput, const ResultAtom& atom, CacheStatus expectedStatus, std::size_t& uploadedBytes, CacheRetention retention = CacheRetention::Keep) {
    if (!atom.ready || atom.metadata.key.frameId != atom.dataId) {
        test.expect(false, "consumer requires a completed atom with the handed-off identity");
        return false;
    }
    GpuDataAccess access = cache.getCacheData(atom.metadata, makeCacheRequest(consumer));
    test.expect(access.status() == expectedStatus, "consumer retrieves the handed-off result through its own stream");
    if (access.status() != expectedStatus || !access) {
        return false;
    }

    bool submitted = true;
    if (access.status() == CacheStatus::CacheFill || access.status() == CacheStatus::TaskFallback) {
        submitted = cudaMemcpyAsync(access.writableData(), atom.data.data(), RESULT_BYTES, cudaMemcpyHostToDevice, consumer.stream) == cudaSuccess;
        if (submitted) {
            uploadedBytes += RESULT_BYTES;
        }
    }
    else {
        test.expect(access.writableData() == nullptr, "result cache hit is read-only and needs no H2D");
    }
    // A GPU consumer reads the selected result; verify all bytes after completion.
    std::array<unsigned char, RESULT_BYTES> consumed{};
    if (submitted) {
        submitted = cudaMemcpyAsync(d_consumerOutput, access.data(), RESULT_BYTES, cudaMemcpyDeviceToDevice, consumer.stream) == cudaSuccess;
    }
    if (submitted) {
        submitted = cudaMemcpyAsync(consumed.data(), d_consumerOutput, RESULT_BYTES, cudaMemcpyDeviceToHost, consumer.stream) == cudaSuccess;
    }
    const bool finished = access.freeCacheData(submitted, retention);
    test.expect(submitted && finished && consumed == atom.data, "consumer GPU result matches the authoritative CPU atom on hit, refill and fallback");
    return submitted && finished && consumed == atom.data;
}

}  // namespace

void testGpuResultCacheHandoff(TestContext& test, const GpuLocation& location, CacheEvictionPolicy policy) {
    TaskGpuResources producer;
    TaskGpuResources consumer;
    GpuDataCache resultCache;
    void* d_algoOutput = nullptr;
    void* d_consumerOutput = nullptr;
    const bool initialized = initializeAccessResources(producer, location, RESULT_BYTES) && initializeAccessResources(consumer, location, RESULT_BYTES) && cudaMalloc(&d_algoOutput, RESULT_BYTES) == cudaSuccess && cudaMalloc(&d_consumerOutput, RESULT_BYTES) == cudaSuccess && resultCache.initialize({location.gpuId}, RESULT_BYTES, 1, std::chrono::milliseconds(0), policy);
    test.expect(initialized, "initialize a nonzero result cache and separate producer/consumer GPU resources");
    if (initialized) {
        test.expect(producer.stream != consumer.stream, "handoff crosses task streams after explicit publication");
        ResultAtom first = makeResultAtom(1001);
        ResultAtom second = makeResultAtom(1002);
        ResultAtom third = makeResultAtom(1003);
        std::size_t uploadedBytes = 0;
        if (produceResult(test, resultCache, producer, d_algoOutput, first, 0x31, CacheStatus::CacheFill)) {
            // Private output can be reused after A completes; the cache and atom
            // preserve the original result independently of that allocation.
            test.expect(cudaMemsetAsync(d_algoOutput, 0, RESULT_BYTES, producer.stream) == cudaSuccess && cudaStreamSynchronize(producer.stream) == cudaSuccess, "reuse producer private output after completed handoff");
            consumeResult(test, resultCache, consumer, d_consumerOutput, first, CacheStatus::CacheHit, uploadedBytes);
            test.expect(uploadedBytes == 0, "initial downstream hit skips CPU-to-GPU transfer");
        }
        if (produceResult(test, resultCache, producer, d_algoOutput, second, 0x72, CacheStatus::CacheFill)) {
            {
                GpuDataAccess held = resultCache.getCacheData(second.metadata, makeCacheRequest(producer));
                test.expect(held.status() == CacheStatus::CacheHit, "hold the replacement result while the consumer requests an evicted ID");
                consumeResult(test, resultCache, consumer, d_consumerOutput, first, CacheStatus::TaskFallback, uploadedBytes);
                test.expect(held.freeCacheData(true), "release replacement reader after consumer fallback");
            }
            consumeResult(test, resultCache, consumer, d_consumerOutput, first, CacheStatus::CacheFill, uploadedBytes);
            consumeResult(test, resultCache, consumer, d_consumerOutput, first, CacheStatus::CacheHit, uploadedBytes);
            test.expect(uploadedBytes == 2 * RESULT_BYTES, "evicted result is recovered from CPU atom for fallback and cache refill");
        }
        {
            GpuDataAccess held = resultCache.getCacheData(first.metadata, makeCacheRequest(consumer));
            test.expect(held.status() == CacheStatus::CacheHit, "keep result entry busy to force producer fallback");
            if (produceResult(test, resultCache, producer, d_algoOutput, third, 0xa5, CacheStatus::TaskFallback)) {
                consumeResult(test, resultCache, consumer, d_consumerOutput, third, CacheStatus::TaskFallback, uploadedBytes);
            }
            test.expect(held.freeCacheData(true), "release entry after producer and consumer fallback");
        }
        consumeResult(test, resultCache, consumer, d_consumerOutput, third, CacheStatus::CacheFill, uploadedBytes);
        consumeResult(test, resultCache, consumer, d_consumerOutput, third, CacheStatus::CacheHit, uploadedBytes);
        test.expect(uploadedBytes == 4 * RESULT_BYTES, "a producer fallback still supports downstream CPU recovery and later hits");
        test.expect(resultCache.statisticsSnapshot().eviction == 3, "result handoff covers actual eviction rather than reset-only misses");
        test.expect(first.ready && second.ready && third.ready && first.data.front() == 0x31 && second.data.front() == 0x72 && third.data.front() == 0xa5, "distinct result IDs retain independent authoritative CPU payloads");

        test.expect(resultCache.resetCache(), "start a clean final-use result interval");
        const std::size_t uploadsBefore = uploadedBytes;
        for (std::uint64_t dataId : {1004U, 1005U}) {
            ResultAtom terminal = makeResultAtom(dataId);
            if (produceResult(test, resultCache, producer, d_algoOutput, terminal, 0x6b, CacheStatus::CacheFill)) {
                consumeResult(test, resultCache, consumer, d_consumerOutput, terminal, CacheStatus::CacheHit, uploadedBytes, CacheRetention::Discard);
            }
        }
        const CacheStatistics statistics = resultCache.statisticsSnapshot();
        test.expect(uploadedBytes == uploadsBefore && statistics.discard == 2 && statistics.eviction == 0 && statistics.fillSucceeded == 2, "final-use consumers discard both policy results so the next producer reuses space without eviction or H2D");
    }
    test.expect(resultCache.release(), "release result cache after every producer/consumer lease finishes");
    if (d_consumerOutput != nullptr) {
        test.expect(cudaFree(d_consumerOutput) == cudaSuccess, "release consumer private output");
    }
    if (d_algoOutput != nullptr) {
        test.expect(cudaFree(d_algoOutput) == cudaSuccess, "release producer private output");
    }
    const bool consumerReleased = releaseAccessResources(consumer);
    const bool producerReleased = releaseAccessResources(producer);
    test.expect(consumerReleased && producerReleased, "release both task streams and fallback buffers");
}

}  // namespace gpuinfra_tests
