#include "ResultPipeline/ResultPipeline.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "Grape/NumaExecutor.h"

namespace result_pipeline {
namespace {

struct Scheduler {
    std::mutex lock;
    std::condition_variable changed;
    std::deque<std::size_t> freeSlots;
    std::deque<std::size_t> cpuQueue;
    std::deque<std::size_t> consumerQueue;
    std::size_t ready = 0;
    std::size_t active = 0;
    std::size_t peak = 0;
    bool start = false;
    bool failed = false;

    void cancel() {
        std::lock_guard<std::mutex> guard(lock);
        failed = true;
        changed.notify_all();
    }

    bool awaitStart(bool loaded) {
        std::unique_lock<std::mutex> guard(lock);
        ++ready;
        if (!loaded) failed = true;
        changed.notify_all();
        changed.wait(guard, [this] { return start || failed; });
        return !failed;
    }

    bool take(std::deque<std::size_t>& queue, std::size_t& slot, bool producing = false) {
        std::unique_lock<std::mutex> guard(lock);
        changed.wait(guard, [this, &queue] { return failed || !queue.empty(); });
        if (failed) return false;
        slot = queue.front();
        queue.pop_front();
        if (producing) peak = std::max(peak, ++active);
        return true;
    }

    void pass(std::deque<std::size_t>& queue, std::size_t slot, bool completed = false) {
        std::lock_guard<std::mutex> guard(lock);
        queue.push_back(slot);
        if (completed) --active;
        changed.notify_all();
    }
};

}  // namespace

bool runPipeline(int numaNode, const PipelineConfig& config, PipelineMeasurement& measurement) {
    measurement = PipelineMeasurement{};
    if (config.frames == 0 || config.frames > 1000000 || config.entries == 0 || config.entries > 65536 || config.inFlight == 0 || config.inFlight > 65536 || config.bytes == 0 || config.bytes > 16777216 || config.cpuStages == 0 || config.cpuStages > 100 || config.cpuDelay.count() < 0 || config.cacheWait.count() < 0) return false;
    std::vector<int> gpuIds;
    if (!GpuContextManager::gpuIdsForCurrentNumaNode(gpuIds)) return false;
    GpuCacheManager cache;
    if (!cache.initialize(gpuIds, config.bytes, config.entries, config.cacheWait, config.policy)) return false;

    Scheduler scheduler;
    const std::size_t slots = std::min(config.inFlight, config.frames);
    std::vector<ResultAtom> atoms;
    atoms.reserve(slots);
    for (std::size_t index = 0; index < slots; ++index) {
        atoms.push_back(makeAtom(0, config.bytes));
        scheduler.freeSlots.push_back(index);
    }
    std::vector<double> latency(config.frames);
    TaskA producer;
    TaskB consumer;
    std::vector<DummyCPUTask> cpuTasks;
    cpuTasks.reserve(config.cpuStages);
    for (std::size_t index = 0; index < config.cpuStages; ++index) {
        cpuTasks.emplace_back(config.cpuDelay);
    }
    std::vector<std::thread> workers;
    workers.reserve(3);
    std::chrono::steady_clock::time_point started{};
    try {
        for (std::size_t stage = 0; stage < 3; ++stage) {
            workers.push_back(NumaExecutor(numaNode).start([&, stage](bool numaReady) {
                try {
                    const bool loaded = numaReady && (stage == 0 ? producer.load(config.bytes) : stage == 2 ? consumer.load(config.bytes) : true);
                    if (!scheduler.awaitStart(loaded)) return;
                    for (std::size_t frame = 0; frame < config.frames; ++frame) {
                        std::size_t slot = 0;
                        if (stage == 0) {
                            if (!scheduler.take(scheduler.freeSlots, slot, true)) return;
                            ResultAtom& atom = atoms[slot];
                            atom.started = std::chrono::steady_clock::now();
                            atom.metadata.key.frameId = frame + 1;
                            atom.ready = false;
                            atom.cpuStages = 0;
                            atom.checksum = 0;
                            for (std::size_t index = 0; index < config.bytes; ++index) {
                                atom.frame[index] = static_cast<unsigned char>((frame + 1 + index) & 0xffU);
                            }
                            if (!producer.execute(cache, atom)) {
                                scheduler.cancel();
                                return;
                            }
                            scheduler.pass(scheduler.cpuQueue, slot);
                        }
                        else if (stage == 1) {
                            if (!scheduler.take(scheduler.cpuQueue, slot)) return;
                            for (const DummyCPUTask& cpu : cpuTasks) {
                                if (!cpu.execute(atoms[slot])) {
                                    scheduler.cancel();
                                    return;
                                }
                            }
                            scheduler.pass(scheduler.consumerQueue, slot);
                        }
                        else {
                            if (!scheduler.take(scheduler.consumerQueue, slot)) return;
                            if (!consumer.execute(cache, atoms[slot], config.cpuStages, config.retention)) {
                                scheduler.cancel();
                                return;
                            }
                            latency[frame] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - atoms[slot].started).count();
                            ++measurement.completed;
                            scheduler.pass(scheduler.freeSlots, slot, true);
                        }
                    }
                } catch (...) {
                    scheduler.cancel();
                }
            }));
        }
        {
            std::unique_lock<std::mutex> guard(scheduler.lock);
            scheduler.changed.wait(guard, [&] { return scheduler.ready == 3 || scheduler.failed; });
            started = std::chrono::steady_clock::now();
            scheduler.start = true;
            scheduler.changed.notify_all();
        }
    } catch (...) {
        scheduler.cancel();
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    measurement.cache = cache.statisticsSnapshot();
    measurement.consumerHits = consumer.hits;
    measurement.consumerUploadedBytes = consumer.uploadedBytes;
    measurement.peakInFlight = scheduler.peak;
    const bool producerReleased = producer.unload();
    const bool consumerReleased = consumer.unload();
    const bool cacheReleased = cache.release();
    if (scheduler.failed || measurement.completed != config.frames || !producerReleased || !consumerReleased || !cacheReleased) return false;
    std::sort(latency.begin(), latency.end());
    measurement.framesPerSecond = static_cast<double>(config.frames) / seconds;
    measurement.latencyP50Us = latency[(latency.size() - 1) * 50 / 100];
    measurement.latencyP95Us = latency[(latency.size() - 1) * 95 / 100];
    return true;
}

}  // namespace result_pipeline
