#include "DummyGraph.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "GpuContextManager.h"

namespace {

bool multiplyFrameCount(std::uint64_t framesPerGpu, std::size_t gpuCount, std::size_t& output) {
    if (gpuCount == 0 || framesPerGpu > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / gpuCount)) {
        return false;
    }
    output = static_cast<std::size_t>(framesPerGpu) * gpuCount;
    return true;
}

FrameMetadata makeFrameMetadata(std::uint64_t frameId, const AlgoRuntimeInfo& runtime) {
    FrameMetadata metadata;
    metadata.key.frameId = frameId;
    metadata.bytes = runtime.inBytes;
    metadata.width = runtime.frameW;
    metadata.height = runtime.frameH;
    metadata.dtype = runtime.frameDtype;
    return metadata;
}

bool validFramePhase(FramePhase phase) {
    return phase == FramePhase::Warmup || phase == FramePhase::Timed;
}

}  // namespace

void PhaseGate::wait() {
    std::unique_lock<std::mutex> guard(lock);
    condition.wait(guard, [this] { return released; });
}

void PhaseGate::release() {
    {
        std::lock_guard<std::mutex> guard(lock);
        released = true;
    }
    condition.notify_all();
}

void GraphSink::deliver(const JobResult& result) {
    std::lock_guard<std::mutex> guard(lock);
    ++deliveredResults;
    if (!result.ok) {
        ++failedResults;
    }
}

std::size_t GraphSink::count() const {
    std::lock_guard<std::mutex> guard(lock);
    return deliveredResults;
}

std::size_t GraphSink::failureCount() const {
    std::lock_guard<std::mutex> guard(lock);
    return failedResults;
}

DummyGraph::DummyGraph(const GraphConfig& graphConfig, GraphSink& outputSink, std::atomic<bool>& globalCancellation)
    : config(graphConfig),
      sink(&outputSink),
      cancellation(&globalCancellation) {}

DummyGraph::~DummyGraph() {
    shutdown();
}

bool DummyGraph::initialize() {
    if (config.gpuIds.size() != 1) {
        std::fprintf(stderr, "[GPUInfra] unsupported graph topology numa=%d gpu_count=%zu; temporary scope requires exactly one GPU per NUMA graph copy\n", config.numaNode, config.gpuIds.size());
        return false;
    }
    if (initialized || config.numaNode < 0 || config.taskInstancesPerGpu == 0 || config.runtime.inBytes == 0 || cancellation == nullptr || sink == nullptr || cancellation->load(std::memory_order_acquire)) {
        return false;
    }
    if (!GpuContextManager::validateGpuIdsForNumaNode(config.numaNode, config.gpuIds)) {
        std::fprintf(stderr, "[GPUInfra] graph GPU/NUMA mismatch numa=%d gpu=%d\n", config.numaNode, config.gpuIds.front());
        return false;
    }
    if (config.taskInstancesPerGpu > std::numeric_limits<std::size_t>::max() / config.gpuIds.size()) {
        return false;
    }
    const std::size_t configuredTaskCount = config.taskInstancesPerGpu * config.gpuIds.size();
    if (configuredTaskCount > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (config.graphThreads == 0) {
        config.graphThreads = configuredTaskCount;
    }
    if (config.graphThreads == 0) {
        return false;
    }

    bool setupSucceeded = false;
    try {
        std::thread setupThread([this, &setupSucceeded] { setupSucceeded = initializeOnNumaNode(); });
        setupThread.join();
    } catch (const std::exception&) {
        cancellation->store(true, std::memory_order_release);
        return false;
    }
    if (!setupSucceeded) {
        cancellation->store(true, std::memory_order_release);
        std::thread teardownThread([this] { unloadOnNumaNode(); });
        teardownThread.join();
        return false;
    }

    try {
        {
            std::lock_guard<std::mutex> guard(schedulerLock);
            freeTasks.reserve(tasks.size());
            for (const std::unique_ptr<DummyTask>& task : tasks) {
                freeTasks.push_back(task.get());
            }
        }
        workers.reserve(config.graphThreads);
        for (std::size_t workerId = 0; workerId < config.graphThreads; ++workerId) {
            workers.emplace_back(&DummyGraph::workerLoop, this);
        }
    } catch (const std::exception&) {
        cancellation->store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> guard(schedulerLock);
            stopping = true;
            workCondition.notify_all();
        }
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers.clear();
        std::thread teardownThread([this] { unloadOnNumaNode(); });
        teardownThread.join();
        return false;
    }

    {
        std::unique_lock<std::mutex> guard(schedulerLock);
        startupCondition.wait(guard, [this] { return readyWorkers == config.graphThreads; });
        if (workerStartupFailed) {
            stopping = true;
            workCondition.notify_all();
        }
    }
    if (workerStartupFailed) {
        for (std::thread& worker : workers) {
            worker.join();
        }
        workers.clear();
        cancellation->store(true, std::memory_order_release);
        std::thread teardownThread([this] { unloadOnNumaNode(); });
        teardownThread.join();
        return false;
    }

    initialized = true;
    return true;
}

bool DummyGraph::startPhase(FramePhase phase, PhaseGate& gate) {
    std::lock_guard<std::mutex> guard(schedulerLock);
    if (!initialized || !validFramePhase(phase) || phaseActive || stopping || cancellation->load(std::memory_order_acquire)) {
        return false;
    }

    bool& phaseSubmitted = phase == FramePhase::Warmup ? warmupSubmitted : timedSubmitted;
    if (phaseSubmitted) {
        return false;
    }

    std::vector<std::unique_ptr<FrameCpuAtom>>& phaseAtoms = atomsForPhase(phase);
    std::deque<ReadyFrame> pendingFrames;
    try {
        for (std::size_t index = 0; index < phaseAtoms.size(); ++index) {
            const std::unique_ptr<FrameCpuAtom>& atom = phaseAtoms[index];
            if (atom == nullptr || !staticData.validateFrame(atom->metadata)) {
                return false;
            }
            pendingFrames.push_back({atom.get()});
        }
    } catch (const std::exception&) {
        cancellation->store(true, std::memory_order_release);
        return false;
    }
    for (const std::unique_ptr<FrameCpuAtom>& atom : phaseAtoms) {
        atom->result.id = atom->metadata.key.frameId;
        atom->result.ok = true;
    }
    readyFrames = std::move(pendingFrames);
    phaseSubmitted = true;

    phaseGate = &gate;
    phaseTotal = phaseAtoms.size();
    phaseTerminal = 0;
    inFlight = 0;
    lastMaxInFlight = 0;
    phaseSucceeded = true;
    phaseActive = phaseTotal != 0;
    if (phaseActive) {
        workCondition.notify_all();
    }
    else {
        phaseCondition.notify_all();
    }
    return true;
}

bool DummyGraph::waitForPhase() {
    std::unique_lock<std::mutex> guard(schedulerLock);
    if (!initialized) {
        return false;
    }
    phaseCondition.wait(guard, [this] { return !phaseActive; });
    return phaseSucceeded && !cancellation->load(std::memory_order_acquire);
}

bool DummyGraph::shutdown() {
    {
        std::lock_guard<std::mutex> guard(schedulerLock);
        if (!initialized && workers.empty() && tasks.empty() && warmupAtoms.empty() && timedAtoms.empty() && !staticData.isInitialized() && staticData.gpuCacheEntryCount() == 0) {
            return true;
        }
        stopping = true;
        if (phaseActive) {
            cancellation->store(true, std::memory_order_release);
            cancelReadyFramesLocked();
        }
        cancelPreparedFramesLocked();
        workCondition.notify_all();
    }

    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers.clear();

    bool unloadSucceeded = false;
    std::thread teardownThread([this, &unloadSucceeded] { unloadSucceeded = unloadOnNumaNode(); });
    teardownThread.join();

    {
        std::lock_guard<std::mutex> guard(schedulerLock);
        initialized = false;
        readyFrames.clear();
        freeTasks.clear();
        phaseGate = nullptr;
        phaseActive = false;
    }
    return unloadSucceeded;
}

std::size_t DummyGraph::taskCount() const {
    return tasks.size();
}

std::size_t DummyGraph::workerCount() const {
    return config.graphThreads;
}

std::size_t DummyGraph::lastMaxConcurrentExecutions() const {
    std::lock_guard<std::mutex> guard(schedulerLock);
    return lastMaxInFlight;
}

bool DummyGraph::initializeOnNumaNode() {
    if (!GpuContextManager::pinCurrentThreadToNumaNode(config.numaNode)) {
        return false;
    }

    std::size_t warmupCount = 0;
    std::size_t timedCount = 0;
    if (!multiplyFrameCount(config.warmupFramesPerGpu, config.gpuIds.size(), warmupCount) || !multiplyFrameCount(config.timedFramesPerGpu, config.gpuIds.size(), timedCount)) {
        return false;
    }
    if (warmupCount > std::numeric_limits<std::size_t>::max() - timedCount) {
        return false;
    }
    const std::size_t totalFrameCount = warmupCount + timedCount;
    if (totalFrameCount > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()) || config.firstFrameId > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(totalFrameCount)) {
        return false;
    }

    try {
        tasks.reserve(config.taskInstancesPerGpu * config.gpuIds.size());
        int taskId = 0;
        for (int gpuId : config.gpuIds) {
            for (std::size_t instance = 0; instance < config.taskInstancesPerGpu; ++instance) {
                tasks.push_back(std::make_unique<DummyTask>(taskId, gpuId, config.executionModel, config.runtime));
                ++taskId;
            }
        }

        for (const std::unique_ptr<DummyTask>& task : tasks) {
            if (!task->load()) {
                return false;
            }
        }
        for (const std::unique_ptr<DummyTask>& task : tasks) {
            if (!task->registerParameters(parameterRegistry)) {
                return false;
            }
        }
        if (!parameterRegistry.setString(DummyTask::NAME_PARAMETER, config.parameters.name) || !parameterRegistry.setBytes(DummyTask::BLOB_PARAMETER, config.parameters.blob) || !parameterRegistry.seal() || !parameterRegistry.snapshot(parameterSnapshot)) {
            return false;
        }
        for (const std::unique_ptr<DummyTask>& task : tasks) {
            if (!task->notifyParameters(parameterSnapshot)) {
                return false;
            }
        }

        warmupAtoms.reserve(warmupCount);
        timedAtoms.reserve(timedCount);
        StaticDataConfig staticDataConfig;
        staticDataConfig.gpuIds = config.gpuIds;
        staticDataConfig.runtime = config.runtime;
        staticDataConfig.gpuCacheEntries = config.gpuCacheEntries;
        std::uint64_t nextId = config.firstFrameId;
        for (std::size_t index = 0; index < warmupCount; ++index) {
            const FrameMetadata metadata = makeFrameMetadata(nextId, config.runtime);
            warmupAtoms.push_back(std::make_unique<FrameCpuAtom>(metadata, config.runtime));
            ++nextId;
        }
        for (std::size_t index = 0; index < timedCount; ++index) {
            const FrameMetadata metadata = makeFrameMetadata(nextId, config.runtime);
            timedAtoms.push_back(std::make_unique<FrameCpuAtom>(metadata, config.runtime));
            ++nextId;
        }
        if (!staticData.init(staticDataConfig)) {
            return false;
        }
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool DummyGraph::unloadOnNumaNode() {
    bool ok = GpuContextManager::pinCurrentThreadToNumaNode(config.numaNode);

    if (!staticData.release()) {
        ok = false;
    }
    warmupAtoms.clear();
    timedAtoms.clear();
    warmupSubmitted = false;
    timedSubmitted = false;

    for (const std::unique_ptr<DummyTask>& task : tasks) {
        if (!task->unload()) {
            ok = false;
        }
    }
    tasks.clear();
    return ok;
}

void DummyGraph::workerLoop() {
    const bool pinned = GpuContextManager::pinCurrentThreadToNumaNode(config.numaNode);
    {
        std::lock_guard<std::mutex> guard(schedulerLock);
        ++readyWorkers;
        if (!pinned) {
            workerStartupFailed = true;
            cancellation->store(true, std::memory_order_release);
        }
        startupCondition.notify_one();
    }
    if (!pinned) {
        return;
    }

    while (true) {
        ReadyFrame readyFrame;
        DummyTask* task = nullptr;
        PhaseGate* gate = nullptr;
        {
            std::unique_lock<std::mutex> guard(schedulerLock);
            workCondition.wait(guard, [this] {
                return stopping || (phaseActive && (cancellation->load(std::memory_order_acquire) || (!readyFrames.empty() && !freeTasks.empty())));
            });
            if (stopping) {
                return;
            }
            if (cancellation->load(std::memory_order_acquire)) {
                cancelReadyFramesLocked();
                finishPhaseIfCompleteLocked();
                continue;
            }

            readyFrame = readyFrames.front();
            readyFrames.pop_front();
            task = freeTasks.back();
            // The task stays exclusively checked out until execute() finishes.
            freeTasks.pop_back();
            gate = phaseGate;
            ++inFlight;
            lastMaxInFlight = std::max(lastMaxInFlight, inFlight);
        }

        gate->wait();
        bool succeeded = false;
        if (!cancellation->load(std::memory_order_acquire)) {
            succeeded = staticData.execute() && task->execute(*readyFrame.atom, staticData);
        }
        if (!succeeded) {
            cancellation->store(true, std::memory_order_release);
        }
        const bool cancelled = cancellation->load(std::memory_order_acquire);
        if (!succeeded || cancelled) {
            readyFrame.atom->result.ok = false;
        }
        deliverFrameResult(*readyFrame.atom);

        {
            std::lock_guard<std::mutex> guard(schedulerLock);
            freeTasks.push_back(task);
            --inFlight;
            ++phaseTerminal;
            if (!succeeded || cancellation->load(std::memory_order_acquire)) {
                phaseSucceeded = false;
                cancelReadyFramesLocked();
            }
            finishPhaseIfCompleteLocked();
            workCondition.notify_all();
        }
    }
}

void DummyGraph::cancelReadyFramesLocked() {
    while (!readyFrames.empty()) {
        ReadyFrame readyFrame = readyFrames.front();
        readyFrames.pop_front();
        readyFrame.atom->result.ok = false;
        deliverFrameResult(*readyFrame.atom);
        ++phaseTerminal;
    }
    phaseSucceeded = false;
}

void DummyGraph::cancelPreparedFramesLocked() {
    auto cancelAtoms = [this](std::vector<std::unique_ptr<FrameCpuAtom>>& atoms, bool& submitted) {
        if (submitted) {
            return;
        }
        submitted = true;
        for (const std::unique_ptr<FrameCpuAtom>& atom : atoms) {
            if (atom != nullptr) {
                atom->result.ok = false;
                deliverFrameResult(*atom);
            }
        }
    };
    cancelAtoms(warmupAtoms, warmupSubmitted);
    cancelAtoms(timedAtoms, timedSubmitted);
}

void DummyGraph::deliverFrameResult(const FrameCpuAtom& atom) {
    sink->deliver(atom.result);
}

void DummyGraph::finishPhaseIfCompleteLocked() {
    if (phaseActive && readyFrames.empty() && inFlight == 0 && phaseTerminal == phaseTotal) {
        phaseActive = false;
        phaseGate = nullptr;
        phaseCondition.notify_all();
    }
}

std::vector<std::unique_ptr<FrameCpuAtom>>& DummyGraph::atomsForPhase(FramePhase phase) {
    return phase == FramePhase::Warmup ? warmupAtoms : timedAtoms;
}
