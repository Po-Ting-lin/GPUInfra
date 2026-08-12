#include "GraphStuff.h"

#include <algorithm>
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
    if (initialized || config.numaNode < 0 || config.gpuIds.empty() || config.taskInstancesPerGpu == 0 || config.runtime.inBytes == 0 || cancellation == nullptr || sink == nullptr || cancellation->load(std::memory_order_acquire)) {
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
    if (!initialized || phaseActive || stopping || cancellation->load(std::memory_order_acquire)) {
        return false;
    }

    std::vector<std::unique_ptr<FrameSlot>>& phaseSlots = slotsForPhase(phase);
    std::deque<FrameSlot*> pendingFrames;
    try {
        for (const std::unique_ptr<FrameSlot>& slot : phaseSlots) {
            if (slot->state != FrameState::Prepared) {
                return false;
            }
            pendingFrames.push_back(slot.get());
        }
    } catch (const std::exception&) {
        cancellation->store(true, std::memory_order_release);
        return false;
    }
    readyFrames = std::move(pendingFrames);
    for (FrameSlot* slot : readyFrames) {
        slot->state = FrameState::Ready;
        slot->result.ok = true;
    }

    phaseGate = &gate;
    phaseTotal = phaseSlots.size();
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
        if (!initialized && workers.empty() && tasks.empty() && warmupSlots.empty() && timedSlots.empty()) {
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
    warmupSlots.clear();
    timedSlots.clear();
    return unloadSucceeded;
}

int DummyGraph::numaNode() const {
    return config.numaNode;
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
    if (warmupCount > std::numeric_limits<std::uint64_t>::max() - timedCount || config.firstFrameId > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(warmupCount + timedCount)) {
        return false;
    }

    try {
        tasks.reserve(config.taskInstancesPerGpu * config.gpuIds.size());
        int taskId = 0;
        for (int gpuId : config.gpuIds) {
            for (std::size_t instance = 0; instance < config.taskInstancesPerGpu; ++instance) {
                tasks.push_back(std::make_unique<DummyTask>(taskId, config.numaNode, gpuId, config.executionModel, config.runtime));
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

        warmupSlots.reserve(warmupCount);
        timedSlots.reserve(timedCount);
        std::uint64_t nextId = config.firstFrameId;
        for (std::size_t index = 0; index < warmupCount; ++index) {
            warmupSlots.push_back(std::make_unique<FrameSlot>(nextId, config.numaNode, FramePhase::Warmup, config.runtime));
            ++nextId;
        }
        for (std::size_t index = 0; index < timedCount; ++index) {
            timedSlots.push_back(std::make_unique<FrameSlot>(nextId, config.numaNode, FramePhase::Timed, config.runtime));
            ++nextId;
        }
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool DummyGraph::unloadOnNumaNode() {
    bool ok = GpuContextManager::pinCurrentThreadToNumaNode(config.numaNode);
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
        FrameSlot* frame = nullptr;
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

            frame = readyFrames.front();
            readyFrames.pop_front();
            task = freeTasks.back();
            freeTasks.pop_back();
            gate = phaseGate;
            frame->state = FrameState::Executing;
            ++inFlight;
            lastMaxInFlight = std::max(lastMaxInFlight, inFlight);
        }

        gate->wait();
        bool succeeded = false;
        if (!cancellation->load(std::memory_order_acquire)) {
            succeeded = task->execute(*frame);
        }
        if (!succeeded) {
            cancellation->store(true, std::memory_order_release);
        }
        if (cancellation->load(std::memory_order_acquire)) {
            frame->state = succeeded ? FrameState::Cancelled : FrameState::Failed;
            frame->result.ok = false;
        }
        else {
            frame->state = FrameState::Completed;
        }
        sink->deliver(frame->result);

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
        FrameSlot* frame = readyFrames.front();
        readyFrames.pop_front();
        frame->state = FrameState::Cancelled;
        frame->result.ok = false;
        sink->deliver(frame->result);
        ++phaseTerminal;
    }
    phaseSucceeded = false;
}

void DummyGraph::cancelPreparedFramesLocked() {
    auto cancelSlots = [this](std::vector<std::unique_ptr<FrameSlot>>& slots) {
        for (const std::unique_ptr<FrameSlot>& frame : slots) {
            if (frame->state == FrameState::Prepared) {
                frame->state = FrameState::Cancelled;
                frame->result.ok = false;
                sink->deliver(frame->result);
            }
        }
    };
    cancelSlots(warmupSlots);
    cancelSlots(timedSlots);
}

void DummyGraph::finishPhaseIfCompleteLocked() {
    if (phaseActive && readyFrames.empty() && inFlight == 0 && phaseTerminal == phaseTotal) {
        phaseActive = false;
        phaseGate = nullptr;
        phaseCondition.notify_all();
    }
}

std::vector<std::unique_ptr<FrameSlot>>& DummyGraph::slotsForPhase(FramePhase phase) {
    return phase == FramePhase::Warmup ? warmupSlots : timedSlots;
}
