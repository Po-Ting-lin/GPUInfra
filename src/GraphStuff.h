#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "GpuContext.h"
#include "GpuContextManager.h"
#include "IAlgo.h"
#include "ThreadSlot.h"

// Minimal stand-ins for the external AOI Graph framework.

struct Frame {
    std::uint64_t id = 0;
    const std::uint8_t* data = nullptr;
    std::size_t bytes = 0;
    int numaNode = 0;
};

class GraphFrameSource {
public:
    GraphFrameSource(std::uint64_t totalFrames, std::size_t frameBytes, int numaNode, std::uint64_t firstId)
        : total(totalFrames),
          bytes(frameBytes),
          numa(numaNode),
          idBase(firstId),
          buffer(frameBytes),
          nextId(0) {
        for (std::size_t index = 0; index < buffer.size(); ++index) {
            buffer[index] = static_cast<std::uint8_t>(index % 251U);
        }
    }

    bool nextFrame(Frame& output) {
        const std::uint64_t sequence = nextId.fetch_add(1, std::memory_order_relaxed);
        if (sequence >= total) {
            return false;
        }
        output.id = idBase + sequence;
        output.data = buffer.data();
        output.bytes = bytes;
        output.numaNode = numa;
        return true;
    }

    GraphFrameSource(const GraphFrameSource&) = delete;
    GraphFrameSource& operator=(const GraphFrameSource&) = delete;

private:
    std::uint64_t total;
    std::size_t bytes;
    int numa;
    std::uint64_t idBase;
    std::vector<std::uint8_t> buffer;
    std::atomic<std::uint64_t> nextId;
};

class GraphSink {
public:
    GraphSink() = default;

    void deliver(const JobResult& result) {
        std::lock_guard<std::mutex> guard(lock);
        ++deliveredResults;
        if (!result.ok) {
            ++failedResults;
        }
    }

    std::size_t count() {
        std::lock_guard<std::mutex> guard(lock);
        return deliveredResults;
    }

    std::size_t failureCount() {
        std::lock_guard<std::mutex> guard(lock);
        return failedResults;
    }

    GraphSink(const GraphSink&) = delete;
    GraphSink& operator=(const GraphSink&) = delete;

private:
    std::mutex lock;
    std::size_t deliveredResults = 0;
    std::size_t failedResults = 0;
};

enum class ExecutionModel {
    Batched,
    Interleaved,
};

class DummyGraph {
public:
    DummyGraph(int numaHint, int gpuHint, ExecutionModel model, GraphSink& outputSink)
        : requestedNuma(numaHint),
          requestedGpu(gpuHint),
          executionModel(model),
          sink(&outputSink) {}

    bool registerParameters() {
        parametersRegistered = true;
        return true;
    }

    bool load() {
        if (!parametersRegistered) {
            return false;
        }
        slot = GpuContextManager::registerThread(requestedNuma, requestedGpu);
        if (slot == nullptr) {
            return false;
        }

        result.outputs.resize(slot->ctx->algos.size());
        for (std::size_t index = 0; index < slot->ctx->algos.size(); ++index) {
            if (!slot->ctx->algos[index]->prepareOutput(result.outputs[index])) {
                GpuContextManager::unregisterThread(slot);
                slot = nullptr;
                return false;
            }
        }

        loaded = true;
        return true;
    }

    // Algorithm allocation is intentionally absent here. All algorithms are
    // configured once by GpuContextManager::configure() before any worker
    // registers, avoiding the OCR version's cross-thread reconfiguration race.
    bool notifyParameters() {
        return parametersRegistered && loaded && slot != nullptr;
    }

    bool execute(const Frame& frame) {
        if (!loaded || slot == nullptr || slot->ctx == nullptr || slot->ownerTid != std::this_thread::get_id() || frame.data == nullptr || frame.bytes == 0 || frame.bytes > slot->inBytes) {
            return false;
        }

        result.id = frame.id;
        result.ok = true;

        // Pageable to pinned host input.
        std::memcpy(slot->h_in, frame.data, frame.bytes);
        if (frame.bytes < slot->inBytes) {
            std::memset(static_cast<std::uint8_t*>(slot->h_in) + frame.bytes, 0, slot->inBytes - frame.bytes);
        }

        // Pinned host input to device.
        CUDA_CHECK(cudaMemcpyAsync(slot->d_in, slot->h_in, slot->inBytes, cudaMemcpyHostToDevice, slot->stream), result.ok = false);

        if (executionModel == ExecutionModel::Batched) {
            // Compute batch.
            if (result.ok) {
                for (IAlgo* algorithm : slot->ctx->algos) {
                    if (!algorithm->launchKernels(*slot, slot->stream)) {
                        result.ok = false;
                        break;
                    }
                }
            }

            // D2H batch, after all algorithms' compute operations.
            if (result.ok) {
                for (IAlgo* algorithm : slot->ctx->algos) {
                    if (!algorithm->launchD2H(*slot, slot->stream)) {
                        result.ok = false;
                        break;
                    }
                }
            }
        }
        else if (executionModel == ExecutionModel::Interleaved) {
            if (result.ok) {
                for (IAlgo* algorithm : slot->ctx->algos) {
                    if (!algorithm->launchKernels(*slot, slot->stream)) {
                        result.ok = false;
                        break;
                    }
                    if (!algorithm->launchD2H(*slot, slot->stream)) {
                        result.ok = false;
                        break;
                    }
                }
            }
        }
        else {
            return false;
        }

        // The one per-frame host-side CUDA wait.
        CUDA_CHECK(cudaStreamSynchronize(slot->stream), result.ok = false);

        if (result.ok) {
            for (std::size_t index = 0; index < slot->ctx->algos.size(); ++index) {
                if (!slot->ctx->algos[index]->collectResult(*slot, result.outputs[index])) {
                    result.ok = false;
                    break;
                }
            }
        }

        const bool succeeded = result.ok;
        sink->deliver(result);
        return succeeded;
    }

    bool unload() {
        if (slot != nullptr) {
            GpuContextManager::unregisterThread(slot);
            slot = nullptr;
        }
        loaded = false;
        return true;
    }

    DummyGraph(const DummyGraph&) = delete;
    DummyGraph& operator=(const DummyGraph&) = delete;

private:
    int requestedNuma;
    int requestedGpu;
    ExecutionModel executionModel;
    GraphSink* sink;
    ThreadSlot* slot = nullptr;
    JobResult result;
    bool parametersRegistered = false;
    bool loaded = false;
};
