# GraphPLAN (N-Algo N-Thread) – Scaled GPU Infrastructure for ~6 Algorithms

Variant of `GraphPLAN.md`, scaled to **M ≈ 6 algorithms × K ≈ 3–4
threads per GPU**. It keeps the same register-thread, own-the-slot model
and the same Graph integration.

**What changed vs GraphPLAN.md (M=2, K=4):**

| Dimension | GraphPLAN.md | This document |
| :--- | :--- | :--- |
| Algorithms per GPU | 2 (CEL, SDD) | **~6** (CEL, SDD, plus 3–4 future algorithms) |
| Threads per GPU (K) | 4 (recommended) | **3–4** (lower K preferred — memory scales as K × M) |
| Per-GPU buffer sets | K × M = 8 | **K × M ≈ 18–24** – multiplicative growth |
| `IAlgo::run()` | Kernel + D2H combined | **Split: `launchKernels()` + `launchD2H()`** (batch compute before DMA) |
| `ThreadSlot` | Input + stream only | **+ shared device scratch** (`d_scratch`) across algorithms |
| `configureAndAlloc()` safety | Not thread-safe | **Must be cold-path only, serialised** |
| Stream chain per frame | H2D → 2 kernels → 2 D2H copies | H2D → **6 kernels → 6 D2H copies** |

All other architectural decisions (no queue, no dispatcher, no internal
threads, NUMA discipline, and a single `cudaStreamSynchronize()` per frame) are
unchanged from GraphPLAN.md.

---

## 1. Context

### Hardware (Lumina)

| Item | Value |
| :--- | :--- |
| NUMA nodes | 2 |
| GPUs | 2 × RTX 3080 (one per NUMA node) |
| GPU arch | Ampere (GA102), consumer GeForce |
| Interconnect | PCIe (no NVLink between the 3080s) |
| MIG support | No (datacenter cards only) |
| MPS | Not part of the plan |

The pipeline is **compatible with multiple GPUs per NUMA node**; the
default deployment is 1 GPU per node (above), but larger boxes (e.g. 3
GPUs per node × 2 nodes = 6 GPUs total) work without architectural
change. The NUMA-to-context map is one-to-many, GPU-to-node mapping is
probed from sysfs, and `register_thread()` accepts an optional `gpu_hint` to pick a specific
GPU when several share one node. See §3.1 and §5.

### Workload

- **~6 algorithms** in steady state: **CEL**, **SDD**, plus 3–4 future algorithms.
- Each algorithm is a per-frame GPU pipeline: **H2D → kernel → D2H**.
- **All algorithms consume the same input frame** (single H2D, fan-out to
all algorithms on the same stream).
- Behaviour repeats on every incoming frame.
- Memory footprint scales as **K × M** (threads × algorithms). With K=3–4 and
M=6, this is 18–24 buffer sets per GPU – the dominant
constraint on K.

### Confirmed requirements

| Topic | Decision |
| :--- | :--- |
| Priority | **Throughput first** (batch / offline) |
| Latency | Per-frame latency is not constrained    |
| Process model | **Single multi-threaded process** |
| Implementation | C++/CUDA from scratch            |
| Frame fan-out | Same frame fed to all algorithms  |

---

## 2. Why this shape, why not a queue + dispatcher

The PLAN.md design assumes the infrastructure must spawn its own worker threads, pull frames from an external producer, and route them to whichever GPU is least loaded. That makes sense when GPU-infra is the **owner** of the parallelism. In our actual deployment it isn't:

- The AOI **Graph framework** already runs N orchestrator threads. Each one walks a `FIG_AtomContainer` and calls into algorithm tasks (`I2IDetector::Execute`, `LenaExecuter::Execute`, ...) **synchronously**.
- The current GPU module (`GpuI2I`) is a per-thread object owned by the algorithm task; its threading model is documented in `GpuI2I.md`.
- Each Graph worker thread creates its own `GpuI2I*`, holds it for the scan lifetime, and drives it on its own CUDA stream.
- One Graph worker = one frame in flight = one stream. The thread blocks at its own `cudaStreamSynchronize()` per frame; the next frame for that thread arrives only after the previous result has been delivered into the `FIG_AtomContainer`.

Given that the **external** scheduler already provides the parallelism, an internal producer thread + job queue + dispatcher just adds a layer that re-introduces the same routing decision the Graph framework already made. Strip it. Keep only what the GPU side actually needs: per-thread streams, per-thread buffers, per-GPU primary context, per-GPU shared const-mem, NUMA discipline, and one process-wide algorithm registry.

Compared with `PLAN.md`, the following components survive or disappear:

| PLAN.md component | Fate | Why |
| :--- | :--- | :--- |
| Producer thread | **dies** | Graph thread IS the producer |
| `JobDispatcher` + dispatch policy | **dies** | No runtime routing – thread bound to GPU at `register_thread()` time |
| `JobQueue` (SPMC mutex+CV+queue) | **dies** | Nothing to queue |
| `SlotPool` acquire/release | **dies** | 1 thread = 1 slot, owned for life |
| `outstanding_frames` atomic | **dies** | Nobody reads it |
| `std::promise` / `std::future` | **dies** | `submit_frame()` returns `JobResult` directly (sync) |
| Backpressure (bounded queue) | natural | Thread blocks on its own `cudaStreamSynchronize()` |
| Frame ordering concerns | **dies** | Each thread is serial within itself |
| `GpuContext` (per GPU) | survives | Holds the algorithm registry, primary CUDA context, and per-GPU constant data |
| `IAlgo` interface | survives | Per-thread state instead of per-slot |
| `NumaUtils` (pin thread, node-of) | survives | Still pin thread before any allocation |
| `PinnedPool` / `DevicePool` | survives | One set per thread (not a shared pool) |
| NUMA discipline (§6) | survives | Unchanged |
| `nsys` / `numastat` verification (§9) | survives | Unchanged |

## 3. Topology

**Frame-level parallelism comes from external Graph threads; all ~6 algorithms are co-located per GPU and chained on one stream per thread.**

```text
Graph thread 0 (NUMA 0) — stream 0 → GPU 0 — H2D → A1..A6 kernels → D2H(A1)..D2H(A6) → sync → deliver
Graph thread 1 (NUMA 0) — stream 1 → GPU 0 — H2D → A1..A6 kernels → D2H(A1)..D2H(A6) → sync → deliver
Graph thread 2 (NUMA 0) — stream 2 → GPU 0 — H2D → A1..A6 kernels → D2H(A1)..D2H(A6) → sync → deliver

Graph thread 3 (NUMA 1) — stream 0 → GPU 1 — H2D → A1..A6 kernels → D2H(A1)..D2H(A6) → sync → deliver
Graph thread 4 (NUMA 1) — stream 1 → GPU 1 — H2D → A1..A6 kernels → D2H(A1)..D2H(A6) → sync → deliver
Graph thread 5 (NUMA 1) — stream 2 → GPU 1 — H2D → A1..A6 kernels → D2H(A1)..D2H(A6) → sync → deliver
```

With **K = 3 threads per GPU** (reduced from GraphPLAN's K=4 default to accommodate M=6 algorithms' memory footprint) and **M = 6 algorithms** per chain, the GPU sees three streams' worth of in-flight work. Each stream's chain is longer (6 kernels + 6 D2H copies vs. 2 + 2), but cross-frame overlap from the other K − 1 streams keeps the pipeline full.

**Why K=3 rather than K=4 with M=6:** each additional thread costs M=6 per-thread buffer sets (output plus any algorithm-specific state). On the RTX 3080's 10 GB, going from K=3 to K=4 adds one full set of six algorithm buffers. Profile to verify that K=3 saturates the GPU; add the fourth thread only if `nsys` shows gaps.

### Comparison of layouts

| Strategy | H2D per frame | GPU util | Notes |
| :--- | :--- | :--- | :--- |
| All algorithms on one GPU, second GPU idle | 1× | 50% | Wastes a GPU |
| Split algorithms across GPU 0 and GPU 1 | 2× (same frame copied twice) | High | Doubles PCIe input traffic |
| **Per-thread chain, multiple threads per GPU (chosen)** | **1×** | **High** | **Same shape as `GpuI2I`; best default** |

The split-algorithm layout wins only if one kernel alone saturates the RTX 3080's SMs; measure first.

### 3.1 Multiple GPUs per NUMA node

The default Lumina layout pairs each NUMA node with exactly one GPU. Systems with several GPUs on the same CPU socket (typically 2–4 GPUs per NUMA node) use the same design with **no architectural change**: the per-thread chain still owns one stream and runs the full pipeline; there are simply more `GpuContext` instances on each node.

Three pieces make this work:

| Piece | What changes vs. the one-GPU-per-node default |
| :--- | :--- |
| GPU-to-NUMA mapping | Read `cudaDeviceGetPCIBusId()` and `/sys/bus/pci/devices/<BDF>/numa_node` during `init()`; do not assume `gpu_id == numa_node`. |
| NUMA-to-context map | `std::map<int, std::vector<GpuContext*>>` — one entry per node containing the GPUs attached to it. |
| Thread-to-GPU selection | `register_thread(numa_hint, gpu_hint)` resolves to a specific `GpuContext`. With `gpu_hint == -1`, the library picks the least-loaded context on the node; with `gpu_hint >= 0`, the caller selects a specific CUDA device. |

Visually, a multi-GPU-per-node deployment repeats the default diagram horizontally for every GPU on each node:

NUMA 0 (3 GPUs, K=3)
- Graph threads 0..2 — streams → GPU 0 — chains → deliver
- Graph threads 3..5 — streams → GPU 1 — chains → deliver
- Graph threads 6..8 — streams → GPU 2 — chains → deliver

NUMA 1 (3 GPUs, K=3)
- Graph threads 9..11  — streams → GPU 3 — chains → deliver
- Graph threads 12..14 — streams → GPU 4 — chains → deliver
- Graph threads 15..17 — streams → GPU 5 — chains → deliver

Two ways the orchestrator can drive this:

- **Auto** (`gpu_hint == -1`): orchestrator just spawns `K × num_gpus_per_node`
  pinned threads per node; each `register_thread()` call returns the least-loaded context on that node. Even distribution follows naturally.
- **Manual** (`gpu_hint >= 0`): the orchestrator routes specific frame classes to specific GPUs (for example, high-priority frames always land on GPU 0). The library validates that `gpu_hint` belongs to `numa_hint`'s node before binding.

What stays the same:

- One `GpuContext` per GPU (not per node) — primary CUDA context, algorithm registry, and slot list are all per-GPU. Three GPUs on one node means three `GpuContext` instances, each with its own `Cel` and `Sdd`.
- One stream per slot, one D2H copy per algorithm, and one `cudaStreamSynchronize()` per frame.
- NUMA discipline – pinned input still lives on the GPU's NUMA node; scaling out across more GPUs per node just multiplies the per-GPU memory footprint, not the per-GPU latency.

Scaling caveats (hardware, not code):

- All GPUs on one socket share a PCIe root complex; combined H2D bandwidth may saturate it well below the per-GPU PCIe limit. Profile with `nvidia-smi dmon -s pucvmet` per GPU and check the aggregate.
- The per-GPU memory budget is unchanged because each card has its own VRAM, but **per-node host memory scales linearly**: pinned input plus per-algorithm pinned output × K × `num_gpus_per_node`. Include it in the budget.
- Without NVLink (the case for RTX 3080), GPU-to-GPU peer DMA traverses PCIe through the root complex. The per-thread chain never needs peer DMA, so this is acceptable for CEL and SDD; it matters only if a future algorithm requires GPU-to-GPU intermediates.

### How the Graph framework distributes frames

GpuInfra **does not see the frame-to-thread mapping**. The AOI Graph orchestrator owns it: some Graph threads may be NUMA-pinned to node 0, others to node 1, and the orchestrator hands each thread the next frame to process. GpuInfra's responsibilities are to:

- Provide a registration API that lets each Graph thread bind itself to a GPU (matching its NUMA node).
- Return a per-thread slot containing the stream, pinned input, device input, and per-algorithm output buffers.
- Provide a synchronous `submit_frame(frame_ptr, ...)` → `JobResult` call.

The Graph orchestrator must provide *enough threads per GPU* to saturate it (typically 3–4 per RTX 3080); the memory budget sets the hard cap.

## 4. Per-thread design

**Design principle: one CUDA context per GPU; one externally-owned Graph worker thread per registered "slot"; each thread holds its slot for life; one stream per slot.** Across-slot overlap (what drives throughput) comes from K Graph threads driving K different streams concurrently – the GPU sees K streams' worth of in-flight work even though each individual thread runs its own slot's pipeline synchronously. Within a slot the chain `H2D → A1 → A2 → ... → A6 → D2H(A1) → ... → D2H(A6)` runs in one stream, so ordering is implicit and no inter-stage `cudaEvent` plumbing is needed.

With M=6 algorithms the chain is longer, but the model is identical. Two optimisations specific to the M=6 case are documented below:
**kernel/D2H batching** and **shared device scratch**.

```text
GpuContextManager (singleton, process-wide)
├── contexts: vector<unique_ptr<GpuContext>>
├── numa_to_contexts: map<int, vector<GpuContext*>>
└── register_thread(numa_hint, gpu_hint) → ThreadSlot*

GpuContext (one per GPU; NUMA-local to that GPU)
├── gpu_id, numa_node, primary CUDA context
├── algorithms: vector<unique_ptr<IAlgo>>
│   ├── CEL  { parameters + per_thread[K] = { dev_out, pinned_out } }
│   ├── SDD  { parameters + per_thread[K] = { dev_out, pinned_out } }
│   └── A3..A6 use the same shape
├── shared constant data (per-GPU LUTs / kernel modules)
└── thread_slots: vector<unique_ptr<ThreadSlot>>

ThreadSlot (one per registered Graph thread, owned by GpuContext)
├── thread_id: 0..K-1, used to index each algorithm's per_thread[] state
├── owner_tid: registering OS thread ID, used for validation
├── stream: owned cudaStream_t
├── pinned_in: NUMA-local page-locked host input
├── dev_in: shared device input for the frame
└── d_scratch: device scratch shared serially across algorithms
```

### Defaults at a glance

| Quantity | Default | Notes |
| :--- | :--- | :--- |
| GPUs (Lumina) | 2 | One `GpuContext` per GPU |
| Algorithms per GPU (M) | **6** | CEL, SDD, plus 3–4 future algorithms. All share one stream per slot. |
| Graph threads per GPU (K) | **3–4** (was 4) | Memory-bounded: K × M buffer sets must fit in 10 GB. Profile K=3 first. |
| Per-GPU buffer sets | K × M = **18–24** | Each has `dev_out` + `pinned_out`, plus algorithm-specific state if needed. |
| Total Graph threads | K × 2 | Plus whatever else the Graph orchestrator runs on the CPU side. |
| Producer threads | 0 | Each Graph thread is its own producer |
| Background threads owned by GpuInfra | 0 | We spawn nothing |

### Ownership / layering

- `IAlgo` instances are **owned by `GpuContext`** — one instance per `(context, algorithm)` pair, never shared across contexts. Each instance owns its algorithm-specific parameters and `per_thread[K]` output buffers.
- The **`ThreadSlot`** owns the stream and resources shared by every algorithm processing that thread's frame: `pinned_in`, `dev_in`, and `d_scratch`. Its lifetime matches the registered Graph thread.
- The **Graph worker thread** owns neither the CUDA resources nor the algorithms; it holds a non-owning `ThreadSlot*` until it unregisters.

Because each stream's M=6 chain is longer, K=3 is likely to saturate the RTX 3080 where the M=2 design needed K=4.

**Trade-off: K × M memory vs. pipeline depth.** Each additional thread costs M=6 per-thread buffer sets. At M=2 the fourth thread was cheap; at M=6 it is expensive. Profile K=3 first — the longer chain makes host-side gaps a smaller fraction of total frame time.

### Per-frame algorithm (Graph thread run loop)

Each Graph worker thread, after `register_thread()` returns its `ThreadSlot*`, loops the following synchronously. The Graph orchestrator provides the `next_frame()` and `deliver()` plumbing.

```cpp
ThreadSlot* slot = GpuContextManager::register_thread();             // once

while (auto frame = graph.next_frame()) {
    // 1. Host-side memcpy onto local NUMA node.
    std::memcpy(slot->pinned_in, frame.data, frame.bytes);

    // 2. The single shared H2D for this frame.
    cudaMemcpyAsync(slot->dev_in, slot->pinned_in,
                    frame.bytes, cudaMemcpyHostToDevice, slot->stream);

    // 3a. KERNEL BATCH: launch all algorithm kernels first.
    // Each algorithm writes its per_thread[thread_id].dev_out; intermediate
    // scratch goes through slot->d_scratch (serial on same stream, so
    // algorithm N's scratch is dead before algorithm N+1 starts). Stream
    // semantics guarantee ordering after H2D and between algorithms.
    JobResult r{ .id = frame.id, .ok = true };
    for (auto& algo : slot->ctx->algos) {
        if (!algo->launchKernels(*slot, slot->stream)) { r.ok = false; break; }
    }

    // 3b. D2H BATCH: launch all algorithm D2H copies after kernels.
    // Keeps SMs busy in step 3a, copy engine busy in step 3b.
    // With M=6, this avoids repeated compute/copy transitions.
    if (r.ok) {
        for (auto& algo : slot->ctx->algos) {
            if (!algo->launchD2H(*slot, slot->stream)) { r.ok = false; break; }
        }
    }

    // 4. The single host-side wait. Blocks only this thread.
    cudaStreamSynchronize(slot->stream);

    // 5. Materialise results on this thread (NUMA-local copy) — 6 algorithms.
    if (r.ok) {
        r.outputs.reserve(slot->ctx->algos.size());
        for (auto& algo : slot->ctx->algos)
            r.outputs.push_back(algo->collectResult(*slot));
    }

    // 6. Hand back into the Graph framework.
    graph.deliver(std::move(r));
}

GpuContextManager::unregister_thread(slot);                           // on shutdown
```

No `cudaLaunchHostFunc`. No CUDA-driver callback plumbing. No
`cudaDeviceSynchronize()` — just **one** `cudaStreamSynchronize()` per frame,
scoped to that frame's slot. No per-frame allocation.

**Kernel/D2H batching** is the key M=6 optimisation: steps 3a and 3b
replace `GraphPLAN.md`'s single `algo->run()` loop. The split keeps the
SM pipeline and copy engine pipeline each running contiguously instead of
alternating between compute and copies throughout the frame. See §5 for
the `IAlgo` interface change.

Adding algorithms 7 through N requires one `IAlgo` implementation and one
registration in `main`. The hot path is algorithm-count-agnostic — it
iterates `slot->ctx->algos` regardless of size.

### Why no internal queue is OK

A queue's job is to (a) keep the GPU pipeline full while ingress jitters,
and (b) provide a backpressure / load-balancing point. In our setup:

- **Pipeline stays full**: K Graph threads independently drive K streams.
  An ingress stall on one thread doesn't stall the others; the GPU still
  sees K − 1 streams of in-flight work.
- **Backpressure is natural**: each Graph thread blocks on its own
  `cudaStreamSynchronize()`. If the GPU is busy, the next `next_frame()` call from that thread happens later. The Graph orchestrator above us
  handles the "how do we throttle the source" question – it's the same problem it already solves for CPU-only algorithms.
- **No load-balancing needed**: the Graph orchestrator decides which thread gets which frame. Our per-thread-per-GPU binding is set at `register_thread()` time and never changes. If load gets uneven across GPUs, the fix is for the Graph orchestrator to redistribute frames across threads – a layer above GpuInfra.

If we ever want a queue, we can add it **inside** a single algorithm task
(e.g. a future-style submit on top of the synchronous core), without touching this layer. Starting without it is the simpler shape.

## 5. Source layout

```text
src/
  core/
    NumaUtils.{h,cpp}        libnuma helpers; thread pinning, numa_node_of_ptr(), ctx_for_node()
    PinnedPool.{h,cpp}        per-Graph-thread pinned-host buffers (pinned_in + pinned_out)
    DevicePool.{h,cpp}        per-GPU device buffer pool
    ThreadSlot.{h,cpp}        per-Graph-thread slot; stream + pinned_in + dev_in + d_scratch
    GpuContext.{h,cpp}        one per GPU; primary CUDA context + algorithm registry + thread_slots
    GpuContextManager.{h,cpp} singleton; register_thread / unregister_thread / init / shutdown
  algos/
    IAlgo.h                  algorithm interface (initStatic / configureAndAlloc /
                             scratchBytesNeeded / launchKernels / launchD2H /
                             collectResult / close)
    Cel.{h,cu}
    Sdd.{h,cu}
  main.cpp                   wires up GpuContextManager + algorithms at startup
```

Compared with `PLAN.md` §5, this design drops the entire `dispatch/`
subtree (`JobQueue` and `JobDispatcher`) and replaces `SlotPool` with
`ThreadSlot` plus `GpuContextManager::register_thread()`.

### `IAlgo` contract (N-algorithm variant)

Seven-method lifecycle: `initStatic` / `configureAndAlloc` /
`scratchBytesNeeded` / `launchKernels` / `launchD2H` / `collectResult` /
`close`. Compared to GraphPLAN.md's five-method contract, the changes
are:

| GraphPLAN.md | This document | Why |
| :--- | :--- | :--- |
| `init(AlgoInitInfo)` | `initStatic(AlgoStaticInfo)` + `configureAndAlloc(AlgoRuntimeInfo)` | Splits static GPU binding from buffer allocation. `configureAndAlloc()` is cold-path only and must be serialised (see §7). |
| `run(slot, stream)` | `launchKernels(slot, stream)` + `launchD2H(slot, stream)` | Enables batching: the run loop records every kernel first (step 3a), then every D2H copy (step 3b). |
| `notify(params)` | Deferred | Not needed for the M=6 scale-up. Add it when runtime parameter hot-swapping is required. |

```cpp
struct AlgoStaticInfo {
    int gpuId;
    int numaNode;
};

struct AlgoRuntimeInfo {
    int numThreads;          // K: threads sharing this context
    size_t inBytes;          // shared input size
    int frameWidth;
    int frameHeight;
    int frameDtype;
    AlgoParams params;       // per-algorithm parameter blob
};

class IAlgo {
public:
    virtual ~IAlgo() = default;

    // Cold path. Called once per GPUContext during startup, BEFORE any
    // Graph thread registers. Stores GPU id and NUMA node. Does NOT
    // allocate per-thread buffers — that happens in configureAndAlloc().
    virtual bool initStatic(const AlgoStaticInfo& info) = 0;

    // Cold path. Called when runtime shape / parameters are known.
    // Allocates per_thread[numThreads] = { dev_out, pinned_out };
    // MUST be called from a single thread (main or a serialised cold
    // path) — concurrent calls from Graph threads are a data race.
    // May be called more than once (e.g. recipe change) — must close()
    // old state first.
    virtual bool configureAndAlloc(const AlgoRuntimeInfo& info) = 0;

    // Cold path. Returns the number of bytes this algorithm needs in the
    // shared device scratch buffer (slot->d_scratch). The infrastructure
    // sizes it to max(scratchBytesNeeded()) across all algorithms. Algorithms
    // that need no scratch return 0.
    //
    // The scratch is per-thread (each ThreadSlot has its own d_scratch)
    // and reused across algorithms within a single stream — safe because
    // algorithms run serially (stream ordering). The algorithm must treat the
    // scratch as undefined on entry and must not assume its contents
    // persist across frames.
    virtual size_t scratchBytesNeeded() const = 0;

    // Hot path, step 3a. Record kernel work onto `stream`, reading from
    // slot.dev_in and writing into per_thread[slot.thread_id].dev_out.
    // May use slot.d_scratch for intermediate results. Must NOT record
    // any D2H; that is launchD2H()'s job.
    //
    // Must not call cudaStreamSynchronize() or block. Error detection is
    // launch-time only.
    virtual bool launchKernels(const ThreadSlot& slot, cudaStream_t stream) = 0;

    // Hot path, step 3b. Record D2H of per_thread[slot.thread_id].dev_out
    // → per_thread[slot.thread_id].pinned_out onto `stream`. Called after
    // every algorithm's launchKernels() has been recorded; stream ordering
    // guarantees that all kernels complete before any D2H copy begins.
    virtual bool launchD2H(const ThreadSlot& slot, cudaStream_t stream) = 0;

    // Hot path, step 5. Called after cudaStreamSynchronize(slot.stream).
    // Materialises an owned AlgoOutput from per_thread[slot.thread_id].pinned_out.
    // Must complete quickly because the Graph thread is blocked here.
    virtual AlgoOutput collectResult(const ThreadSlot& slot) = 0;

    // Cold path. Free everything configureAndAlloc() allocated. Called
    // once at context teardown.
    virtual bool close() = 0;
};
```

Adding algorithms 7 through N means implementing `IAlgo`, instantiating it in `main`, and registering
it with each `GpuContext`. **No changes to `ThreadSlot`, no new streams,
no new events, no changes to the Graph thread run loop.** The new algorithm's
kernels extend the kernel batch (step 3a) and its D2H extends the D2H
batch (step 3b).

### Shared device scratch (`d_scratch`)

With M=6 algorithms, per-algorithm-per-thread intermediate buffers scale as
K × M × `intermediateBytes`. The shared scratch eliminates the M multiplier
for intermediates:

| Approach | Device memory for intermediates |
| :--- | :--- |
| Per-algorithm, per-thread (`GraphPLAN.md`) | K × M × maximum intermediate size |
| **Shared scratch (this document)** | **K × maximum intermediate size across algorithms** |

Example: six algorithms, each needing a 4 KB intermediate buffer, with K=3:
- Per-algorithm: 3 × 6 × 4 KB = 72 KB
- Shared: 3 × 4 KB = 12 KB (6× reduction)

Real savings scale with intermediate size. For algorithms with large working
sets (e.g. multi-stage filters with full-frame intermediates), the savings
can be tens of MB per GPU.

**Contract:** the algorithm receives `slot.d_scratch` in `launchKernels()`.
It may write anything into the first `scratchBytesNeeded()` bytes. The
contents are undefined on entry and are overwritten by the next algorithm in
the chain. The scratch is never copied back to the host.

### Result delivery (`collectResult()` flow)

The recommended default for returning algorithm outputs to the Graph orchestrator is **copy at `collectResult()`** — already encoded in step 5 of the run loop and in the `IAlgo::collectResult()` contract.

What the Graph thread does, in order, after `cudaStreamSynchronize()`:

1. For each algorithm, `algo->collectResult(*slot)` copies (or moves from an internally pooled buffer) `per_thread[slot->thread_id].pinned_out` into an owned `AlgoOutput`. The output is appended to the frame's `JobResult`.

2. `graph.deliver(std::move(result))` fills the caller's `FIG_AtomContainer` (or equivalent Graph object) with the `JobResult` outputs.

Why this is the recommended default:

| Property | Why it matters here |
| :--- | :--- |
| Slot stays alive on this thread | Each Graph thread owns its slot for life. There is no pressure to release the slot back to a pool, so `collectResult()` does not block another worker from acquiring it. The trade-off described for Option A in `PLAN.md` is gone. |
| NUMA locality is clean | The copy runs on the NUMA-pinned Graph worker, reading from `pinned_out` on the local node and writing into a buffer that remains on the same node. |
| Lifetime is trivial | Result is fully owned by the caller's `JobResult` – no lease, no view-into-pool, no caller-side discipline. |
| Cost is small | `pinned_out` → owned buffer is a plain `memcpy` at host-memory bandwidth and is typically much faster than the frame's PCIe D2H plus kernel time. |

**When to consider a caller-provided destination instead.** If profiling later shows that the `collectResult()` memcpy is a measurable fraction of frame time — most likely for an algorithm with a **large** output, such as a full-frame image rather than a short list of detections — add a caller-provided destination for that algorithm. Extend the per-frame API with a pinned destination buffer and have `launchD2H()` write directly into it. Other algorithms can retain the default path; treat this as a **per-algorithm optimisation**, not a global policy.

### Registration API (`GpuContextManager`)

The Graph-facing surface for the orchestrator. Static singleton; all methods are thread-safe under an internal mutex on `register_thread` / `unregister_thread`.

```cpp
struct GpuInfraConfig {
    int threads_per_gpu = -1;             // -1 = caller will register on demand;
                                        // >0 = enforce maximum K per context
    bool require_numa = true;             // reject a thread whose NUMA node has no matching GPU
};

struct AlgoFactory {
    std::string name;
    std::function<std::unique_ptr<IAlgo>()> make;
};
```

```cpp
class GpuContextManager {
public:
    // Cold path. Called once from main() before any Graph thread registers.
    // Discovers GPUs, builds the NUMA-node → context map, creates each
    // GpuContext, instantiates each algorithm per context, and calls each
    // algo->initStatic(info).
    //
    // The calling startup thread does this work; it pins itself to the
    // matching NUMA node per context, sets the CUDA device, primes the
    // primary context, then constructs the GpuContext and its algorithms. All
    // first-touch allocations land on the right node.
    static bool init(const GpuInfraConfig& cfg, std::vector<AlgoFactory> algos);

    // Cold path. Called once per Graph worker thread, from that thread's
    // own context. The matching GpuContext is selected (see hint rules
    // below); cudaSetDevice + cudaFree(nullptr) are run; a ThreadSlot is
    // allocated (stream + pinned_in + dev_in, first-touched here);
    // thread_id 0..K-1 is assigned within that context. Each registered
    // algorithm is informed via an internal pre-touch pass so per-thread
    // output buffers are warm.
    //
    // numa_hint:
    //   -1 (default) auto-detect from this thread's CPU affinity mask
    //           (via pthread_getaffinity_np + numa_node_of_cpu).
    //           Upstream orchestrator pin is required for deterministic
    //           node selection.
    //   >= 0 explicit node; register_thread() also applies
    //           numa_run_on_node + numa_set_membind defensively, so an
    //           extra upstream pin is optional/redundant.
    //
    // gpu_hint:
    //   -1 (default) pick the least-loaded GpuContext on the resolved
    //           NUMA node. Returns the unique context in the default
    //           one-GPU-per-node deployment and the least-loaded context
    //           when multiple GPUs share one node.
    //   >= 0 explicit CUDA device id; must live on the resolved
    //           NUMA node, otherwise registration fails.
    //
    // Returns nullptr if registration fails (no GPU for that NUMA node,
    // gpu_hint not on that node, context already at threads_per_gpu cap,
    // OR auto-detect failed and require_numa is true).
    static ThreadSlot* register_thread(int numa_hint = -1, int gpu_hint = -1);

    // Cold path. Called by the Graph worker thread on shutdown or before
    // exit. Tears down the slot's stream + buffers. Algorithms keep their
    // per_thread[] entries valid until close() – slots can be re-bound
    // later if needed, though the common case is "registered for life".
    static void unregister_thread(ThreadSlot* slot);

    // Cold path. Called once from main() at exit, AFTER all Graph workers
    // have unregistered. Walks each context calling algo->close(), tears
    // down GpuContexts, releases CUDA primary contexts.
    static void shutdown();
};
```

The producer-side API is gone — there is none. Graph threads call `register_thread()` once, then drive the hot loop themselves (§4).

**Memory ownership contract.** In step 1 of the run loop, the Graph thread copies `frame_ptr` into `slot.pinned_in` on the worker's NUMA node. The caller **must keep the frame buffer valid until that memcpy returns**. Because submission is synchronous, keeping it valid until the call returns satisfies the contract. There is no zero-copy contract and no producer-side future; ownership is simpler than in `PLAN.md`.

**Backpressure.** Implicit: each Graph thread blocks on its own `cudaStreamSynchronize()`. If the GPU is busy, the next call into GpuInfra from that thread happens later, which throttles the source. No bounded queue.

**Dispatch policy hook.** None at runtime: a Graph thread's GPU binding is fixed at `register_thread()` time. If load gets uneven across GPUs, the Graph orchestrator above us is the right place to fix it. **NUMA routing** still applies, but only at registration: `numa_hint` chooses the `GpuContext`. If `require_numa = false`, a thread with no matching GPU node can fall back to any context; if `true`, registration fails.

---

## 6. NUMA discipline

- If Graph workers call `register_thread()` in auto mode (`numa_hint = -1`), each worker thread must be pinned with `pthread_setaffinity_np` to its intended NUMA node **before** registration so affinity-based detection is deterministic.
- If Graph workers pass explicit `numa_hint >= 0`, pre-pinning is optional: `register_thread()` already applies NUMA pin + membind defensively before any CUDA call.
- `cudaSetDevice(gpu_id)` once per thread inside `register_thread()`, never on the hot path.
- Pinned host buffers (`cudaHostAlloc` / `cudaMallocHost`) are allocated **inside `register_thread()` after NUMA pinning**, so first-touch places the pages on the correct node.
- `PinnedPool` is **per Graph thread**, not a shared pool. Slot-input buffers (`pinned_in`) and per-algorithm outputs (`pinned_out`) all come from the per-thread allocator.
- Verify with `nvidia-smi topo -m` and `numastat -p <pid>` once running: `numastat` should show ~0 `other_node` pages for each Graph thread.

(This follows the NUMA discipline in `PLAN.md`; replacing an internal worker with a Graph worker does not change it.)

---

## 7. Implementation rules (non-negotiable on the hot path)

- No per-frame `cudaMalloc` or `cudaMallocHost`; allocate only during `register_thread()` or `IAlgo::configureAndAlloc()`.
- No synchronous `cudaMemcpy` on the hot path.
- No `cudaDeviceSynchronize`. The **single acceptable host-side wait** is one `cudaStreamSynchronize(slot.stream)` per frame in step 4. It gates `collectResult()` after D2H and blocks only the calling Graph thread, not the other K − 1 threads in the context.
- No unified memory (`cudaMallocManaged`) for frame buffers.
- Create synchronization events, when needed, with `cudaEventCreateWithFlags(&event, cudaEventDisableTiming)`.
- No process-wide queues or mutexes on the hot path. The only infrastructure mutex protects `GpuContextManager::register_thread()` and `unregister_thread()`, both cold-path operations. Algorithms must not introduce hot-path shared mutable state across threads.

### `configureAndAlloc()` thread safety (N-algorithm addition)

With M=6 algorithms, `configureAndAlloc()` allocates K × M buffer sets — a significant amount of device memory. This call **must be serialised**:

- **Preferred:** call `configureAndAlloc()` for all algorithms from the main thread (or one cold-path thread) **before** any Graph worker executes a frame. `DummyGraph::notifyParameters()` must invoke it exactly once per algorithm, not concurrently from multiple Graph threads.
- **For a mid-run recipe change:** drain all K streams first (`cudaStreamSynchronize()` on every slot), call `configureAndAlloc()` from one thread, then resume. Running `configureAndAlloc()` concurrently with `launchKernels()` on the same algorithm instance is a data race on its `per_thread` state.
- The current `DummyGraph::notifyParameters()` implementation is **not thread-safe** if multiple Graph threads call it concurrently. Fix it with a `std::once_flag` per context, or move the call to the `load()` cold path behind a mutex.

## 8. What we deliberately do NOT build (yet)

| Avoid | Reason |
| :--- | :--- |
| DeepStream / Holoscan / Triton | Too much surface area; we don't need codec or model serving. |
| Multi-process design | Single process is simpler and avoids IPC; no isolation need. |
| MPS | Not needed for this single-process, multi-threaded design. |
| MIG | Unsupported on RTX 3080. |
| Unified / managed memory | PCIe page faults hurt throughput on a non-NVLink consumer GPU. |
| Internal `JobQueue` / `JobDispatcher` | The Graph orchestrator above already owns the producer and routing problem. |
| Internal worker threads / thread pool | We don't own the threads; the Graph framework does. |
| Asynchronous submission (`std::future<JobResult>`) | Adds unnecessary overhead because the Graph thread is the only consumer of its result. Re-add only for a concrete asynchronous use case. |
| Cross-thread result handoff | The result is delivered on the same thread that submitted the frame. |
| Frame-content cache (à la `GpuI2I::GpuCache`) | Each frame goes to exactly one thread and all algorithms run there, so there is no cross-thread frame reuse to deduplicate. Re-add only if a future workload reuses frames across threads. |

## 9. Verification & measurement plan

Run after each milestone:

1. **`nsys` timeline** — confirm that H2D, CEL, SDD, and D2H stages
   overlap across slots. Look for gaps on the copy-engine and SM rows.
   With K Graph threads driving K streams, expect K parallel rows in the
   timeline.

2. **`nvidia-smi dmon -s pucvmet`** — inspect per-GPU SM utilisation and PCIe
   RX/TX bandwidth. Compare against theoretical PCIe Gen4 x16 (~25 GB/s
   usable).

3. **End-to-end FPS** vs theoretical limit:
   - PCIe-bound limit: `pcie_bandwidth / frame_size_bytes`.
   - Compute-bound limit: `1 / sum(kernel_time_A1 ... kernel_time_A6)` (all
     six algorithms are serial within one stream, with cross-frame overlap from K streams).
   - **Memory audit**: check device-memory usage with `nvidia-smi`. With K=3–4 and M=6, verify that 18–24 buffer sets leave adequate headroom within 10 GB.

4. **`numastat -p <pid>`** — verify that pinned-host pages are local to each
   Graph thread's NUMA node, with zero or near-zero `other_node` counts.

5. **`cudaDeviceProp::asyncEngineCount`** — log it at startup. A value of at
   least 2 is required for full-duplex H2D/D2H overlap on each GPU.

### Decision rules

**Topology — which GPU runs which algorithm:**
If profiling shows the kernels keep both GPUs busy under the
multi-thread layout – **keep it**.

If one algorithm saturates the SMs and aggregate throughput is bottlenecked
on one GPU, switch to the **split-algorithm** layout (CEL on GPU 0, SDD on
GPU 1), accepting 2× H2D per frame. Implementation: register half the
Graph threads to ctx[0] with only CEL, the other half to ctx[1] with
only SDD. Caller-side decision: No GpuInfra change.

**Stream layout within a slot:**
- **Default: one stream per slot**, full pipeline serialised in it. CEL →
SDD run back-to-back. Simplest code, simplest profiling, no events.
- **Fallback split-kernels onto separate per-slot streams** **only if**
  `nsys` shows both of these:
  1. CEL and SDD each occupy well under half the SMs (clear headroom),
  2. Slot kernel-rows show a measurable gap that concurrent execution
    could fill.
  Implement the fallback as: per slot, one stream for H2D, one stream
  per algorithm kernel, one stream per algorithm D2H copy, plus one event per algorithm.
  Apply per-thread; the rest of the architecture is unchanged.
  Do **not** adopt the fallback unless it measurably beats the default.

**Thread count per GPU (K)** – updated for M=6:

Caller-decided, but the M=6 memory footprint shifts the sweet spot down:

| K per GPU | Buffer sets (K × M) | Expected behaviour on Lumina RTX 3080 (M=6) |
| :--- | :--- | :--- |
| 1 | 6 | Single-stream baseline. The GPU may be idle during host-side result collection and driver work; use for debugging only. |
| 2 | 12 | Two streams overlap H2D, kernels, and D2H. With six-algorithm chains, each stream stays busy longer, so K=2 may suffice for moderate workloads. |
| **3** | **18** | **Recommended default.** Three streams keep the SMs and copy engines busy. The longer six-algorithm chain makes host-side gaps a smaller fraction of total frame time, reducing the need for the fourth thread used at M=2. |
| 4 | 24 | Try only if `nsys` shows pipeline bubbles at K=3. This adds M=6 buffer sets, so verify memory headroom within 10 GB. |
| 5+ | 30+ | Unlikely to help: K=3 with six-algorithm chains already generates enough in-flight work. Memory pressure is the binding constraint. |

Decision rule for production: profile with K=2, K=3, K=4 against an end-to-end benchmark and pick the **lowest K** that hits the FPS target. With M=6, expect K=3 to be optimal in most cases.

**NUMA pinning policy during thread registration:**

- **`require_numa = true`**: refuse to bind a thread to a GPU on the wrong NUMA node. The Graph orchestrator is expected to pin threads correctly upstream.
- **`require_numa = false`**: fall back to any GPU when no matching node exists (for example, on a single-socket development system or when the orchestrator forgot to pin a thread). Useful for debugging, not production.

**`gpu_hint` policy for one-GPU and multi-GPU NUMA nodes:**

The same call works for both deployment shapes:

| Mode | `gpu_hint` | 1 GPU / node (Lumina default) | N GPUs / node |
| :--- | :--- | :--- | :--- |
| Auto | `-1` (default) | Returns the unique context on the node, equivalent to the old single-context lookup. | Picks the least-loaded context among the GPUs on that node; spawning enough threads naturally distributes the load. |
| Manual | `>= 0` | Requests a specific CUDA device ID; the library validates that it belongs to `numa_hint`'s node. | Same. Useful when the orchestrator routes specific frame classes to specific GPUs, such as priority lanes. |

Production rule:

- For **even distribution** across GPUs on each node, pass `gpu_hint = -1`; the library selects the least-loaded context.
- For explicit **affinity**, such as routing one frame class to GPU 0, pass the CUDA device ID. Registration fails if that device is not attached to the requested NUMA node.

---

## 10. Implementation milestones

1. **Skeleton + `IAlgo` interface update**: implement the `launchKernels()` / `launchD2H()` split and `scratchBytesNeeded()` on `IAlgo`. Add `d_scratch` to `ThreadSlot`. Update the run loop to perform the step 3a/3b batching. Verify with one dummy memcpy-passthrough algorithm, K=1, and one GPU.
2. **Integrate CEL and SDD with the new interface**: split the existing `Cel::run()` into `launchKernels()` and `launchD2H()`. Move intermediate-buffer use to `slot.d_scratch`; do the same for SDD. Verify no regression with K=3.
3. **Multi-thread context (K=3)**: validate cross-thread stream concurrency with the batched layout. Confirm the kernel-burst and D2H-burst pattern in the `nsys` timeline, and confirm with `numastat` that pages are NUMA-local.
4. **Memory audit**: measure per-GPU device memory at K=3, M=2. Project the footprint for M=6 and verify 10 GB headroom.
5. **Add algorithms 3–6**: implement each behind `IAlgo` with no core changes. After each addition, verify that the stream chain extends correctly and `nsys` shows no unexpected gaps.
6. **Multi-GPU test**: instantiate `GpuContextManager` with two contexts, one per GPU/NUMA node. Register K=3 threads per context and measure aggregate FPS.
7. **Fix `configureAndAlloc()` serialisation**: add a `std::once_flag` or mutex guard to `DummyGraph::notifyParameters()` so concurrent Graph threads cannot race during buffer allocation.
8. **Sweep K**: profile K=2, K=3, and K=4 with M=6 end-to-end. Pick the lowest K that meets the FPS target within the 10 GB memory budget.

## 11. Open questions for later

- **Where do frames originate (camera, disk, network)?** Same as GraphPLAN.md – NUMA placement of the producer affects step 1 memcpy.
- **Output sink**: are algorithm outputs consumed on the CPU, written to disk, or fed into another GPU stage? M=6 can produce six D2H transfers per frame. If an algorithm's output is never consumed on the host, skip its D2H entirely: return an empty `AlgoOutput` and leave the data on the device for downstream GPU consumers.
- **Steady-state memory budget per GPU (K×M scaling)**: with K=3 and M=6, the per-GPU device memory budget is:
  - Shared input: K × `inBytes` = 3 × 4 MB = 12 MB
  - Per-algorithm output: K × Σ `outBytes[i]`
  - Shared scratch: K × max(`scratchBytesNeeded[i]`)
  - Constant data: Σ `constantBytes[i]` for LUTs, modules, and other per-algorithm state
  - Total must fit in 10 GB with headroom for CUDA runtime overhead.
- **Action**: audit each algorithm's `configureAndAlloc()` footprint before choosing K. If K=3 does not fit, reduce K to 2 or shrink per-algorithm working sets.
- **Will future algorithms share intermediate device buffers (DAG-style), or remain independent?** The shared scratch (`d_scratch`) handles a linear chain. A DAG, where algorithm A's output feeds algorithm B instead of `dev_in`, would require `IAlgo` to expose named tensor inputs and outputs. Defer this until a concrete use case appears.
- **Dynamic K (registration mid-run)?** As in `GraphPLAN.md`, growing K dynamically is difficult. Each new thread requires M=6 new buffer sets, and each algorithm's `per_thread[]` must be extended under a write lock while no thread is executing `launchKernels()`.
- **Kernel/D2H batching trade-off**: the batched layout assumes all algorithms can keep `dev_out` valid until every kernel has been recorded and D2H begins. If a future DAG reuses another algorithm's `dev_out` as scratch, the ordering may require partial D2H interleaving. Measure before adding that complexity.

## 12. Comparison: GraphPLAN.md (M=2) vs this document (M=6)

| Aspect | GraphPLAN.md (M=2, K=4) | This document (M=6, K=3) |
| :--- | :--- | :--- |
| Algorithms per GPU | 2 (CEL, SDD) | **6** |
| Threads per GPU (K) | 4 | **3 (memory-bounded)** |
| Buffer sets per GPU | 8 | **18** |
| `IAlgo::run()` | Combined kernel + D2H | **Split into `launchKernels()` + `launchD2H()`** |
| Intermediate scratch | Per algorithm, per thread | **Shared `d_scratch` per slot** |
| Stream chain per frame | H2D → 2 kernels → 2 D2H copies | **H2D → 6 kernels → 6 D2H copies**, batched |
| Compute/copy transitions per frame | Kernels and D2H copies interleaved | **One kernel batch followed by one D2H batch** |
| `configureAndAlloc()` safety | Not addressed | **Must be serialised (§7)** |
| `scratchBytesNeeded()` | N/A | **New `IAlgo` method** |
| Memory scaling | Linear in K | **K × M — quadratic only when K and M grow together** |
| Graph integration | Unchanged | **Unchanged — same run loop, longer chain** |


## 13. Mental model in one paragraph

This is the same architecture as `GraphPLAN.md`, scaled to **M ≈ 6 algorithms** with three targeted optimisations. Each Graph thread owns a private GPU pipeline represented by one stream, one shared input, and one `ThreadSlot`. For each frame, it copies the input into NUMA-local pinned memory, launches asynchronous H2D, records a **kernel batch** (`launchKernels()` for all six algorithms), and then records a **D2H batch** (`launchD2H()` for all six). This keeps the SMs and copy engines busy in contiguous phases. All six algorithms share a **per-slot device scratch buffer** sized to the largest algorithm's requirement, reducing intermediate storage from K × M × intermediate size to K × maximum intermediate size. After one `cudaStreamSynchronize()`, the thread calls `collectResult()` for all algorithms and delivers the results to the Graph framework. K defaults to **3** rather than 4 because the longer six-algorithm chain creates more in-flight work per stream, while the K × M memory footprint (18 buffer sets at K=3 versus 8 at M=2, K=4) is the binding constraint on an RTX 3080 with 10 GB. `configureAndAlloc()` must run serially on the cold path; concurrent calls from `DummyGraph::notifyParameters()` would race. The architecture still has no internal queue, dispatcher, or worker threads. Adding algorithms 7 through N requires one `IAlgo` implementation and one registration in `main` per algorithm.
