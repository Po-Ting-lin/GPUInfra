# Open Issues

## StaticData 與真實 graph 的整合點

真實 framework 可以使用下列介面，將同一個 graph-copy `StaticData` 直接
傳給每次 task execution：

```cpp
bool DummyTask::execute(FrameCpuAtom& atom, StaticData& staticData);
```

因此 task 取得 `StaticData` 的方式已確定。示範版 `DummyGraph` 只加入必要的
生命週期呼叫，以模擬真實 framework 已提供的擴充點，並未改變
[`graph.md`](graph.md) 定義的 scheduler 選擇邏輯。

實際導入前必須確認 framework 能提供：

- graph-copy scope 的 `StaticData::init()`/`release()` 時機；
- 每個 run 開始前、所有舊 execute 已結束且新 execute 尚未開始時，呼叫
  一次 graph-copy-scoped `StaticData::resetCache()`；不需要每個 task 各自呼叫，
  也不需要 frame registration list；
- 保證兩次 reset 之間，相同 `frameId + cameraId` 永遠代表相同 immutable
  bytes；若新 run 重用 identity，漏掉 reset 會造成 stale cache hit；
- 每次 task execute 時取得同一個 graph-copy `StaticData&`；
- framework 原有 scheduler／collections 自行管理 frame 的 ready、in-flight、
  completed、failed 與 cancelled 狀態；`StaticData` 不保存這些狀態；
- frame 完成、失敗或取消時送出 `FrameCpuAtom::result` 的 terminal hook；
- 所有 workers 停止後、CUDA contexts 釋放前的 teardown hook。

若缺少其中任何一項，必須先做 adapter 或採用 framework 正式 extension
point；不可用 process-global mutable singleton 取代。

## 未來兩張 GPU／NUMA

目前初始化明確要求一個 NUMA graph copy 只有一張 GPU。介面仍以 GPU ID
查找 replica，但尚未實作：

- 每個 cache entry 的 per-GPU preallocation；
- P2P capability discovery 與 path warmup；
- local replica miss 時的 lazy P2P 或 staged copy；
- per-replica `Loading`/`Valid` 狀態與同時 readers；
- 逐 GPU VRAM budget 與 transfer diagnostics。

完成前不可靜默忽略第二張 GPU，也不可用 residency 改變 scheduler 選擇。

## Mutable GPU intermediate data

目前 cache 正確性的基礎是所有 task stages 只讀原始 input，而且 cache miss
可由 immutable `FrameCpuAtom` 重新 H2D。若未來某個 stage 產生無法從 CPU
atom 重建、且下游需要的 GPU intermediate，必須另行定義 authoritative
frame-owned output plane、spill/recompute 規則及其 lifetime。Best-effort
`GpuCacheManager` 不能作為這類資料的唯一 owner。

## Non-frame GPU data

`GpuCacheManager`、`GpuCacheEntry` 與 `GpuDataAccess` 的名稱預留未來保存其他
GPU data 的空間。雖然現在已有 `GpuDataKey(frameId, cameraId)`，它仍是
frame-specific；`acquire()` 也只接受
`FrameMetadata`，所有 entries 使用相同的 frame byte size。正式支援 static
data 或 algorithm intermediate 以前，仍須把 key／descriptor 泛化，並定義
不同 payload size、mutability、重建或 fallback source，以及 eviction
lifetime。本次 key 擴充不代表已具備這些功能。

## Cache sizing 與觀測

`GraphConfig::gpuCacheEntries` 目前預設 4，沒有 CLI option。正式 workload
應觀測 cache hit、fill、fallback、eviction 與 transferred bytes，再由量測決定
容量。Capacity 0 可作為 correctness baseline；調大容量只應影響效能與 VRAM，
不應改變結果。
