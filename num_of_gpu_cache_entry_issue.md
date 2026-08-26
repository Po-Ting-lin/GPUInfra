# GpuCacheEntry 數量問題

## 狀態

已透過「registered frames 與 GPU cache entries 分離」解決。`220` 現在只
是 demo 預設 workload 的 logical frame 數量：

```text
20 warmup + 200 timed = 220 FrameCpuAtom / registered metadata entries
```

`220` 不再是 `GpuCacheManager` 的固定 entry 數量，也不代表需要 220 個
device input buffers。`GraphConfig::gpuCacheEntries` 預設為 4；有效容量為：

```text
min(gpuCacheEntries, configured frames)
```

`gpuCacheEntries = 0` 也是合法設定，代表完全停用 GPU frame cache。

## 為何 task instances 少，仍可處理 220 frames

4 個 task instances 限制的是 concurrent `execute()` 數量，不是整個 run 的
frame 總數。每個 instance 完成一次呼叫後即可被 scheduler 重用：

```text
Task instance 0: F0, F4, F8,  ... F216
Task instance 1: F1, F5, F9,  ... F217
Task instance 2: F2, F6, F10, ... F218
Task instance 3: F3, F7, F11, ... F219
```

因此 graph-owned frame execution state 與 atom-owned result 必須與 task
lifetime 分離，但不代表每個 logical frame 都必須永久佔用一份 GPU input
buffer。

## 目前兩層模型

```text
StaticData
  ├─ registered FrameMetadata[NumConfiguredFrames]
  │    immutable frame-ID index + graph NUMA
  │
  └─ GpuCacheManager
       └─ GpuCacheEntry[min(ConfiguredGpuCacheEntries, NumConfiguredFrames)]
            cached metadata + LRU/lease state + GpuReplica
```

- `StaticData` 保存 immutable registered metadata，並由
  `frame ID -> metadata index` hash 查找；不保存 execution state。
- `FrameCpuAtom` 同時擁有 CPU input、metadata 與預配置的 `JobResult`。
- `FramePhase` 由 `DummyGraph` 的 `warmupAtoms`／`timedAtoms` collections 表示，
  ready/in-flight/terminal bookkeeping 則管理 execution progress。
- `GpuCacheEntry` 是 best-effort GPU cache entry：可被 LRU 替換，並不與某個
  logical frame 永久綁定。
- `TaskGpuResources::d_input` 是每個 task instance 的 persistent fallback
  device buffer，同樣只在 `load()`/`unload()` 配置與釋放。

## Cache hit、fill 與 fallback

`GpuCacheManager::acquire()` 掃描預先配置的少量 entries：

1. 完整 metadata 相符且 entry 為 `Valid`：回傳 `CacheHit`，不做 H2D。
2. miss 且有 `Empty` 或 inactive LRU entry：保留該 entry，回傳
   `CacheFill`；task 從 immutable `FrameCpuAtom` 重新 H2D。
3. 相同 frame 正在 `Loading`，或全部 entries 都有 active accesses：立即回傳
   `TaskFallback`，不等待、不修改 scheduler。
4. capacity 為 0：所有執行都走 `TaskFallback`。

Cache fill 只會在 task stream 成功同步後成為 `Valid`。失敗或 RAII abort
會把 entry 恢復成 `Empty`。Cache hit 失敗不會破壞原本 immutable payload。

Lookup 是 `O(K)`，其中 `K = gpuCacheEntries`，預設 4；這個固定 array scan
不配置記憶體，且比在 hot path 維護會 rehash 的 residency map 更單純。
Registered metadata lookup 仍是平均 `O(1)`。

## 為何 correctness 不再要求 220 GPU cache entries

目前所有 algorithms 都只讀原始 frame input。Cache miss 時，task 可以從仍
有效且 immutable 的 `FrameCpuAtom` 重新 H2D 到自己的 `d_input`，所以 cache
只是避免重複 H2D 的效能優化：

```text
cache hit         -> 使用 GpuCacheEntry.GpuReplica.d_data
cache fill        -> FrameCpuAtom -> H2D -> GpuCacheEntry.GpuReplica.d_data
cache unavailable -> FrameCpuAtom -> H2D -> TaskGpuResources.d_input
```

這讓 GPU cache 容量可小於 simultaneous live logical frames，而不需要修改
`graph.md` 的 ready-frame/task/worker selection，也不需要 entry wait/requeue。

## 重要邊界

這個縮小方案成立的必要條件是 GPU payload 可由 immutable CPU atom 重建。
若未來 `TaskA` 產生只能存在 GPU、且 `TaskB` 必須讀取的 mutable intermediate
payload，eviction 後便不能靠原始 CPU frame 重建。該資料必須另建
frame-owned authoritative output plane、明確的 spill/recompute contract，或
具有足夠容量的非 best-effort storage；不能直接把目前 cache 當成唯一真實
資料來源。

## 結論

- `NumConfiguredFrames` 決定 registered metadata 與 atom-owned CPU/result
  storage 數量。
- `gpuCacheEntries` 決定 best-effort device cache 數量，預設 4。
- `NumTaskInstances` 決定 task streams、fallback input、scratch 與 algo-private
  buffers 數量。
- 三者互相獨立；不再因預設有 220 frames 就配置 220 份 GPU input cache。
