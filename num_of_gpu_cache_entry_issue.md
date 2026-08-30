# GpuCacheEntry 數量問題

## 狀態

`220` 只代表 demo 預設 workload 的 logical frame 數量：

```text
20 warmup + 200 timed = 220 FrameCpuAtom
```

它不是 `GpuCacheManager` 的 entry 數量，也不需要建立 220 筆 frame registry
或 220 個 device input buffers。GPU cache 容量是獨立設定：

```text
K = GraphConfig::gpuCacheEntries
```

目前預設 `K = 4`；`K = 0` 也合法，表示完全停用 GPU cache，所有 frame
都走 task-private fallback buffer。

## 為何 task instances 少，仍可處理 220 frames

4 個 task instances 限制的是 concurrent `execute()` 數量，不是整個 run 的
frame 總數。每個 instance 完成一次呼叫後即可被 scheduler 重用：

```text
Task instance 0: F0, F4, F8,  ... F216
Task instance 1: F1, F5, F9,  ... F217
Task instance 2: F2, F6, F10, ... F218
Task instance 3: F3, F7, F11, ... F219
```

因此 graph-owned execution state 與 atom-owned result 必須和 task lifetime
分離，但不代表每個 logical frame 都要永久占用一份 GPU buffer。

## 目前模型

```text
StaticData
  ├─ fixed frame layout + resetCache() run boundary
  └─ GpuCacheManager
       ├─ fixed open-addressing residency table [約 2K 至 4K slots]
       ├─ O(1) empty stack + intrusive global LRU
       └─ GpuCacheEntry[K]
            cached metadata + lease/LRU state + persistent GpuReplica
```

- `StaticData` 不保存 frame list 或 per-frame metadata registry。
- `FrameCpuAtom` 擁有 CPU input、metadata 與預配置的 `JobResult`。
- 只要 incoming frame 的 layout 相符，就可嘗試使用 cache；`frameId` 與
  `cameraId` 不必事先登記。
- `GpuCacheEntry` 是 best-effort cache，可被 global LRU 替換，不與某個
  logical frame 永久綁定。
- `TaskGpuResources::d_input` 是每個 task instance 的 persistent fallback
  buffer，只在 `load()`／`unload()` 配置與釋放。

## Fixed table 與 global LRU 如何配合

`GpuCacheManager::acquire()` 先在 fixed open-addressing table 查找目前
resident 或 loading 的 key：

1. key 與完整 metadata 相符且 entry 為 `Valid`：回傳 `CacheHit`，不做 H2D。
2. miss 且有 `Empty` 或 inactive global-LRU victim：將 incoming key 插入
   fixed table，回傳 `CacheFill`，由 `FrameCpuAtom` 重新 H2D。
3. 相同 key 正在 `Loading`，或全部 entries 都有 active leases：立即回傳
   `TaskFallback`，不等待，也不修改 scheduler。
4. `K = 0`：直接回傳 `TaskFallback`。

fixed table 最多保存 K 個 resident/loading keys，初始化時一次配置為至少
`2K` 個 slots，之後不再 grow 或 rehash。查找、linear-probing insert 與 erase
平均為 `O(1)`；erase 使用 backward-shift deletion，避免 tombstone 隨長時間
churn 累積。Empty stack 與 intrusive global LRU 的 victim selection 為
`O(1)`。

hot path 會更新已配置的 slots，但不配置 CPU/GPU memory，也不執行
`cudaMalloc()`／`cudaFree()`。`resetCache()` 在 quiescent cold boundary 清除
table、entry identity/validity 與 LRU state，時間為 `O(K)`，但保留所有 device
allocations。

## 為何 correctness 不要求 220 個 cache entries

目前 algorithms 只讀原始 frame input。Cache miss 時，task 可以從仍有效且
immutable 的 `FrameCpuAtom` 重新 H2D 到 cache entry 或自己的 `d_input`：

```text
cache hit         -> 使用 GpuCacheEntry.GpuReplica.d_data
cache fill        -> FrameCpuAtom -> H2D -> GpuCacheEntry.GpuReplica.d_data
cache unavailable -> FrameCpuAtom -> H2D -> TaskGpuResources.d_input
```

所以 cache 只是避免重複 H2D 的效能優化。K 可以遠小於整個 run 的 logical
frame 數量，且不需要修改 [`graph.md`](graph.md) 的 ready-frame、task 或
worker selection。

## 重要邊界

這個縮小方案成立的必要條件是 GPU payload 可由 immutable CPU atom 重建。
若未來 `TaskA` 產生只能存在 GPU、且 `TaskB` 必須讀取的 mutable intermediate
payload，eviction 後便不能靠原始 CPU frame 重建。該資料必須另建
frame-owned authoritative output plane、明確的 spill/recompute contract，或
具有足夠容量的非 best-effort storage；不能把目前 cache 當成唯一真實資料
來源。

## 結論

- `NumLogicalFrames` 決定 `FrameCpuAtom`、CPU input 與 result storage 數量。
- `ConfiguredGpuCacheEntries` 決定 `GpuCacheEntry`、residency table 與 cache
  device buffer 數量；目前 `K = ConfiguredGpuCacheEntries`。
- `NumTaskInstances` 決定 streams、fallback input、scratch 與 algo-private
  buffers 數量。
- 三者互相獨立；220 個 incoming frames 可以輪流使用 4 個、2 個，甚至 0 個
  GPU cache entries。
