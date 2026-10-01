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

- task construction 後立即且只呼叫一次 parameter-table registration，並在
  初始 values 完整定義後才呼叫 load；
- 明確的 start/stop execution-cycle hooks，且 stop 發生在所有 in-flight
  execute 結束之後、unload 之前；示範版 hooks 只做 lifecycle 狀態守門，
  不改變 CUDA resource lifetime；
- parameter value 真正變更時，在沒有 active execute 的邊界通知每個 task
  instance；不可依賴每個 phase 或每個 run 都固定收到 notify；
- graph-copy scope 的 `StaticData::init()`/`release()` 時機；
- 每個 run 開始前、所有舊 execute 已結束且新 execute 尚未開始時，呼叫
  一次 graph-copy-scoped `StaticData::resetCache()`；不需要每個 task 各自呼叫，
  也不需要 frame registration list；
- 保證兩次 reset 之間，相同 `frameId + cameraId + variantId` 永遠代表相同 immutable
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

`GpuCacheRequest` 已將 GPU、stream、fallback pointer 與容量交由 caller
提供；不同 `GpuCacheManager` instances 可有不同的固定 payload 大小。
`GpuDataAccess::status()` 回傳 CacheStatus，`getStream()` 回傳 request 的
stream。CacheFill／TaskFallback 的上傳或計算由 caller 執行，最後才由
`freeCacheData()` 同步並發布／回滾，不提早發布。

Payload variant 已加入 key；AOI graph owner 的一致 ID 對應仍待接線，
契約見 [payload identity](integration/payload_identity.md)。

目前 demo 的 `StaticData` 仍只有 frame cache，CEL／SDD／MI outputs 尚未
接入 result cache。Key 仍是 `GpuDataKey(frameId, cameraId, variantId)`，descriptor
仍是 `FrameMetadata`。加入 result cache 時，caller 必須定義參數／算法版本
與 key 的關係或安全 reset 時機、可重算來源，以及獨立的 fallback buffer。
每個 manager 仍使用固定 entry size；mutable 或不可重建的 GPU intermediate
仍需另外定義 ownership 與 eviction lifetime。

## Cache sizing 與觀測

`GraphConfig::gpuCacheEntries` 目前預設 4，沒有 CLI option。正式 workload
已有 manager-local hit、fill、fallback 原因、eviction 與等待統計，
定義及 legacy 對照見 [cache_statistics.md](cache_statistics.md)。實際 transferred
bytes 尚需 caller 量測，再由量測決定
容量。Capacity 0 可作為 correctness baseline；調大容量只應影響效能與 VRAM，
不應改變結果。

## Batch completion backlog

目前保留每個 access 各自呼叫 freeCacheData()，不新增 session 或 batch API。
若 master 加上 64 個 reference 都持有 lease，逐一完成會呼叫 65 次 stream
synchronize；這與 fallback buffer 數量無關。同一 stream 的工作已全部提交時，
第一次成功同步已等待這些工作，後續呼叫不會重做 GPU 計算，但仍有呼叫成本。

目前沒有真實 GpuI2I throughput 量測，不能認定成本很大或可忽略。整合後量測
completion CPU 時間、execute 延遲與 throughput；只有確認成本顯著，才考慮
每個 stream 同步一次後逐一 publish／rollback。須保留失敗及 RAII 清理語意，
不要求跨 cache 原子發布，也不提早發布。這不是初次整合的必要條件。

Master／reference buffer 的預配置與安全重用契約見
[old_vs_new.md §5.3](integration/old_vs_new.md#53-caller-owned-masterreference-fallback-buffers)。

## Profiler context-ID mapping backlog

目前僅記錄初始化取得的 driver_context_id 與已知 context handle，
不將 cuCtxGetId() 數值視為 Nsight／CUPTI contextId。未加入 CUPTI 依賴。
未來若需要精確對照，以帶有 Driver ID 的 NVTX 標記包住已知 CUDA 操作，
利用 trace correlation 建立映射，並驗證多 context／重新建立 context 的情境。
目前環境直接在 Nsight 下呼叫 cuptiGetContextId() 曾遇到
CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED，因此不採用該方式。

## Variable-size GPU result cache and memory pool backlog

狀態：需求已記錄，尚未實作；memory pool 的配置策略與 API 尚未定案。
目前 GpuCacheManager 的每個 entry 都預配置固定大小，沒有按實際結果大小
借還區塊的 memory pool。

未來演算法輸出大小由資料決定，多數結果遠小於已知上限，只有低機率接近
上限。若每個 entry 都配置上限，即使增加 validBytes，也不會釋放或共用
未使用空間，無法改善這種分布下的 VRAM 利用率。

建議研究方向：

- 保留目前固定大小的 frame cache；為可變大小的 GPU result cache 加入
  預配置、固定總容量的 memory pool。
- 初始化時配置 GPU storage 與管理資料；熱路徑只借還區塊，不呼叫
  cudaMalloc／cudaFree，也不動態擴張管理容器。
- Pool 管理區塊容量、對齊與回收；result cache 管理 key、有效長度、
  Loading／Valid、reader lease、等待與 eviction。按 byte budget 管理容量，
  大結果可能需要淘汰多個未使用 entry；不可淘汰 Loading 或仍有 reader 的資料。
- 結果大小若能先計算，再借適當區塊直接產生結果；若執行後才知道大小，
  可考慮 caller 預配置上限大小的 scratch，產生後將有效資料 D2D 到 pool。
  後者只節省常駐 cache 空間，仍須計入並行 caller 的 scratch 與額外複製成本。

實作前需確認：大小分布與上限、大小取得時機、最大並行數、區塊策略與
碎片化、空間不足／超過容量時的等待或 fallback、GPU 工作完成後才能回收的
同步契約，以及結果 key、參數版本、可重算來源。不可把 best-effort cache
當成不可重建結果的唯一 owner；也不可只靠事後長度檢查防止 kernel 寫越界。

驗收應比較實際 VRAM 利用率、可保留結果數、eviction／fallback、額外 D2D
與 throughput／延遲，並涵蓋大量小結果混入少量接近上限結果的 workload。
