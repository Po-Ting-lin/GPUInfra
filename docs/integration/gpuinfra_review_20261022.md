# GPUInfra 程式碼與文件審查（2026-10-22）

## 範圍與證據

本次審查閱讀 README、`docs/` 主要契約文件，以及核心 cache（`DataCache/`）、
`StaticData`、`GpuContextManager`、`DummyGraph`、`DummyTask` 與 `main.cpp`。
`docs/graph.md`、`docs/design/*`、`cuda_logging.md`、`profiling.md`、HTML 導覽
與 benchmark 結果未逐行審查。

驗證環境：單張 RTX 3080、Release、`-Wall -Wextra -Wpedantic -Wconversion`。

- 審查前：建置 0 警告，CTest 7/7 通過。
- 修正後：建置 0 警告，CTest 7/7 通過；新增測試在還原 #1 修正時會失敗
  （`shutdown` 卡在關閉的 `PhaseGate`），套用修正後通過。
- 本機只有一張 GPU、一個 NUMA node：多 GPU 的 device 還原路徑與多 NUMA graph
  沒有硬體驗證（見 #2、#6）。

核心 cache 的狀態機（lease、RAII 回滾、residency table、LRU/FIFO、Discard）
未發現確定的正確性 bug。以下為需要處理或追蹤的項目。

## 狀態總表

| # | 項目 | 類別 | 狀態 |
| --- | --- | --- | --- |
| 1 | `DummyGraph::shutdown()` 在 `PhaseGate` 未釋放時死鎖 | 程式 | 已修正 |
| 2 | 冷路徑 `cudaSetDevice` 改變呼叫 thread 的 current device | 程式 | 已修正（見範圍） |
| 3 | `freeCacheData()` 遇不合法 `retention` 不釋放 lease | 程式 | 已修正 |
| 4 | 同步失敗只記 log 仍清除 lease | 契約 | 已修正（同 09-24 F2） |
| 5 | 測試為單一 executable／單一 ctest 項目，無 sanitizer | 測試 | 未處理 |
| 6 | core 無獨立測試、無 CI | 測試 | 未處理（同 09-24 F3） |
| 7 | CMake：PUBLIC 連結 cuFFT、無 `install()`、架構預設寫死 | 建置 | 未處理 |
| 8 | `docs/README.md` 漏列 `cache_policies.md` | 文件 | 已修正 |
| 9 | `old_vs_new.md` 與 header 註解過時 | 文件 | 已修正（同 09-24 F4） |
| 10 | 程式 TODO 與 `open_issues.md` 的「已確定」矛盾 | 文件 | 已修正 |
| 11 | README 與 `architecture.md` 大量重複並開始漂移 | 文件 | 未處理 |
| 12 | README 對 archive 文件的描述自相矛盾 | 文件 | 未處理 |
| 13 | 問題追蹤分散、中英文混用 | 文件 | 部分（本文件已在 `open_issues.md` 加入指標） |
| 14 | `results/` 的 log 含本機絕對路徑 | 文件 | 未處理 |
| 15 | `GpuContextManager` 為全域 static 狀態、`shutdown()` 無回傳值 | 設計 | 未處理 |
| 16 | `GpuDataCache.cpp` 狀態重置重複、條件式過長 | 可維護性 | 未處理 |
| 17 | 持鎖時做 I/O、`notify_all` 驚群 | 效能 | 未處理 |
| 18 | `DummyGraph` 持 `schedulerLock` 執行 `executor.run()` | 設計 | 未處理 |

## 1. 已修正項目

### #1 `DummyGraph::shutdown()` 在 gate 未釋放時死鎖

位置：[DummyGraph.cpp](../../demo/Grape/DummyGraph.cpp)，`workerLoop()` 的
`gate->wait()` 與 `shutdown()`。

worker 領取 frame 後會等待 caller 持有的 `PhaseGate`。`shutdown()` 只設
`stopping` 並取消 ready frames，不會釋放 gate；若 `startPhase()` 之後、
`gate.release()` 之前就 shutdown，已領取 frame 的 worker 永遠不會醒來，
`worker.join()` 無法返回。`main.cpp` 每條路徑都會 release gate，所以 demo 沒有踩到，
但這是容易誤用的 API。

修正：`shutdown()` 在 phase 仍 active 時（已設全域 cancellation 後）釋放目前
`phaseGate`；worker 醒來看到 cancellation，不執行 frame，回報 failed terminal result。
`PhaseGate::release()` 以 mutex 保護且可重複呼叫，與 caller 自己的 release 不衝突。

測試：`testGraphShutdownReleasesPhaseGate`——gate 保持關閉、等 worker 領取 frame、
從另一 thread 呼叫 `shutdown()`，限時 10 秒；逾時會自行 release gate 避免測試卡死，
並驗證 8 個 frame 各得到一個 failed terminal result。

### #2 冷路徑 `cudaSetDevice` 的副作用

位置：`GpuDataCacheEntry::initialize()/release()`、`StaticData::init()`、
`StaticGpuData::initialize()/upload()/finalize()/release()`。

這些函式為了配置或釋放 GPU 記憶體呼叫 `cudaSetDevice()`，結束後不還原。
當 library 被整合進既有 AOI thread，呼叫者的 current device 會被悄悄改掉；
文件只說明 hot path 要 `makeTaskCurrent()`，沒有提到這個副作用。

修正：新增 [CudaDeviceGuard.h](../../src/CudaDeviceGuard.h)——RAII，建構時記錄並選擇
目標 device，解構時還原原本的 device；目標已是 current device 時不呼叫
`cudaSetDevice()`。上述冷路徑改用它。

範圍界線：
- `GpuContextManager::init()`（process 初始化）、`makeTaskCurrent()`（刻意設定
  device）、demo 的 `Cel/Sdd/Mi.cu`（task 擁有的執行路徑）與 `tools/` 仍直接呼叫
  `cudaSetDevice()`，不在本次修改內。
- 單 GPU 機器上 guard 的「還原」分支無法被真正觸發（current device 與目標相同）。
  `testCudaDeviceGuard` 在單 GPU 時只驗證 no-op 與錯誤輸入；多 GPU 機器會真正驗證
  選擇與還原。這是已知的驗證缺口。

### #8 `docs/README.md` 索引

補上 [cache_policies.md](../cache_policies.md)，並加入本文件。

### #9 過時敘述（09-24 review 的 F4）

- [old_vs_new.md](old_vs_new.md)：移除「dual fallback、one-sync completion、
  integration adapter 為待辦」的結論，加入 2026-10-22 狀態說明；更新相容性表
  （master/reference fallback、ROI fallback、Loading 等待、統計）、§6 圖示與 §10 最終決定。
  原本的分析段落保留。
- [GpuDataAccess.h](../../src/DataCache/GpuDataAccess.h)：區分「CacheFill 必須完整填入
  共享 payload」與「TaskFallback 為 caller-private，可只填 consumers 讀取的範圍」。
- [GpuDataCache.h](../../src/DataCache/GpuDataCache.h)：補上同一 fallback allocation
  在同 stream 有序條件下可重用的規則。

### #10 TODO 與文件矛盾

[open_issues.md](../open_issues.md) 已寫明 `execute(atom, StaticData&)` 是確定介面，
但 `DummyTask::execute` 仍留 `TODO: not sure where is the correct place for static data`。
已移除該 TODO，以文件為準。

### #3 不合法 `retention`

位置：[GpuDataAccess.cpp](../../src/DataCache/GpuDataAccess.cpp)，`freeCacheData()`。

原本回傳 false 但不釋放 lease，CacheFill 要等到 destructor 才被回滾，期間同 key 的
請求只能等到逾時走 fallback。

修正：不合法的 `retention` 視為失敗的 completion——仍同步 stream、以 Keep 結束 lease
（fill 回滾、hit 釋放 reader），回傳 false，access 變為 Invalid。原本期待「lease 保留給
後續清理」的測試已改為新語意，並新增 fill 回滾與 manager 不被標記 failed 的驗證。

### #4 同步失敗（09-24 review F2）

`freeCacheData()` 與 RAII cleanup 的 `cudaStreamSynchronize` 失敗時，該 access 的 GPU 工作
可能尚未結束，但 lease 仍被清除並使 entry 可重用，且沒有任何 manager 層級的訊號。

修正（採用 F2 的最小要求：失敗向上傳播並阻止新工作，不引入自動恢復或 CPU fallback）：
- 兩條路徑同步失敗都會呼叫 `noteSyncFailure()`：`syncFailures` 加一，manager 標記 failed，
  並喚醒等待中的請求。
- failed manager 對新的 `getCacheData()` 回傳 `Invalid`（含等待中被喚醒的請求），
  `resetCache()` 被拒絕。`release()` 仍可用，`initialize()` 重新開始並清除 failed 與統計。
- `GpuDataCache::hasFailed()`／`StaticData::cacheFailed()` 供 framework 輪詢；
  demo 的統計輸出新增 `sync_failures`。
- 已進行中的其他 lease 照常完成；這次不改變其行為。

測試：`testCacheCompletionFailures`。因為對已銷毀 stream 同步會 segfault，改用
`cudaStreamBeginCapture` 讓同步合法地失敗（非 sticky、不破壞 context），分別覆蓋
explicit completion 與 RAII 兩條路徑，驗證計數、拒絕新請求與 reset、release 後重新
initialize 可恢復。

限制：failed 是 manager 層級的粗粒度狀態，不區分哪個 stream 失敗；真正的 CUDA sticky
error 仍會讓整個 context 不可用，依舊需要 framework 重建。

## 2. 未處理項目

### 測試與建置

**#5**：全部測試編進 `gpuinfra_tests`，只有一個 ctest 項目，自製 `TestContext`；
失敗時無法單獨篩選或重跑，也沒有 sanitizer／`compute-sanitizer` 設定，但 cache 的
並行邏輯很重。

**#6**：core 測試仍依賴 `gpuinfra_demo_support`（`BUILD_TESTING AND GPUINFRA_BUILD_DEMO`）。
專案沒有 CI，測試需要真 GPU 與 NUMA，多 NUMA 路徑被跳過。

**#7**：`gpuinfra` 以 PUBLIC 連結 `CUDA::cufft`（僅診斷使用）；沒有 `install()`／export，
README 卻說 production caller 連結 `gpuinfra`；`build.sh` 的 `CUDA_ARCHITECTURES`
環境變數未記載，預設 sm_86 寫死。

### 文件

**#11**：README（663 行）與 `architecture.md` 重複冷路徑、ownership 表與驗證清單，已出現漂移：
`architecture.md` §12 缺 FIFO、Discard、result cache 與並行 reader 測試；README 對 victim
選擇只寫 O(1)，未註明 FIFO 最壞 O(K)（`architecture.md` 有寫）。建議 README 縮成入口，
細節只留一份。

**#12**：README 的 Design references 把 archive 計畫寫成「specifies the implemented
one-GPU cache」，但 [archive/README.md](../archive/README.md) 說它們不是目前 API 契約。

**#13**：09-24 review 的 F2／F3 沒有進 `open_issues.md`；`open_issues.md` 與 review 為中文，
其餘文件為英文。本次只在 `open_issues.md` 加入指向本文件的指標。

**#14**：`results/profiling/20260920-nvtx/*.stats.log` 含 `/home/ptl/project/...`。
洩漏程度低，但 repository 已有防洩漏 hook，建議一併清理。

### 設計與可維護性

**#15** [GpuContextManager.cpp](../../src/Context/GpuContextManager.cpp)：全域 static 狀態
（原始碼第 81 行已留 Open Issue）；`shutdown()` 被 active task 擋住時為 `void`，
只印 log，caller 無從得知。

**#16** [GpuDataCache.cpp](../../src/DataCache/GpuDataCache.cpp)：狀態重置欄位在
`resetEntriesLocked`、`resetEntry`、`GpuDataCacheEntry::release` 重複；`initialize()`
前置條件整段複製兩次；多行條件式超過 300 字元。建議抽出 `clearState()` 與命名條件。

**#17**：`GPUINFRA_REPORT_FAILURE`（`fprintf`）在 cache `lock` 持有期間呼叫；文件只保證
持鎖時沒有 CUDA call。`addEvictableEntry`／`resetEntry` 每次 `notify_all`，K 大時會驚群。
目前 K 預設為 4，影響不大，需要實測再決定。

**#18**：`DummyGraph::start/changeParameters/stop` 持 `schedulerLock` 執行
`executor.run()`（含 CUDA callback），期間擋住所有 worker；callback 若回頭呼叫 graph API
會死鎖。`changeParameters` 失敗時 registry 已被改動，與 `config.parameters` 不一致
（此時已設全域 cancellation）。

## 3. 建議順序

1. #5、#6：拆出 core 測試 target，讓 AOI 工具鏈能執行；之後再評估 CI 與 sanitizer。
2. #11–#14：文件整併，一次處理以免再次漂移。
3. #15–#18：與真實 framework 整合後，依實際瓶頸與使用方式決定。
