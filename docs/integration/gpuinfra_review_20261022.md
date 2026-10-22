# GPUInfra 程式碼與文件審查（2026-10-22）

## 範圍與證據

本次審查閱讀 README、`docs/` 主要契約文件，以及核心 cache（`DataCache/`）、
`StaticData`、`GpuContextManager`、`DummyGraph`、`DummyTask` 與 `main.cpp`。
`docs/graph.md`、`docs/design/*`、`cuda_logging.md`、`profiling.md`、HTML 導覽
與 benchmark 結果未逐行審查。

驗證環境：單張 RTX 3080、Release、`-Wall -Wextra -Wpedantic -Wconversion`。

- 審查前：建置 0 警告，CTest 7/7 通過。
- 修正後：建置 0 警告，CTest 通過（#5、#6 之後共 50 項）；新增測試在還原 #1 修正時會失敗
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
| 5 | 測試為單一 executable／單一 ctest 項目，無 sanitizer | 測試 | 已修正 |
| 6 | core 無獨立測試、無 CI | 測試 | core 測試已拆出；CI 未處理（同 09-24 F3） |
| 7 | CMake：PUBLIC 連結 cuFFT、無 `install()`、架構預設寫死 | 建置 | 未處理 |
| 8 | `docs/README.md` 漏列 `cache_policies.md` | 文件 | 已修正 |
| 9 | `old_vs_new.md` 與 header 註解過時 | 文件 | 已修正（同 09-24 F4） |
| 10 | 程式 TODO 與 `open_issues.md` 的「已確定」矛盾 | 文件 | 已修正 |
| 11 | README 與 `architecture.md` 大量重複並開始漂移 | 文件 | 已修正 |
| 12 | README 對 archive 文件的描述自相矛盾 | 文件 | 已修正 |
| 13 | 問題追蹤分散、中英文混用 | 文件 | 已修正（語言不翻譯，改為標註） |
| 14 | `results/` 的 log 含本機絕對路徑 | 文件 | 已修正 |
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

### #6 core 測試獨立（09-24 review 的 F3）

原本所有測試編進同一個 `gpuinfra_tests`，連結 `gpuinfra_demo_support`（含 demo 的 `.cu`、
`DummyGraph`、`DummyTask`），因此 `-DGPUINFRA_BUILD_DEMO=OFF` 的 core-only 建置沒有任何
可執行測試，AOI 舊工具鏈只能驗證到編譯與連結。

修正：
- 測試拆成兩個 executable、兩個 ctest 項目：
  - `gpuinfra_core_tests`（`tests/core/`）：cache、policy、completion failure、result cache、
    StaticData、device guard、logger／cuFFT、topology。只連結 `gpuinfra`，不含 demo 標頭。
  - `gpuinfra_demo_tests`（`tests/demo/`，ctest 名稱 `gpuinfra_demo_protocol_tests`）：
    DummyTask／DummyGraph 生命週期、graph 取消與 shutdown、A→CPU→B pipeline。
- `core_test_support` 以獨立的 `gpuinfra_test_support` static library 提供，`makeRuntime()`
  不再用 demo 的 `ImageSizing`（公式為 side = 8 × factor）。demo 測試新增
  `testDemoRuntimeConsistency`，確認它與 `ImageSizing` 一致，避免兩邊漂移。
- `NumaExecutor`（host-only，模擬 framework 的 NUMA affinity）獨立成
  `gpuinfra_numa_executor` library，供 demo 與 core 測試共用；它不屬於 `gpuinfra`。
- `testGpuTopologySelection` 原本放在 graph 測試，實為 core 邏輯，移到
  `tests/core/topology_tests.cpp`。
- `BUILD_TESTING` 開啟時，core-only 建置也會建立並註冊 `gpuinfra_core_tests`。

驗證：完整建置 8/8 通過（原 7 項中的 protocol 測試拆成兩項）；`-DGPUINFRA_BUILD_DEMO=OFF`
建置不呼叫 nvcc，`gpuinfra_core_tests` 通過。原有的每個測試函式都仍被呼叫（另增
`testCudaDeviceGuard`、`testDemoRuntimeConsistency`）。

限制：本機是 CUDA 13.1 單 GPU，未在 AOI 的舊 CUDA／編譯器上驗證；測試仍需真 GPU 與
NUMA topology，因此仍不能在無 GPU 的 CI runner 上執行。

### #5 測試可篩選與 sanitizer

原本每個 executable 只有一個 ctest 項目，失敗時看不出是哪個測試，也無法單獨重跑。

修正（刻意維持精簡）：
- 兩個 executable 改用一張具名測試表（`core_tests.cpp`／`demo_tests.cpp`）與共用的
  `runTestSuite()`：無參數執行全部、`--list` 列出名稱、帶名稱只跑指定測試。
- CMake 從測試表讀出名稱，為每個測試註冊獨立 ctest 項目（`core.*`、`demo.*`，共 44 項），
  `ctest -R` 可篩選，失敗會指出測試名稱。測試表是唯一來源，不需同步第二份清單。
- 新增 `GPUINFRA_SANITIZER=address|thread|undefined`，只套用到 host C++ 目標；device
  程式碼改用 `compute-sanitizer`（未納入 CMake）。

驗證：
- 每個測試以獨立 process 執行皆通過（沒有隱含的順序依賴），ctest 共 50 項通過。
- ThreadSanitizer 跑完整 core 測試與各並行測試（`concurrent_requests`、`wait_wakeup_*`、
  `wait_shared_deadline`、`discard_waiter`）：0 個 data race 警告。
- AddressSanitizer 跑完整 core 測試：通過。

使用限制（已寫入 README）：新核心上 TSAN 需以 `setarch -R` 關閉 ASLR，否則在啟動時因
記憶體映射位址而中止；ASAN 需設定 `protect_shadow_gap=0:replace_intrin=0`，並關閉
leak 偵測（`detect_leaks=0`），否則 `cuInit` 會因 shadow 記憶體而失敗，或回報 libcuda
自身的一次性配置為 leak。沒有納入 CI，也沒有 suppression 檔案。

### #11 README 與 `architecture.md` 重複

README 從 663 行縮為約 280 行，只保留入口資訊：功能摘要、文件導覽、需求、建置與測試、
workload 倍率、簡短的運作概觀（拓樸、排程、cache 樹狀圖、frame 執行流程、失敗與測試）、
原始碼配置、NVTX 與診斷。

- 詳細的 API 用法（lookup 狀態、miss 處理、saved access、獨立 cache、StaticGpuData、
  統計、等待、run boundary）整段移到新文件 [api_guide.md](../api_guide.md)，內容不改寫，
  只調整相對連結。
- 與 `architecture.md` 重複的段落（冷路徑、排程、ownership 表、memory、teardown、
  驗證清單）從 README 移除。移除前逐段比對，README 獨有的內容補進 `architecture.md`：
  GPU 探索與 `requireNuma` 規則（§1）、Batched／Interleaved 與 frame 尺寸（§7）、
  完整的驗證清單（§12，含 FIFO、Discard、completion failure、result pipeline、
  device guard、拓樸測試）；記憶體公式補進 `design/gpu_mem_consumption.md`。
- FIFO 最壞 O(K) 的說明原本只在 `architecture.md`；README 的 O(1) 描述隨 API 段落移到
  api_guide，並指向 architecture 與 cache_policies。

驗證：所有 Markdown 內部連結檢查 0 個壞連結；以 8 字為單位比對舊 README 與現有文件，
未找到的片段經人工檢視，皆為已改寫或已併入 architecture 的內容。

### #12 archive 描述

README 的 Design references 改為 Documentation：列出現行契約，archive 單獨列為
「僅供背景，可能描述已被取代的 API」，與 `archive/README.md` 一致。

### #13 追蹤與語言

- 尚未完成的 review 項目（CI、#7、#15–#18）集中列在
  [open_issues.md](../open_issues.md) 的「Review follow-ups」；已修正項目不重複列出。
- 語言不翻譯：`docs/README.md` 說明預設為英文，並在中文文件旁標註 **(中文)**；
  `open_issues.md`、兩份 review 與 AOI 整合筆記維持中文，視為工作紀錄。

### #14 本機路徑

`results/` 內 6 個檔案的 `/home/ptl/project/GPUInfra` 以 `<repo>` 取代（`commands.json`
仍可解析）；數據未變。`results/README.md` 說明這個佔位符。

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

**#6 的剩餘部分（CI）**：專案沒有 CI，測試需要真 GPU 與 NUMA，多 NUMA 路徑在單 GPU
機器上被跳過。core 測試拆分已完成，見上節。

**#7**：`gpuinfra` 以 PUBLIC 連結 `CUDA::cufft`（僅診斷使用）；沒有 `install()`／export，
README 卻說 production caller 連結 `gpuinfra`；`build.sh` 的 `CUDA_ARCHITECTURES`
環境變數未記載，預設 sm_86 寫死。

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

1. CI：需要有 GPU 的 runner，視環境再決定。
2. #7：CMake 整合（cuFFT 依賴、`install()`、架構預設）。
3. #15–#18：與真實 framework 整合後，依實際瓶頸與使用方式決定。
