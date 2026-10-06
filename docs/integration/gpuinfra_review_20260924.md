# GPUInfra 綜合審查（2026-09-24）

## 結論與證據範圍

GPUInfra 已具備 AOI fixed-size frame cache 所需的核心機制；尚不能宣稱完成
production 整合。剩餘工作以 caller 交易清理、共用 owner、variant 對應與
framework 冷路徑排序為主，不需要重寫 cache 或新增 AOI adapter。

本次直接審查工作區的 core、部分 demo、tests、CMake 與文件。
AOI caller 不在本機；其行為來自另一位 agent 對 AOI commit `b96d8ca2ad`
與 GrapeWiki 6.15.8 的 Q1–Q22 回覆，包含補齊的 Q9／Q10。
這些是帶來源位置的外部審查證據，不是本機獨立重現 AOI。

本文件是 review，沒有修改 runtime 實作。工作區已有的 variantId 等未提交修改
包含在審查範圍，不能把 HEAD 的狀態當成本次檢查的完整版本。

## 1. Findings（按處理優先順序）

### F1／P1：負值 NUMA mapping 被直接改成 node 0

修正進度（2026-09-24）：已修正。未知 PCI placement 改查 system-wide node/online；
只有明確單節點才推導其真實 ID。嚴格模式對多節點、缺失／無法讀取或格式錯誤
拒絕初始化。非嚴格模式才保留明確記錄的 node 0 fallback。已補純拓樸測試。
以下為修正前的 finding，保留供追蹤。

位置：[GpuContextManager.cpp](../../src/Context/GpuContextManager.cpp)，
`probeNumaNodeOfGpu()` 的 `return node < 0 ? 0 : node;`。

即使 requireNuma=true，sysfs 讀到負值仍會被當成 0。註解說單節點工作站
可出現這種值，但程式沒有驗證只有一個 NUMA node。
在多 NUMA 部署中，這會讓未確認的 GPU placement 被當成有效 node 0，
削弱「依本 graph NUMA 找唯一 GPU，無法確定就 init 失敗」的契約。

建議：嚴格模式拒絕不確定的 mapping；如需支援單 NUMA 工作站，必須先確認
唯一有效節點並明確記錄推導結果，不直接猜 0。增加純拓樸測試，涵蓋負值、
單節點、多節點與正常 PCI mapping。本機沒有多 NUMA GPU，因此此部署情境
目前是 code-path finding，沒有硬體重現。

### F2／P1 整合契約：同步失敗後不能把 lease 清空當成 GPU 已安全完成

位置：[GpuDataAccess.cpp](../../src/DataCache/GpuDataAccess.cpp)，
`abortUnfinishedAccess()` 與 `freeCacheData()`。

正常 completion 同步失敗會回傳 false 並回滾；RAII abort 同步失敗只記 log，
仍執行 metadata cleanup。兩條路徑都不提供 manager-wide failed 狀態。
因此「access 已 Invalid／計數已歸零」不是成功 GPU drain 或可恢復執行的證明。

這是必須由 caller 遵守的錯誤邊界，並非本次證實了正常路徑的記憶體錯誤。
AOI legacy 又有 deferred CUDA errors 與 getDefects early return，不能原封不動
沿用「下一次 loadMaster 清一下就繼續」的策略。

最小整合要求：失敗狀態向 detector／framework 傳播、阻止新工作；所有交易出口
清理 leases，且 teardown 另外 drain static-only／private-buffer streams。
若 production 要求 CUDA 失敗後繼續接受 requests，才需另設可驗證的恢復／隔離
契約；目前不引入 CPU fallback 或 circuit breaker。

### F3／P2：core-only build 沒有可執行的獨立測試 target

位置：[CMakeLists.txt](../../CMakeLists.txt)，測試建立條件為
`BUILD_TESTING AND GPUINFRA_BUILD_DEMO`；測試連結 gpuinfra_demo_support。

關閉 demo 能避開舊 nvcc 的 C++17 限制，但也無法執行現有 cache／logger／static
測試。這使 AOI 舊工具鏈的驗證停在 library compile/link，而非 core runtime。

建議下一步拆出不依賴 synthetic .cu、DummyGraph 與 ImageSizing 的 core test
support／target，保留 demo tests 為另一組。不需為此修改 public cache API。

### F4／P2：公開契約與比較文件有過時敘述

- [GpuDataAccess.h](../../src/DataCache/GpuDataAccess.h) 把 Fill 與 Fallback
  都描述成必須填滿整個 payload；AOI 指引允許 full-sized private buffer 的 ROI-only
  fallback。應明確限制完整填入要求於 CacheFill，Fallback 由 consumer 讀取範圍決定。
- [GpuDataCache.h](../../src/DataCache/GpuDataCache.h) 要求每個 live payload
  有 independent fallback storage，未明確描述同 stream 有序重用同一 ref allocation。
- [old_vs_new.md](old_vs_new.md) §2／§10 仍列 integration adapter、one-sync
  completion 與 separate constant storage 為剩餘工作，與「不新增 adapter」、
  batch 留 backlog、StaticGpuData 已完成衝突。§5 部分內容雖已更新，結論未同步。

建議整理文件與 header 契約，避免 AOI agent 因舊結論增加不必要功能。
本次新增 review 指出差異，未默默將舊文件內容視為當前決策。

## 2. 已具備的能力與界線

| 項目 | 現況 | AOI 尚需完成 |
| --- | --- | --- |
| Fixed-size cache | 固定 entries、open addressing、inactive LRU、no hot-path allocation | MiB→K、固定 stride/layout 與總 budget |
| 多 callers／多 managers | metadata lock、reader leases、獨立固定 payload managers | 保存正確 request stream／fallback lifetime |
| Movable access | source Invalid、無 GPU copy、目的物件持續保護 entry | 成員保存；檢查 Inputs 的 memset／memcpy／copy 假設 |
| variantId | 完整 key equality/hash、log、variant 隔離測試 | owner 共用精確 mapping；0 不可用來代表所有 processed variants |
| Waiting | Loading/full 共用每次 request deadline，預設 50ms，可設 0 | AOI 第一階段明確設 0；不要將其誤認為整筆交易總預算 |
| Statistics／NVTX | manager outcomes、reasons、wait、scope ranges | H2D bytes、policy bypass、run interval、實際 workload A/B |
| StaticGpuData | 獨立 bytes 配置、upload/finalize、Ready、cold reinit | layout、完整性檢查、owner-first 排序、所有 readers drain |
| Diagnostics | Runtime/Driver/cuFFT 統一 CUDA_CHECK 與已知 context/key log | framework error 傳播；不是自動恢復機制 |
| Toolchain | enum probes、C++17 core-only build | AOI CUDA/compiler/ABI 全矩陣驗收 |

LRU 的可淘汰性為 Valid 且 activeAccesses=0；Loading 不可淘汰。
最近使用排序在 entry 回到 inactive list 時更新。不能把容量夠大當成 pointer lifetime
保證，也不能因上傳已完成就假設 Fill 已發布。

## 3. AOI 回覆帶來的設計收斂

- Cached bytes 是完整 raw 或 undistorted frame；smoothing 進私有 buffers。
  變體識別至少考慮 raw/undistorted、distortion generation/index/rotation。
- 各 instance 可帶不同 distortion 設定；來源未證明同 key 在同 cycle 只會有
  一個 variant，所以不能只靠 run reset。變更 variant 也不能取代 frame ID reset。
- Q10 確認一個 task instance 的完整交易在一次 doExecute 內完成且不重入；
  不新增交易 mutex／多交易容器。不同 host workers 間仍需 CUDA device 設定。
- 標準單 stream 路徑可重用一個 ref fallback；master 長期保留。
  特殊分支需 regression，不以最多 64 leases 推導要 64 個 fallback allocations。
- 重複 ref 可要求同一 Loading key；AOI 設 0 wait 時 fallback 重建。
  Master-as-reference 原本直接重用 master 的路徑保留。
- Q9 確認 initial load 會 notify、後續 notify 在 cycles 之間，但沒有 owner-first
  保證。Ready 僅是守門，不能代替真正的 dependency／barrier。
- finalize 確認上傳 stream 同步，**不確認業務必要的 DZ/table 是否都已上傳**。
  預先 zeroing 不能替代 owner 的資料完整性驗證。
- Grape host callback 結束不自動保證任意 GPU streams 完成。正常 getDefects 的
  同步不能覆蓋所有 early return，必須加入 abortTransaction 與上層失敗清理。
- DZ-only 更新在語意上可不清 frame cache，但目前 StaticData 整體 release/init
  會一起清掉；第一階段接受此行為，無須先做局部更新 API。
- 報告中的 deriveStop 呼叫 Start 須由 AOI agent 核實；本機無 caller 原始碼，
  不能把報告中的疑點寫成已修復的 bug。

## 4. 接下來的最小工作順序

1. 在 GPUInfra 修正 F1 NUMA 負值政策，統一 F4 文件／header 契約。
2. 拆出可獨立建置的 core tests，交由 AOI 正式工具鏈執行。
3. AOI 依 [直接整合指引](aoi_caller_integration.md) 修改現有 GpuI2I／Inputs，
   不新增 adapter；先涵蓋正常與失敗交易 cleanup。
4. 確認 graph owner multiplicity／affinity、load/notify 順序及共用 variant mapping。
5. K>0、waitTimeout=0 執行 correctness 與 A/B；本階段跳過 K=0 整合測試。
6. 有實測瓶頸再研究 batch/event completion、非零等待或 slab。
   Variable-size result pool 是獨立未來需求；PcrResultsCache 不隨此次 frame cache 遷移。

## 5. 驗證與未覆蓋範圍

本次重新建置 NVTX ON，全部 7 個 CTests 通過；包含目前 variantId 名稱、move、
variant collision／eviction／rollback 等測試。多 NUMA case 依硬體條件略過。
NVTX OFF 在前次 variantId rename 後已通過 7/7，本次不重複聲稱全矩陣驗證。

本次未驗證：AOI caller 編譯與輸出、舊 CUDA 正式 toolchain、真實多 NUMA placement、
GPU failure injection、各 AOI 特殊分支、長時間 production throughput。
沒有證據認定 65 次同步一定昂貴，亦沒有證據認定可忽略；維持先量測決策。

Profiler 原始資料可能包含環境 secrets，SQLite／Nsight raw artifacts 留本機；
新 clone 必須啟用 repository hooks。Allowlist 降低子行程暴露，不代表 raw trace 已淨化。
