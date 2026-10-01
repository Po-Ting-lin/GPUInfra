# AOI 直接整合 GPUInfra 指引

日期：2026-09-24

本文件交給可修改 AOI_AOS 完整原始碼的開發者／agent 執行。
目前工作區沒有 GpuI2I caller，只有 legacy GpuCache 與來源審查回覆；
下列是實作指引，不代表已完成或編譯驗證 AOI 修改。
來源審查基準：AOI commit `b96d8ca2ad`、GrapeWiki 6.15.8。
請以實際 checkout 校正類別、成員名稱與呼叫位置，不依 OCR 行號直接套 patch。

## 1. 範圍與既定決策

直接修改既有 `GpuI2I`／`GpuI2IInputs`／detector 與 graph resource owner，
**不新增 AOI adapter 類別**，也不把交易角色放進 GpuCacheManager。
保留既有 `loadMaster → runRef × N → getDefects` 對外流程與 scheduler。

第一階段從 **K>0、waitTimeout=0** 整合，先跳過 K=0 整合測試。
不自動在配置失敗時降成 K=0；必要配置失敗應傳回初始化錯誤。
暫不加入 batch completion、提早發布、memory pool 或 slab allocation。

已提供的 GPUInfra 能力：

- `GpuCacheRequest` 接收 caller 的 GPU、stream、fallback pointer 與容量。
- `GpuDataAccess` 可 move、不可 copy，可存入預配置成員／陣列。
- `CacheStatus`：Invalid／CacheHit／CacheFill／TaskFallback。
- Key：`frameId + cameraId + variantId`，包含完整 equality 與 hash。
- `freeCacheData(bool)` 同步 stream，發布／回滾並結束 lease。
- 未完成 access 解構，或被 move assignment 覆寫時，同步後 abort，不發布 Fill。

## 2. 修改位置

| 現有位置／候選類別 | 修改內容 |
| --- | --- |
| GpuI2IInputs 的定義檔 | 保存一個 master access 與預配置 ref access 陣列 |
| GpuI2I::loadMaster | 清除殘留交易，解析完整 key，取得並保存 master access |
| GpuI2I::runRef | 取得 ref access，沿用 H2D／預處理／kernel 順序 |
| GpuI2I::getDefects | 提交最後 consumers 後，逐一完成所有有效 accesses |
| GpuI2I 私有方法 | 增加 abortTransaction 與正常 completion helper |
| Detector／processor 的失敗、stop、unload | 呼叫交易清理並 drain GPU 工作，傳播錯誤 |
| 現有 graph-copy resource owner | 持有 StaticData、variant 對應與冷路徑生命週期 |
| Backend 設定處 | 初始化時固定 legacy 或 GPUInfra，同交易不可混用 |

`GeneralStaticDTask` 是 owner 候選，不是已確認選擇。先查部署 `.grf/.hwm`
與 graph clone 行為，證明每個 NUMA graph copy 恰有一份，且 clients 連到同一份。
不得用新 process-global singleton 取代。

## 3. 保存 accesses，保留既有對外 API

以下成員名稱是示意，請合併到真實 GpuI2IInputs，而非建立同名替代類別。
`GPU_I2I_MAX_REF` 使用現有已驗證上限。

```cpp
#include <array>
#include "DataCache/GpuDataAccess.h"

// Add to the existing GpuI2IInputs definition:
GpuDataAccess masterAccess;
std::array<GpuDataAccess, GPU_I2I_MAX_REF> refAccesses;
```

注意：加入 RAII 成員後，必須檢查所有 `memset`、`memcpy`、結構複製、
序列化與 C allocation。不可把整個 GpuI2IInputs 當 POD 清零／搬移；
只重設普通欄位，access 以正常 C++ 建構、move、完成與解構處理。
若此結構確實不能改成非 POD，將 access 成員直接放在既有 GpuI2I 類別，
維持相同交易 lifetime；不為此新增 adapter。

來源回覆確認同一 task instance 由 Grape 序列化，整筆交易在一次 doExecute 內。
因此不新增交易 mutex／多交易容器。Host worker 可能改變，交易不可存於 thread-local。
每次 GPU 操作仍需依既有 context 機制選定正確 GPU；NUMA affinity 不等於 CUDA device。

## 4. Request、metadata 與 variantId

以下變數皆需對應到 AOI 真實欄位；片段不是可直接編譯的完整函式。

```cpp
FrameMetadata metadata;
metadata.key.frameId = frameId;
metadata.key.cameraId = cameraId;
metadata.key.variantId = resolvedVariantId;
metadata.bytes = frameBytes;
metadata.width = frameWidth;
metadata.height = frameHeight;
metadata.dtype = frameDtype;

GpuCacheRequest request;
request.gpuId = gpuId;
request.stream = stream;
request.d_fallback = m_pGPUbufMaster;
request.fallbackBytes = masterAllocationBytes;

inputs.masterAccess = staticData.getCacheData(metadata, request);
```

- `staticData` 必須來自本 graph owner，透過既有 resource port／初始化依賴傳入。
  使用 borrowed reference／pointer 時，owner 必須比 caller 活得久。
- Reference request 改用該 caller 的 `m_pGPUbufRef` 與實際 allocation bytes。
- Legacy full-frame buffer 是 stride × height；checked bytes 計算必須含實際
  元素大小與 padding。FrameMetadata 沒有 stride 欄位，owner 必須保證固定 stride，
  與 StaticDataConfig.runtime 的 bytes／width／height／dtype 一致。
- 不將 camera ID 縮成 unsigned char；保留完整 ID，檢查負值等輸入契約。
- Owner 維護所有 callers 共用的精確 variant 對應。Raw 為 0；undistorted
  根據 table generation、index、rotation 分配不衝突 ID。不可每個 task 自行編號，
  不可用未處理碰撞的 hash 代替身分。詳見 [payload identity](payload_identity.md)。
- 相同完整 key 必須代表相同 CPU 來源／預處理結果。Variant 不能解決 frame ID
  重用或原始影像被修改；新 run 的 quiescent reset 仍需要。

## 5. loadMaster 與 runRef 分支

### loadMaster

1. 確認前次交易已完成；若有殘留，先 abortTransaction，再重設 ref counter／狀態。
   若前次已有 GPU 錯誤，走既有失敗／重新初始化政策，不因清空 handle 就宣告恢復。
2. 解析 variant，取得並保存 master access；此時 stream／fallback／owner 均已初始化。
3. 按下表處理資料，再沿用既有 smoothing／diff consumers。
4. 任何失敗都清理整筆交易，傳回 detector／framework 錯誤。

| Status | Caller 行為 |
| --- | --- |
| Invalid | 不提交使用該 pointer 的工作；abort 並報錯 |
| CacheHit | 用 data() 讀取已完成的 payload；跳過 H2D 與已包含的 undistort |
| CacheFill | 用 writableData() 上傳完整 frame，必要時完成該 variant 的去畸變 |
| TaskFallback | 用 writableData() 沿用既有 ROI／full-row copy 與對應預處理 |

Smoothing 是私有結果，照常寫入 task-private smoothed buffer。
不可對 CacheHit 再做 in-place undistort，亦不可把 cache-owned pointer 當可覆寫 scratch。
讀取使用 `const void* data()`；若舊 kernel wrapper 宣告為可寫 pointer，先確認其實際
讀寫行為，再調整唯讀介面，不以盲目 const_cast 消除新 API 的保護。

CacheFill 必須完成整個可共享 payload。TaskFallback 可保留 AOI 已驗證的 ROI
路徑，但其 consumers 必須只讀已填入區域，allocation 仍至少是完整 payload 容量。
現有 GpuDataAccess header 對「整個 payload 填入」的註解較保守；整合時應明確
區分完整 cache fill 與 caller-private ROI fallback，並用實際 kernel 讀取範圍驗證。

### runRef

1. 在存取陣列之前檢查 ref 數量上限，不等取得 access 後才檢查。
2. 使用完整 ref key、ref fallback，存入尚未使用的 refAccesses slot。
3. 套用同樣的 status 分支，將本 ref 的 consumers 全部提交到 request stream。
4. 下一個 ref 可重用同一 fallback allocation，前提是所有舊內容讀取已排在
   下一次覆寫之前，且在同一 stream。每個 fallback lease 仍分別完成。
5. 保留 master-as-reference 直接使用 master pointer 的既有路徑；它不必新增
   一個 ref lease。不可因 ref slot 為 Invalid 就對此合法路徑報錯。

重複 ref key 仍在 Loading 時，waitTimeout=0 會 fallback 並重新生成資料。
不引入 legacy 的 same-loading-owner 例外。不得依賴 pointer 相同才算相同影像。

## 6. 正常完成與 abortTransaction

下列 helper 使用示意成員 `inputs`；正常 completion 可加在現有 GpuI2I 私有區。

```cpp
bool GpuI2I::finishCacheAccesses(bool submittedSuccessfully) {
    bool succeeded = submittedSuccessfully;
    if (inputs.masterAccess) {
        const bool finished = inputs.masterAccess.freeCacheData(succeeded);
        succeeded = finished && succeeded;
    }
    for (GpuDataAccess& access : inputs.refAccesses) {
        if (access) {
            const bool finished = access.freeCacheData(succeeded);
            succeeded = finished && succeeded;
        }
    }
    return succeeded;
}

void GpuI2I::abortTransaction() noexcept {
    inputs.masterAccess = GpuDataAccess();
    for (GpuDataAccess& access : inputs.refAccesses) {
        access = GpuDataAccess();
    }
    // Also clear borrowed cached_master/cached_ref aliases and transaction state.
    // Preserve the original error so the caller still reports failure.
}
```

- **不能用 short-circuit 跳過後續 access 的完成**；某個失敗也要清理其餘 leases。
  首次失敗後，後續未完成 Fill 以失敗狀態回滾。這不是跨 entries 原子 transaction。
- 正常在 getDefects 已提交所有使用這些 pointers 的工作後呼叫 helper，檢查回傳值。
  freeCacheData 同步整條 stream；第一階段接受與 legacy event 提早釋放的時間差。
- 後續不得讀取已完成 access 的 pointers，所有保存的 aliases 同時清除。
- abortTransaction 的 move assignment 會同步舊 lease 並 abort，重複呼叫安全；
  它不發布 Fill，也不回傳 CUDA 清理成功結果。錯誤會記錄，原交易必須仍判失敗。
- GPU 工作可能不持有 cache lease，例如僅使用 static data 或私有 buffer 的路徑。
  **abortTransaction 不是通用 GPU drain**。Stop／unload／owner release 仍需檢查
  所有相關 streams 已完成，再銷毀資源。同步失敗時停止後續執行並傳播錯誤，
  不能把 Invalid handles 視為 GPU 已安全完成的證明。
- 在新 lookup 前先清舊 lease。`member = getCacheData(...)` 會先做 lookup，
  再清 destination；不可依賴賦值清理來避免等到自己持有的舊 entry。

必接出口：loadMaster／runRef／getDefects 的每個失敗出口、上層 caller 失敗或
跳過 getDefects、cancel、stop、unload 與 destructor。最好在現有 doExecute 邊界
使用 scope-exit 保護涵蓋例外／early return；它只是本地清理保護，不是 adapter。
成員 accesses 不會在 doExecute 返回時解構，不能只等 task 最終解構。

## 7. Graph owner、初始化與 teardown

1. Process 層先初始化 GpuContextManager；framework 在 task callbacks 前建立
   NUMA affinity。每個 graph NUMA 恰有一顆 GPU，否則 init 報錯。
2. 確認 owner 與 clients 的實際 load／notify 依賴，安排 owner 冷路徑準備與
   task 私有配置。每個 task 的 master／ref fallback 在 load 的初始化路徑預配置。
3. Owner 初始化 StaticData（K>0，gpuCacheWaitTimeout=0ms），一次上傳 DZ 與
   distortion，再 finalize。Ready 才能執行 consumers；Ready 檢查不能取代排序。
   移除 `unique_id==0` 上傳規則，避免每個 task 重複處理共享資料。
4. 每個新 run 在所有舊 CPU／GPU 工作、leases 與新 request 都停住的邊界，由
   owner 呼叫一次 resetCache；不用每個 detector 的 reset counter barrier。
5. 參數變更發生在 cycles 之間仍不代表 GPU 自動完成，先 drain。
   需要 static data 更新時，第一階段用完整 release/init/upload/finalize；這會
   一起清 frame cache。重新解析 pointers／variant 對應後才准許 execute。
6. Stop 等到 task 回呼結束，執行交易清理並 drain streams。
   Unload 明確釋放 task GPU state；報告指出原 deriveUnload 為空，不能忽略。
7. 所有 clients 停止使用 owner 後才 release StaticData，最後 shutdown contexts。
   GpuI2I destructor 必須在銷毀 stream／fallback 之前清理 accesses。

AOI source review 指出 `WinFuncD2DPProcessor::deriveStop()` 似乎呼叫 Start；
請查實際程式並修正，不能在 stop 錯誤觸發新的 reset。

容量以 legacy MiB × 1024 × 1024 除以完整 frame bytes 得到 K，做 overflow 檢查。
Legacy 16 個 static slots 不包含在 cache MiB；新配置分別指定 DZ／distortion bytes。
報告 frame-cache bytes 與總 VRAM 時，須另外計入 task-private buffers／scratch。
不減少現有演算法必需 buffers；「master＋一個 ref fallback」不是總配置只有兩塊。

## 8. Build 與驗收

- C++17 host caller 連結 `gpuinfra`，正式整合可用 `GPUINFRA_BUILD_DEMO=OFF`。
  原 CUDA 10.1 .cu caller 的語言限制與 ABI 要在實際 AOI 工具鏈確認；
  不把本機 CUDA 13.1 通過視為正式環境已驗證。見 [工具鏈文件](../toolchain_compatibility.md)。
- Backend 在初始化／重新初始化選定，整筆交易不可混用 legacy get/free 與新 leases。
- 驗證 K>0 下 hit、fill、eviction、同 key Loading／full fallback、最大 refs、重複 refs、
  master-as-reference、neighbors-diff、mask-out、ROI offset 與資料輸出。
- 同 frame/camera 不同 variant 必須返回各自正確 bytes；hash 碰撞也不可誤命中。
- 故障驗證涵蓋每個 early return、CUDA 錯誤、上層跳過 getDefects、重複 cleanup、
  cancel、stop/unload、live lease reset/release 拒絕與 static-only stream drain。
- 靜態更新後舊 pointer 不得再使用；變更 frame layout 需重建相關配置。
- 同 workload、cache bytes、debug/modality、concurrency 與 preprocessing 做 A/B。
  保留既有 defect-list 比較流程；數值容許誤差需有產品定義，不能為了通過測試自訂。
- 統計以同 execution cycle 標記／輸出，policy bypass 在 caller 記錄；補 full-frame／
  ROI H2D bytes、completion CPU 時間、延遲、throughput、cold-init 與總 VRAM。

## 9. 請整合者回報

- 實際修改檔案、owner multiplicity／affinity 證據與 callback 呼叫順序。
- GpuI2IInputs 是否存在 POD／複製假設，以及改採何種成員儲存位置。
- 每個交易失敗出口的清理位置、static-only GPU 工作的 drain 位置。
- variantId 對應產生方式、generation 更新、reset 與 ID 重用規則。
- 正式工具鏈 build/link 結果、correctness 與 A/B 結果、尚未驗證的分支。

相關 API：[GpuDataAccess](../../src/DataCache/GpuDataAccess.h)、
[GpuCacheRequest](../../src/DataCache/GpuCacheRequest.h)、
[StaticData](../../src/StaticData/StaticData.h)、[variant key](../../src/DataCache/GpuDataKey.h)。
