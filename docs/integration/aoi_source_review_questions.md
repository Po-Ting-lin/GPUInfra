# AOI_AOS 整合 GPUInfra：原始碼確認問題

日期：2026-09-23

## 給能查看 AOI_AOS／Grape 完整原始碼的 agent

請依完整原始碼回答下面問題，目的是決定最小的整合 adapter 與必要的
GPUInfra 修改。請勿只依先前 OCR 報告推論，也先不要修改 AOI 程式碼。
優先回答第 1 節（預處理與 cache identity），再確認 owner 與生命週期。

每題請提供：

- 結論：已確認／條件成立時才成立／尚無法確認。
- 證據：repository、commit 或版本、檔案路徑、類別／函式、行號與必要短片段。
- 呼叫順序及成立條件，包含影響行為的 compile flag／runtime option。
- 對整合的影響、最小修改位置；若缺資料，明確指出缺哪個定義或呼叫端。

請區分「原始碼已保證」與「建議新增的保證」。OCR 中的拼字／符號錯誤，
請以實際原始碼校正。不要把另一份報告的描述當成獨立驗證。

## 已確定的決策與目前實作

以下是既定方向，請找出實際接線位置；若原始碼有衝突，提供反例，不需重新詢問偏好。

- 每個 NUMA 一個 graph copy。Framework 在呼叫 load 與其餘 task callbacks 前
  建立該 graph 的 NUMA affinity；task 由當前 NUMA 查詢 GPU。
- 每個 graph NUMA 必須恰有一顆 GPU；零顆／多顆在初始化時報錯，不任選 GPU。
- 每個 graph copy 一個共用資源 owner；不以新的 process-global singleton 取代舊 cache。
- 保留 AOI scheduler 與 loadMaster → runRef × N → getDefects 的演算法流程。
- Caller 在 load 預配置 master＋一個 ref fallback buffer；額外 buffer 只在實際
  delayed-read 路徑需要時增加。同 stream 前一輪讀取必須排在下一輪覆寫之前。
- CacheFill 填完整可共享 payload；不在上傳完成後提早發布。
  freeCacheData() 同步 request stream 後發布／回滾並結束 lease。
- GpuDataAccess 已支援 move construction／assignment，仍不可 copy；可存入
  caller 成員或預配置陣列。Move 後來源 Invalid；覆寫 active destination 會同步
  並 abort 舊 lease。正常流程仍須明確完成並檢查結果。
- DZ／distortion 各自配置，容量以 bytes 指定，由 StaticData 持有；在 init／notify
  冷路徑上傳並 finalize，execute 前必須完成。之後唯讀，更換需停工並重新初始化。
- 本次先跳過 K=0 的整合與測試；從 K>0、waitTimeout=0 開始，不改通用等待預設值。
- 非零 cache、必要 fallback 或 static data 配置失敗，都在 init／load 報錯；
  不自動降級成 K=0。
- Batch／event completion、slab allocation、variable-size result memory pool
  先留 backlog；不能只憑推測列為第一階段必要功能。
- PcrResultsCache 第一階段不遷移；未來仍有 GPU 演算法結果 cache 的需求。
- 工具鏈已加入 cuFFT enum header probes 與 GPUINFRA_BUILD_DEMO=OFF；
  舊 CUDA／AOI compiler 的完整編譯與執行仍待正式環境驗證。

## 1. 預處理與 cache identity（優先）

### Q1：Cache entry 實際保存什麼資料？

請列出所有 getCacheFrame 後 load_flag=true 的填入路徑，包括 master、reference、
smoothed buffer、undistortFrame，以及條件編譯分支。各路徑保存的是原始 frame、
去畸變結果、平滑結果，還是其他中間結果？不同路徑是否共用同一個 legacy cache？

### Q2：哪些輸入會改變 cached bytes？

請列出真正影響填入內容的參數，而非列出所有演算法參數。至少檢查：

| 候選因素 | 是否影響 cached bytes | 來源／版本識別 | 可變更時機 | 原始碼證據 |
| --- | --- | --- | --- | --- |
| 原始 frame 內容與 camera | 待確認 | | | |
| Distortion table 內容／版本／index | 待確認 | | | |
| Rotation／orientation | 待確認 | | | |
| Smoothing／preprocessing mode | 待確認 | | | |
| ROI、座標系、尺寸、dtype、pitch／stride | 待確認 | | | |
| Master／reference 角色與演算法版本 | 待確認 | | | |
| DZ／mask／其他參數 | 待確認 | | | |

請區分只影響後續計算結果的參數，與真的改變 cache payload 的參數。

### Q3：同一執行期間，同 key 能否代表不同內容？

兩個 task 或同一 task 的兩次呼叫，在同一段不 reset 的期間，是否可能以相同
(frameId, cameraId) 要求不同的預處理結果？請給出實際路徑或排除它的程式保證。
若會發生，衝突是否來自角色、distortion index、mode，或不同 graph 共用 cache？

### Q4：ID 的唯一範圍與重用時機是什麼？

Frame／camera ID 的型別、範圍與產生位置為何？何時跨 run、產品、recipe、批次
或影像重載而重用？相同 ID 的 CPU frame bytes 是否可能被覆寫？是否有其他
影像來源共用同一編號空間？請指出能安全清除舊 identity 的邊界。

### Q5：只用 reset 是否足夠？

針對 Q1–Q4，請選擇有證據支持的最小方案：

- 所有影響 payload 的變更都在 quiescent boundary：沿用現有 key，在該處 reset。
- 同期存在多種 payload：增加 caller 定義的 preprocessId／版本識別，或分開 manager。

若需 preprocessId，請列出完整組成、產生／更新位置及唯一性保證。不要假設
未檢查碰撞的 hash 一定唯一，也不要假設全域 parameter revision 能涵蓋每個
request 的 distortion index／mode。若建議分開 manager，請說明其 lifetime 與容量。

## 2. AOI owner 與 framework 生命週期

### Q6：哪個真實物件可以持有一份 graph-copy StaticData？

提供具體類別、建構／銷毀位置、graph clone/copy 行為與 multiplicity。
可用的 DB/resource-task／framework extension point 是什麼？如何確保不同
NUMA 的 graph 不共用同一份 mutable owner？Tasks 如何取得該 owner 的 reference？

### Q7：請把生命週期對應到真實函式。

| GPUInfra 動作／保證 | AOI／Grape 函式與位置 | 執行緒／NUMA | 次數與順序 |
| --- | --- | --- | --- |
| Process-wide GpuContextManager 初始化 | | | |
| Graph-copy StaticData 初始化 | | | |
| Task load 與 fallback 配置 | | | |
| 初始參數與 DZ／distortion 上傳、finalize | | | |
| Execute 使用共用資源 | | | |
| 新 run 前一次 resetCache | | | |
| 參數變更：停工、drain、重新初始化／reset | | | |
| Stop／cancel／錯誤後 drain | | | |
| Lease 清理、task unload、graph owner release | | | |
| 最後釋放 GPU contexts | | | |

請明確指出 task load 與 static owner 初始化之間是否有依賴，避免套用不符合
AOI 的固定順序。NUMA affinity 請提供 framework 實際保證的位置。

### Q8：哪個 barrier 真的涵蓋 GPU 工作？

Framework 等到 execute 返回，是否也代表 GPU 工作完成？有無其他 streams、
背景工作或延後 callback 仍讀取 cache／static pointers？哪個位置能保證
所有 lease 已結束、新 request 尚未開始，且 static readers 也已 drain？

### Q9：參數變更會如何通知多個 instances？

Notify 的順序、並行性、初始呼叫時機、失敗傳播方式為何？如何讓 graph owner
只上傳一次，所有 tasks 都看到 Ready，且不沿用「unique ID 0 負責上傳」？
若 notify 後無法完成重新初始化，哪個機制阻止下一次 execute？

## 3. Caller access 與 buffer 使用期間

### Q10：每筆交易的所有權與並行性為何？

loadMaster／runRef／getDefects 是否一定屬於同一個 instance？是否可能重入、
交錯處理兩筆交易、切換 host thread，或缺少 getDefects 而提早離開？請指出
保存 master/ref GpuDataAccess 的成員位置，以及清除前次交易的入口。

### Q11：Master 與 reference 最後一次被 GPU 讀取在哪裡？

請逐一路徑追蹤 cached_master／cached_ref 的讀取，包含
GPU_OPTIMIZE_MASTER_AS_REF、m_bNeighborsDiff、mask-out-by-ref、getDefects
及任何 debug／optional branch。原有 event 的 record、wait 與最後 stream sync
各在哪一行？提供每條特殊路徑的最後 consumer 與可釋放點。

### Q12：Master＋一個 ref fallback 是否足夠？

請核對下一次 ref H2D 後是否還會讀舊 ref buffer、smoothed buffer 或其 alias。
同 stream 順序是否涵蓋所有 consumers？若有多 stream，現有 event dependency
為何？若某分支需額外 buffer，精確列出分支與原因，不推論需要 64 個 fallback。

### Q13：同交易會不會再次要求仍在填入的同 key？

Legacy 允許相同 loading_unique_id 在 Loading 時取得同一份資料；GPUInfra
目前不提供這種例外。請查 master-as-ref、重複 ref、重複 frame/camera 的實例。
使用 waitTimeout=0 後回到 caller fallback，是否仍能保持結果一致？有無依賴
指標相同、原地更新或尚未可重建內容的行為？本階段不新增提早發布。

### Q14：ROI fallback 與完整 cache payload 如何共用 caller？

請列出 full-frame 與 ROI 的 allocation bytes、copy 範圍、pitch、pointer offset、
kernel addressing。Fallback allocation 是否至少有完整 payload 容量？
只填 ROI 時，所有 consumers 是否確實只讀已填區域？CacheHit 取得完整 frame
時，是否需要不同 offset？Legacy buffer 能否直接用，或需 adapter 修正？

### Q15：所有失敗出口如何完成 cleanup？

列出 loadMaster、runRef、getDefects 的 early return、例外、CUDA 失敗、cancel、
最大 reference 數量超限路徑。每個出口如何使 saved accesses 在 streams／buffers／
manager 銷毀前完成或 abort？原本的錯誤是否會傳回 graph 並阻止後續工作？

## 4. Static data 與容量配置

### Q16：DZ／distortion 的真實 layout 與大小如何計算？

請列出各區域的 bytes 上限、offset、alignment、table index、packing、來源 lifetime
與必要資料完整性條件。Legacy 的 16 個 frame-sized slots 各存什麼？是否還有
DZ／distortion 以外的資料需要保留？誰能提供新配置與上傳清單？

### Q17：哪些更新需要同時清除 frame cache？

列出變更事件及其影響：僅更新 static data、僅 reset frame cache，或兩者都要。
哪些更新改變 frame layout／entry bytes，需要重新配置 cache？哪些位置保存舊
static pointer，重新初始化後如何更新，避免繼續使用已釋放的 pointer？

### Q18：Cache budget 與必要記憶體如何分帳？

Configured MB 是十進位 MB 還是 MiB？它只含 frame cache，還是包含 static slots？
每 GPU／每 NUMA 的有效配置值在哪裡決定？請列出 fallback、scratch、演算法
buffers、static data 的額外用量，以及 checked arithmetic 與 init 失敗傳播位置。
A/B 請同時列 frame-cache payload bytes 與總 VRAM，不把不同 budget 當等量比較。

## 5. Build、觀測與驗收

### Q19：正式支援的工具鏈矩陣是什麼？

請區分仍在使用與已淘汰的 CUDA 10.1／11.0／11.3 設定，提供 OS、compiler、
CMake／原生 build system、C++ dialect、cuFFT、GPU architecture、ABI 與 link flags。
哪些呼叫 cache 的函式在 .cpp，哪些在 .cu？舊 nvcc 是否可以透過 C++17 host
adapter 隔離？如可執行驗證，回報實際 configure／build／link 結果與錯誤原文；
不要將 enum probes 修正視為完整舊工具鏈驗收。

### Q20：Modality 哪些仍在正式使用？

請列實際設定值、排除規則與使用理由。若放在 caller bypass，在哪裡記錄
policy bypass 與所有 requests 的分母，以便與 legacy cache_skipped 對照？
確認同次 A/B 的 policy 一致，不把 policy 變更與 replacement policy 混在一起。

### Q21：A/B 的最小切換點與輸出比較方式是什麼？

在哪個初始化位置選擇 legacy／GPUInfra，能避免同一交易混用兩套 cache？
切換是否要求重新初始化？有哪些可重播資料可涵蓋 hit、fill、eviction、loading/full
fallback、重複 IDs、預處理變更與特殊 master/ref 分支？
演算法輸出預期 bitwise 一致，還是已有明確數值容許誤差？由哪個現有驗證工具比較？
本階段只規劃 K>0、waitTimeout=0，不要求 K=0 測試。

### Q22：效能資料由誰輸出，量測區間怎麼對齊？

請指出可收集 throughput、延遲分位數、實際 full-frame／ROI H2D bytes、cache
outcomes、policy bypass、completion CPU 時間、cold-init 時間及總 VRAM 的位置。
Legacy CacheStatistic 的 reset／輸出區間如何對齊新統計？是否有代表性 workload
能量測命中與競爭，而非全 unique frames？先提供現有量測方式與缺口，不因尚無
量測就要求 batch completion／memory pool。

## 期望交付

請提供以下四份結果，可合併在一份 Markdown：

1. Q1–Q5 的 identity 結論：reset 足夠，或必須區分同時存在的 payload variants；
   附具體 key／reset 方案及原始碼依據。
2. 完成 Q7 的 owner／callback 對照表與成功、參數變更、取消／錯誤三種呼叫順序。
3. Caller 最小修改清單：檔案／類別／函式、saved lease 成員、fallback 配置、
   cache status 分支、completion／abort 位置；保留原本 scheduler。
4. 分類清單：整合前必修、caller adapter 工作、待量測 backlog、仍缺的證據。

每題保留 Q 編號方便逐項討論。未確認項目請明列，不以一般設計原則代替原始碼證據。

## GPUInfra 參考

- [Graph golden rules](../graph.md)
- [目前架構](../architecture.md)
- [Legacy 對照與既定整合決策](old_vs_new.md)
- [Static GPU data legacy usage](legacy_constant_gpu_memory_usage.md)
- [工具鏈相容性與驗證限制](../toolchain_compatibility.md)
- [統計定義](../cache_statistics.md)
- [Backlog](../open_issues.md)
- [GPUInfra 使用與 move lease 說明](../../README.md)
