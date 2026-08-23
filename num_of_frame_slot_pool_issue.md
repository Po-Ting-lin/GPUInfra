# Number of FrameSlotPool Entries — Open Issue

## 問題摘要

目前 `StaticData::FRAME_SLOT_POOL_SIZE` 固定為 220，但 220 不是由 task
instance 數量或 GPU concurrency 推導出的架構常數。它來自目前 demo 的預設
workload：

```text
20 warmup frames + 200 timed frames = 220 configured frames
```

目前實作又採用一個 configured logical frame 固定綁定一個 `FrameSlot`，而且
graph lifetime 內不 recycling，因此需要 220 個 slots 才能執行預設 workload。

這是目前實作的保守容量，不代表所有 graph 都需要 220 個同時存活的 GPU
frame payloads。

## 目前程式行為

[`main.cpp`](src/main.cpp) 預設每張 GPU 建立：

- 20 個 warmup frames；
- 200 個 timed frames；
- 4 個 task instances；
- 4 個 graph workers。

[`DummyGraph::initializeOnNumaNode()`](src/DummyGraph.cpp) 會在 workers 啟動前
建立所有 `FrameCpuAtom`，並為每個 atom 建立一筆 `StaticFrameConfig`。
[`StaticData::init()`](src/StaticData.cpp) 接著：

1. 建立固定 220 個 `FrameSlot` objects；
2. 為每個 configured frame 綁定一個 slot；
3. 為每個 bound slot 配置 persistent GPU frame buffer；
4. 建立 immutable `frame ID -> slot index`；
5. 保留 slot、GPU allocation、state 和 result 到 graph teardown。

目前 `FrameSlot::bind()` 不支援將 terminal slot 重新綁定至另一個 logical
frame。因此預設 mapping 是：

```text
Warmup Frame 0  ... 19   -> Slot 0  ... 19
Timed  Frame 20 ... 219  -> Slot 20 ... 219
```

即使 warmup phase 已結束，前 20 個 slots 仍然保持 bound 和 allocated，不會
提供給 timed frames 使用。

## Task instance 數量不等於 FrameSlot 數量

4 個 `TaskA` instances 只限制同一時間最多有 4 個 `TaskA::execute()` calls，
不限制 `TaskA` 在整個 run 中能處理多少 frames。Task instance 是 reusable
execution resource：

```text
TaskA instance 0: F0, F4, F8,  ... F216
TaskA instance 1: F1, F5, F9,  ... F217
TaskA instance 2: F2, F6, F10, ... F218
TaskA instance 3: F3, F7, F11, ... F219
```

每個 instance 依序執行 55 次，4 個 instances 總共仍可處理 220 frames。

`TaskGpuResources` 中的 stream、scratch、pinned staging 和 task-private
buffers 可以在 `execute()` 結束後供下一個 frame 重用。但 frame payload 若
尚未完成所有 graph stages，必須繼續存放於該 frame 的 `FrameSlot`：

```text
TaskA0 executes F0
  -> F0 GPU data 留在 FrameSlot(F0)，等待 TaskB
  -> TaskA0 free，可以開始執行 F4
```

因此 task instance 數量只代表 stage execution concurrency，不代表同時存活
的 frame payload 數量。

## 可能需要 220 個 live slots 的例子

考慮：

```text
TaskA -> TaskB -> TaskC
```

假設：

- 有 220 個 timed frames；
- TaskA 和 TaskB 各有 4 個 instances；
- 至少有 8 個可執行這些 tasks 的 graph workers；
- TaskA 每批 4 frames 花費 1 ms；
- TaskB 處理一批 frames 花費 100 ms；
- scheduler 沒有 input admission limit 或 bounded pipeline depth。

可能出現：

```text
t=0~1 ms
  A0~A3 process F0~F3

t=1~101 ms
  B0~B3 process F0~F3

t=1~2 ms
  A0~A3 process F4~F7

t=2~3 ms
  A0~A3 process F8~F11

...

t=54~55 ms
  A0~A3 process F216~F219
```

在 `t=55 ms`：

```text
F0~F3       正在 TaskB
F4~F219     已完成 TaskA，等待 TaskB
```

此時可能有：

```text
4 executing in TaskB + 216 waiting for TaskB = 220 live frame payloads
```

每個 payload 都不能被下一個 frame 覆蓋，因為 TaskB 或 TaskC 尚未完成讀取。
在這個沒有 admission control 的例子中，即使每種 task 只有 4 個 instances，
仍可能需要 220 個 slots。

## 何時不需要 220

上述 220-live-frame 情境不是 task instance 數量必然造成的結果，還需要足夠
workers、允許 TaskA 持續執行，以及沒有 bounded admission/backpressure。

如果 framework 保證：

- 同時最多只有 `K` 個 frames 進入 graph；
- downstream stages 具有足夠優先權，不會無限制累積 upstream output；
- terminal result 已被 consumer 複製或確認不再使用；
- slot 可以在 terminal callback 後安全 rebind；

則 pool 只需要涵蓋最大 live-frame 數：

```text
required FrameSlotPool size >= maximum simultaneous live frames
```

對目前 checked-in 的單一 stage graph：

```text
start -> DummyTask -> end
```

在 4 個 tasks、4 個 workers，而且 frame terminal 後立即 recycling 的前提
下，約 4 個 slots 可能已經足夠。至少在 phase boundary recycling，20 個
warmup slots 也可以提供給 timed phase 重用，使需求由：

```text
20 + 200 = 220
```

降低為：

```text
max(20, 200) = 200
```

目前尚未實作上述 recycling。

## 記憶體成本範例

預設 `size_factor=128` 時，input frame 是 `1024 x 1024` 的 `uint8` data：

```text
每個 GPU frame buffer = 1 MiB
220 個 bound slots     = 220 MiB GPU memory / graph copy / replica
```

此外，每個 configured frame 還有約 1 MiB 的 `FrameCpuAtom` input，以及
預先配置的 CEL、SDD、MI result vectors。固定使用總 frame 數作為 pool size
會以較高的 host/GPU memory 使用量換取：

- stable slot 和 device pointers；
- execution hot path 無配置；
- 不需要 slot acquisition failure 或 retry protocol；
- 不依賴 scheduler fairness、queue order 或 pipeline depth。

## 為什麼目前採用 total configured frames

[`graph.md`](graph.md) 定義 task instance 和 graph thread 的選擇方式，但沒有
定義：

- graph 同時最多 admission 多少 frames；
- stage queue 的最大深度；
- upstream/downstream scheduling priority；
- slot 不足時的 wait、retry 或 requeue protocol；
- terminal result consumer 何時不再引用 slot data。

又因為 scheduler 選擇邏輯不能修改，使用一個 slot 對應一個 configured
frame 是目前唯一不需要額外 backpressure contract 的保守方案。Hash index
只把 selection lookup 從 O(number of bound slots) 降為平均 O(1)，不會減少
live payload 數量，也不會自動提供 recycling safety。

## 待確認事項

決定真正的 pool size 前，需要從真實 framework 取得以下保證：

1. 每個 graph copy 的最大 simultaneously admitted frame 數；
2. 每個 stage 的最大 queue depth；
3. graph workers 與各 task type instance 數量；
4. scheduler 是否保證 downstream priority 或 bounded backlog；
5. frame 到達 terminal stage 後，GPU payload 何時可被覆寫；
6. result delivery 是否複製資料，或 consumer 是否仍保存 slot reference；
7. slot 不足時是否存在不修改 scheduler selection 的安全等待或重試 hook；
8. frame ID 是否可能重用，以及需要何種 generation/epoch。

## 目前結論

- `220` 是目前預設 workload 加上一-frame-per-slot、不 recycling 政策的結果。
- `220` 不是 task instance 數量推導出的必要常數。
- task instance 數量只限制 concurrent `execute()` calls。
- 真正的 pool size 應由最大 simultaneous live-frame 數決定。
- 在真實 scheduler 沒有提供 admission/backpressure/lifetime 保證以前，220
  是安全但記憶體成本較高的 conservative upper bound。
- 若要縮小 pool，必須先定義 slot acquisition、terminal recycling、index
  更新與 frame-ID generation contract。
