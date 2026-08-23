# Open Issues

## StaticData 與真實 graph 的整合點

最後一個必要條件是：若真實 graph 完全不能修改，也沒有既有的
static-data hook、task factory injection 或 terminal callback，這個方案便
無法安全接入。示範版 `DummyGraph` 應只加入必要的生命週期呼叫，以模擬
真實 framework 已提供的擴充點，而不改變 [`graph.md`](graph.md) 定義的
scheduler 選擇邏輯。

實際導入前必須確認真實 framework 能提供：

- graph-copy scope 的 `StaticData` 建立與銷毀時機；
- 每次 task execute 時傳入同一個 `StaticData&` 的能力；
- frame 完成、失敗或取消後的 terminal notification；
- 所有 workers 停止後、GPU contexts 釋放前的 teardown hook。

若缺少其中任何一項，必須先定義 adapter 或使用 framework 既有的
extension point；不可用 process-global mutable singleton 取代。

## FrameSlot recycling 與 hash index 更新

目前 `StaticData` 在 `init()` 建立 `frame ID -> FrameSlot index`，workers 啟動
後只讀取該 index，並在所有 workers 停止後才由 `release()` 清除。這符合目前
每個 logical frame 固定綁定一個 slot、graph lifetime 內不 recycling 的範圍。

未來若改成少量 slot 重複綁定新 frame，不能在 worker 查詢同一個
`std::unordered_map` 時直接 insert、erase 或 rehash。屆時必須定義：

- bind/unbind 與 task lookup 的同步邊界；
- frame ID 重用時的 generation/epoch，避免舊 atom 命中新 payload；
- index 更新失敗時，slot metadata、state 與 index 的一致性回復方式；
- terminal callback 之後，何時才可安全移除舊 frame ID。

在這些條件確定前，hash index 必須維持 graph-lifetime immutable。
