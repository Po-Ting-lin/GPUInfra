以下是 AOI_AOS 目前 constant-frame memory 的用法（來源： .cpp 、 GpuI2I.cpp）：

# 配置（GpuCache::register_thread）

• 與 frame cache 同一次 cudaMalloc : cache_size + frame_size * GPU_CACHE_NUM_CONST_FRAMES (16個 frame) ，const_frames_mem = mem + cache_size ，緊接在 cache 區之後。
• 只有第一個註冊的 instance（unique_id == 0）配置並以 cudaMemsetAsync +
cudaStreamSynchronize 清零。
• 一個 process 内所有 GpuI2I instance 共用同一份 (file-scope global gpuCache)。

# 取用 API

• getConstantFrameMemory(&size_in_frames) : 回傅基底指標與可用 frame 數 (16)。
• loadConstantFrameMemory(thread_id, src, size, offset) : 只有 thread_id == 0 會真的 H2D (cudaMemcpyAsync 到 stream 0 + cudaStreamSynchronize)，其他 thread 直接回 true；只做 offset + size 邊界檢查。

# 空間切分（GpuI2I::init，固定 offset）

- **frame 0-14：detection zones（GPU_MAX_DETECTION_ZONES_SIZE_IN_FRAMES_CACHE = 15）。**
- **frame 15：distortion tables（GPU_MAX_DISTORTION_SIZE_IN_FRAMES = 1），offset 寫死為「DZ之後」。**
- **兩者各有 fallback：若 cacheSizeInMB == 0、const mem 為 null、或空間不足，就改為 instance 私有 cudaMalloc（DZ 的私有大小 GPU_MAX_DETECTION_ZONES_SIZE_IN_FRAMES_LOCAL = 0，即實際上 DZ 只能靠 cache 常數區；distortion 私有配置 GPU_MAX_DISTORTION_IMAGES = 12 張表）。以 m_bConstantThDetectionZoneInCache / m_bConstantDistorionTablesInCache 記錄來源。**

# Detection zone 寫入（loadDetectionZones）

- **每個 DZ 先在 CPU 對齊 64 bytes 到 pinned 暂存區（重用 m_pCPUbuffers），再 loadConstantFrameMemory(m_UniqueId, ..., m_nDetectionZonesAllocationOffset）。**
- **最多 GPU_MAX_DETECTION_ZONES_PER_FRAME = 5 個，index 必須連續，累計 offset 不得超過 15 個 frame；unloadDetectionZones() 只把 offset 歸零，不清 GPU 記憶體。**
- **執行期 kernel（第 1429、1844 行）以 m_pGPUbufConstantDetectionZone + offset_in_buffer 讀取。**

# Distortion table 寫入 (loadDistortionTable)

• CPU 端把 float 位移表下採樣 8 倍、轉成 interleaved short 定點 (m_nDistortionStride × m_nDistortionHeight 個 unsigned int)，再 loadConstantFrameMemory 到 frame 15 基底 + index × 表大小。

• 執行期 (第 1598 行) 以 m_pGPUbufDistortionTables + m_nCurrentDistortionIndex × 表大小 餵給 undistort kernel。

• 註解有 TODO - move to const mem，表示原作者也認為現況是暂時做法。

# 生命週期

• 不參與 LRU、不受 resetCache() 影響 (resetCache 只 memset cache_lines)。

• close() 時隨整塊 mem 一起 cudaFree；私有 fallback 版本則各自 cudaFree (第 1973、1988 行)。

# 對 GPUInfra 的含意

• 需要一個 graph-copy scope、非 best-effort、跨 run 保留的靜態區，有兩種型別 (DZ、distortion) 與各自的 offset/容量驗證。

• 現行「只有 thread 0 上傳、其他 instance 假設已就位」的隱含契約，以及 DZ 空間為 0 時無法運作的 fallback，是整合時要明確化的地方。
