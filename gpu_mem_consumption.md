`NumGpuCacheEntries = min(ConfiguredGpuCacheEntries, NumConfiguredFrames)`
`TotalTaskGpuResourcesSize = NumTaskInstances × (HostInputBufferSize + DeviceFallbackInputSize + MaxScratchSize)`
`TotalAlgoPrivateSize = NumTaskInstances × Σ(EachAlgoDevicePrivateSize + EachAlgoHostStagingSize)`
