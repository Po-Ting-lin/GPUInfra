`NumGpuCacheEntries = ConfiguredGpuCacheEntries`
`TotalTaskGpuResourcesSize = NumTaskInstances × (HostInputBufferSize + DeviceFallbackInputSize + MaxScratchSize)`
`TotalAlgoPrivateSize = NumTaskInstances × Σ(EachAlgoDevicePrivateSize + EachAlgoHostStagingSize)`
