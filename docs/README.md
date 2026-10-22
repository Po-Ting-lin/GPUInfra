# GPUInfra documentation

Documents are written in English unless marked **(中文)**. Open issues, review
notes and the AOI integration notes are kept in Chinese; they are the working
record and are not translated.

Current implementation and integration contracts:

- [Architecture](architecture.md): ownership and lifecycle.
- [Cache and static-data API guide](api_guide.md): lookup statuses, miss handling, static GPU data, waiting and run boundaries.
- [Graph golden rules](graph.md): NUMA affinity and framework callbacks.
- [Toolchain compatibility](toolchain_compatibility.md): core-only builds and validation limits.
- [CUDA logging](cuda_logging.md): checks, caller context and output examples.
- [Cache policies](cache_policies.md): LRU/FIFO eviction and Keep/Discard completion.
- [Result pipeline simulation](result_pipeline.md): A → CPU tasks → B tests and benchmarks.
- [Cache statistics](cache_statistics.md): counters and A/B interpretation.
- [NVTX profiling](profiling.md): build, collection and recorded results.
- [Open issues](open_issues.md) **(中文)**: integration work, future improvements and review follow-ups.

Integration references:

- [GPUInfra code and documentation review (2026-10-22)](integration/gpuinfra_review_20261022.md) **(中文)**: findings, fixes applied and remaining follow-ups.
- [GPUInfra combined review (2026-09-24)](integration/gpuinfra_review_20260924.md) **(中文)**: findings, AOI integration status and priorities.
- [AOI caller integration instructions](integration/aoi_caller_integration.md) **(中文)**: direct GpuI2I changes, saved leases and transaction cleanup.
- [Payload identity](integration/payload_identity.md): variant keys and AOI mapping contract.
- [AOI source review questions](integration/aoi_source_review_questions.md) **(中文)**: pending integration evidence and caller mapping.
- [Legacy versus current infrastructure](integration/old_vs_new.md).
- [Legacy constant GPU memory usage](integration/legacy_constant_gpu_memory_usage.md) **(中文)**.

Visual guides:

- [Cache workflow — English](guides/gpu_cache_explained_en.html).
- [Cache workflow — 中文](guides/gpu_cache_explained.html).
- [Class diagram](guides/gpuinfra_class_diagram.html).
- [Resource flow](guides/gpuinfra_resource_plot.html).

Design rationale:

- [GPU memory consumption](design/gpu_mem_consumption.md).
- [Cache capacity versus logical frame count](design/num_of_gpu_cache_entry_issue.md) (mixed English/中文).
- [Residency index choices](design/unordered_map_vs_fixed_open_addressing.md).
- [Why retain GPU contexts](design/why_keep_gpu_context.md).

Historical plans are retained in [archive](archive/README.md). They may describe
superseded APIs or unimplemented extensions; use the current contracts above.
The repository [README](../README.md) contains build and execution instructions.
