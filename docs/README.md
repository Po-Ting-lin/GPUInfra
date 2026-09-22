# GPUInfra documentation

Current implementation and integration contracts:

- [Architecture](architecture.md): ownership and lifecycle.
- [Graph golden rules](graph.md): NUMA affinity and framework callbacks.
- [CUDA logging](cuda_logging.md): checks, caller context and output examples.
- [Cache statistics](cache_statistics.md): counters and A/B interpretation.
- [NVTX profiling](profiling.md): build, collection and recorded results.
- [Open issues](open_issues.md): integration work and future improvements.

Integration references:

- [Legacy versus current infrastructure](integration/old_vs_new.md).
- [Legacy constant GPU memory usage](integration/legacy_constant_gpu_memory_usage.md).

Visual guides:

- [Cache workflow — English](guides/gpu_cache_explained_en.html).
- [Cache workflow — Chinese](guides/gpu_cache_explained.html).
- [Class diagram](guides/gpuinfra_class_diagram.html).
- [Resource flow](guides/gpuinfra_resource_plot.html).

Design rationale:

- [GPU memory consumption](design/gpu_mem_consumption.md).
- [Cache capacity versus logical frame count](design/num_of_gpu_cache_entry_issue.md).
- [Residency index choices](design/unordered_map_vs_fixed_open_addressing.md).
- [Why retain GPU contexts](design/why_keep_gpu_context.md).

Historical plans are retained in [archive](archive/README.md). They may describe
superseded APIs or unimplemented extensions; use the current contracts above.
The repository [README](../README.md) contains build and execution instructions.
