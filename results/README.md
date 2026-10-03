# Recorded results

- `benchmarks/`: model/factor timings and cache-policy lookup measurements.
- [Cache policy benchmark, 2026-10-03](benchmarks/20261003-cache-policy/README.md):
  LRU/FIFO with long-held readers, latency and complete-request throughput.
- `profiling/`: NVTX capture summaries, logs and CSVs; raw traces stay local.

Run tools from the repository root; see [profiling instructions](../docs/profiling.md).
Benchmark output defaults to `results/benchmarks/`.
Commands and source paths recorded inside historical logs describe the original
capture. Raw profiler files can embed process environment credentials, so SQLite,
Nsight reports and intermediate captures are never tracked, including reference
captures. Keep local raw files only as needed. Summaries, logs and CSVs should
also be reviewed for sensitive content before committing.

Enable the repository hooks in every clone:

```sh
git config core.hooksPath .githooks
```

The commit hook checks staged content; the push hook also checks outgoing commit
history, so deleting a credential in a later commit does not bypass the check.
The scanner blocks raw artifact extensions and selected credential formats; it
is not a complete secret detector. Keep GitHub push protection enabled.
