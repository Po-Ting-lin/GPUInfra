# Recorded results

- `benchmarks/`: timing CSVs and plots from model/factor comparisons.
- `profiling/`: NVTX capture summaries, logs, CSVs and retained reference traces.

Run tools from the repository root; see [profiling instructions](../docs/profiling.md).
Benchmark output defaults to `results/benchmarks/`.
Existing artifacts are preserved verbatim. Commands and source paths recorded
inside historical logs describe the original capture, before directory moves.
New raw profiler traces and SQLite exports are ignored by default; summaries,
logs and CSVs remain reviewable. The existing reference capture is retained.
