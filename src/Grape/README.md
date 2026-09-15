# Grape Simulation Boundary

This directory simulates the external graph framework described by
[`graph.md`](../../graph.md). It exists only so GPUInfra can be built and tested
without the real framework.

## Rules

- Do not change graph scheduling, task selection, frame ordering, worker
  assignment, or task-instance exclusivity.
- Do not add GPU/cache policy to this layer.
- Treat `DummyGraph`, `FrameCpuAtom`, and `GraphTypes` as framework-owned code.
- Integrate GPUInfra through the existing lifecycle calls and task APIs.
- `NumaExecutor` establishes CPU affinity before all task callbacks, including
  registration and load, and before worker execution. Production supplies this
  guarantee; keep affinity out of task callbacks and `GpuContextManager`.
- GPU IDs are resolved by GPUInfra from the established NUMA environment.
  `GraphConfig` carries workload settings, not NUMA placement or GPU selection.
  Keep the existing task/frame/worker calculations using the resolved GPU count.
- Simulate one explicit graph execution cycle: initialize/register/load,
  `start()`, zero or more frame phases, `stop()`, then shutdown/unload.
- Treat parameter notification as change-driven and quiescent; a phase boundary
  by itself does not notify tasks.
- Keep this simulation replaceable by the real graph implementation.

Changes in this directory are limited to faithfully matching the external
framework contract. GPUInfra features belong outside this boundary.
