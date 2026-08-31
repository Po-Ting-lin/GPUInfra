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
- Keep this simulation replaceable by the real graph implementation.

Changes in this directory are limited to faithfully matching the external
framework contract. GPUInfra features belong outside this boundary.
