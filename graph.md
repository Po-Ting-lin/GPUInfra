# Graph Protocol

## Concepts

- A **frame** is one unit of data that moves through the graph.
- A **task type** is one processing stage, such as `TaskA`.
- A **task instance** is one reusable object of a task type. A task instance can
  process only one frame at a time.
- A **graph thread** is a worker thread that runs a task instance. A graph
  thread can run only one `execute()` call at a time.

A frame is not permanently bound to a task instance or graph thread. For each
`execute()` call, the graph scheduler may select any free instance of the
required task type and any free graph thread. Both remain busy until that call
finishes.

## Task API and lifecycle

Each task class provides the following API:

1. `load()`: Allocate the resources required by the task instance.
2. `registerParameters()`: Register the task's parameters.
3. `notifyParameters()`: Apply the parameter values for the current run.
4. `execute()`: Process one frame at this graph stage.
5. `unload()`: Release the resources owned by the task instance.

The lifecycle is:

```text
load -> registerParameters -> notifyParameters -> execute (for each frame) -> unload
```

`load()` and `registerParameters()` are initialization operations.
`notifyParameters()` runs once for every task instance before frame processing
begins for the current run. `execute()` is the per-frame operation, and
`unload()` is the shutdown operation.

---

## One-task graph

Consider this graph:

```text
start -> TaskA -> end
```

Suppose the graph has six graph threads and must process 200 frames. The graph
creates six instances of `TaskA`.

### Cold path (initialization order)

1. Call `TaskA::load()` once on each of the six instances.
2. Call `TaskA::registerParameters()` once on each of the six instances.

### Hot path (run order)

1. Call `TaskA::notifyParameters()` once on each of the six instances.
2. Call `TaskA::execute()` 200 times in total: once for each frame. Up to six
   calls can run concurrently because six `TaskA` instances and six graph
   threads are available.

### End cold path (shutdown order)

1. Call `TaskA::unload()` once on each of the six instances.

### The `execute()` model for a one-task graph

The graph can dispatch a frame to `TaskA` when:

1. The frame is ready.
2. A `TaskA` instance is free.
3. A graph thread is free.

The frame is complete when its `TaskA::execute()` call finishes.

For example, suppose there are:

- Two `TaskA` instances
- Three frames
- Four graph threads
- A five-second execution time for every `TaskA::execute()` call

The following timestamps are the start times of the calls:

```text
00:00 TaskA Instance0 processes Frame0 on Thread0 (finishes at 00:05)
00:01 TaskA Instance1 processes Frame1 on Thread1 (finishes at 00:06)
00:05 TaskA Instance0 processes Frame2 on Thread2 (finishes at 00:10)
```

Although four threads are available, only two frames can run concurrently
because there are only two `TaskA` instances. `Frame2` waits until
`TaskA Instance0` becomes free at `00:05`. The scheduler then runs that
instance on `Thread2`, demonstrating that a task instance is not permanently
bound to one graph thread.

---

## Multi-task graph

Consider this graph:

```text
start -> TaskA -> TaskB -> TaskC -> end
```

Suppose the graph has six graph threads and must process 200 frames. The graph
creates six instances of each task type: six `TaskA` instances, six `TaskB`
instances, and six `TaskC` instances.

### Cold path (initialization order)

1. Call `TaskA::load()` once on each of its six instances.
2. Call `TaskB::load()` once on each of its six instances.
3. Call `TaskC::load()` once on each of its six instances.
4. Call `TaskA::registerParameters()` once on each of its six instances.
5. Call `TaskB::registerParameters()` once on each of its six instances.
6. Call `TaskC::registerParameters()` once on each of its six instances.

### Hot path (run order)

1. Call `TaskA::notifyParameters()` once on each of its six instances.
2. Call `TaskB::notifyParameters()` once on each of its six instances.
3. Call `TaskC::notifyParameters()` once on each of its six instances.
4. For 200 frames, call each task's `execute()` method 200 times in total.
   Calls from different frames may overlap, but each individual frame must be
   processed in the order `TaskA -> TaskB -> TaskC`.

This produces 600 `execute()` calls: 200 for `TaskA`, 200 for `TaskB`, and 200
for `TaskC`.

### End cold path (shutdown order)

1. Call `TaskA::unload()` once on each of its six instances.
2. Call `TaskB::unload()` once on each of its six instances.
3. Call `TaskC::unload()` once on each of its six instances.

### The `execute()` model for a multi-task graph

The graph can dispatch a frame to its next task when:

1. The frame has completed its previous task, if any.
2. An instance of the required task type is free.
3. A graph thread is free.

Tasks for the same frame always run in graph order. Tasks belonging to
different frames may be interleaved and executed in parallel.

For example, suppose there are:

- Two instances of each task type
- Three frames
- Four graph threads
- A five-second execution time for every `execute()` call

The following timestamps are the start times of the calls:

```text
00:00 TaskA Instance0 processes Frame0 on Thread0 (finishes at 00:05)
00:01 TaskA Instance1 processes Frame1 on Thread1 (finishes at 00:06)
00:05 TaskB Instance0 processes Frame0 on Thread2 (finishes at 00:10)
00:05 TaskA Instance0 processes Frame2 on Thread0 (finishes at 00:10)
00:06 TaskB Instance1 processes Frame1 on Thread3 (finishes at 00:11)
00:10 TaskC Instance1 processes Frame0 on Thread0 (finishes at 00:15)
00:10 TaskB Instance0 processes Frame2 on Thread1 (finishes at 00:15)
00:11 TaskC Instance0 processes Frame1 on Thread2 (finishes at 00:16)
00:15 TaskC Instance1 processes Frame2 on Thread3 (finishes at 00:20)
```

The instance number is local to its task type. For example, `TaskA Instance0`
and `TaskB Instance0` are different objects.

The timeline illustrates the following rules:

- Each frame preserves its required task order. For example, `Frame0` runs
  `TaskA`, then `TaskB`, and then `TaskC`.
- Different frames can be at different graph stages at the same time. At
  `00:05`, `Frame0` runs `TaskB` while `Frame2` runs `TaskA`.
- A frame can use a different task instance and graph thread at each stage.
  `Frame0`, for example, runs on `TaskA Instance0` and `Thread0`, then
  `TaskB Instance0` and `Thread2`, and finally `TaskC Instance1` and `Thread0`.
- A task instance can process different frames over time. For example,
  `TaskB Instance0` processes `Frame0` and later processes `Frame2`.

Therefore, frames, task instances, and graph threads are scheduled
independently. They are associated only for the duration of one `execute()`
call.
