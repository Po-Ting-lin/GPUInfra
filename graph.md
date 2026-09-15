# Graph Protocol

## Concepts

- A **frame** is one unit of data that moves through the graph.
- A **task type** is one processing stage, such as `TaskA`.
- A **task instance** is one reusable object of a task type. A task instance can
  process only one frame at a time.
- A **graph thread** is a worker thread that runs a task instance. A graph
  thread can run only one `execute()` call at a time.
- A **graph copy** is the graph assigned to one NUMA node. There is one graph
  copy in each NUMA node.

A frame is not permanently bound to a task instance or graph thread. For each
`execute()` call, the graph scheduler may select any free instance of the
required task type and any eligible free graph thread. Both remain busy until
that call finishes.

All graph threads eligible to run a given task are guaranteed to come from the
same NUMA node. Therefore, "any free graph thread" in the scheduling rules
below means any free thread that is NUMA-local and eligible for that task. The
examples below describe one graph copy and its NUMA-local graph threads.

## Task API and lifecycle

The real framework invokes protected `doXxx()` callbacks. This simulation uses
short public names with the same ordering and meaning:

| Simulation | Real callback | Meaning |
| --- | --- | --- |
| `registerParameters()` | `doRegisterTaskTables()` | Register the parameter schema once, immediately after construction. |
| `load()` | `doLoad()` | Allocate parameter-derived resources after all initial parameter values are defined. |
| `notifyParameters()` | `doNotifyParameters()` | Apply values that actually changed while no `execute()` call is active. |
| `start()` | `doStart()` | Enter the execution cycle. |
| `execute()` | `doExecute()` | Process one frame at this graph stage. |
| `stop()` | `doStop()` | Leave the execution cycle after all calls to `execute()` finish. |
| `unload()` | `doUnload()` | Release resources owned by the task instance. |

### Golden rule: NUMA locality for every task callback

The framework establishes CPU affinity to the graph copy's assigned NUMA node
before invoking any callback in the table above. This guarantee applies to
`registerParameters()`, `load()`, `notifyParameters()`, `start()`, `execute()`,
`stop()`, and `unload()`, including their corresponding production callbacks.
It holds for the entire duration of each callback, not just at entry.

Callbacks may run on different threads or CPUs, but all of those CPUs belong
to the same NUMA node assigned to that graph copy. In particular, the NUMA
node observed during `load()` is guaranteed to be the graph copy's assigned
node. A task may therefore determine its NUMA node during `load()` and use
that node to look up GPU resources in `GpuContextManager`.

Establishing this affinity is the framework's responsibility; task callbacks
do not need to set CPU affinity themselves. The simulation and test harness
must provide the same execution guarantee before invoking these callbacks.
A graph copy's numeric ID alone does not identify its CPU or NUMA node.

This rule does not specify how many GPUs belong to a NUMA node or select a
CUDA device for the calling thread. GPU selection and CUDA device binding
remain separate from the framework's CPU-affinity guarantee.

GPUInfra currently requires exactly one GPU on the callback's NUMA node.
`load()` reports an error and fails if lookup finds zero or multiple GPUs;
it must not silently select the first GPU. This temporary GPU restriction
is separate from the graph protocol. Task/frame/worker count calculations
continue to use the resolved GPU-list size.

The lifecycle is:

```text
constructor
  -> registerParameters
  -> define initial parameter values
  -> load
  -> [notifyParameters only when values later change]
  -> start
  -> execute (zero or more frames)
  -> stop
  -> unload
  -> destructor
```

`registerParameters()` is called exactly once for each task instance, before
any other task callback. Initial values are visible to `load()` and are treated
as changed there, so an initial `notifyParameters()` call is not required.

`notifyParameters()` is a change callback, not a mandatory per-run or per-phase
callback. It may run repeatedly after later parameter changes, but only at a
quiescent boundary where no `execute()` call is active. Writing the same values
does not create a change notification.

`start()` and `stop()` bound one execution cycle. `execute()` may be called many
times between them. The real callbacks distinguish execution-cycle
context/state from parameter-derived `load()` resources. This simulation keeps
`start()` and `stop()` as lifecycle guards only; it intentionally leaves the
existing CUDA allocations owned by `load()` and `unload()`.

---

## One-task graph

Consider this graph:

```text
start -> TaskA -> end
```

Suppose one graph copy has six graph threads and must process 200 frames. That
graph copy creates six instances of `TaskA`.

### Cold path (initialization order)

1. Construct each `TaskA` instance and immediately call
   `TaskA::registerParameters()` on it.
2. Define all initial parameter values.
3. Call `TaskA::load()` once on each of the six instances. Each load observes
   and applies the initial values.
4. Call `TaskA::start()` once on each instance to begin the shared execution
   cycle.

### Hot path (run order)

1. Call `TaskA::execute()` 200 times in total: once for each frame. Up to six
   calls can run concurrently because six `TaskA` instances and six graph
   threads are available.
2. If parameters change between quiescent runs or phases, call
   `TaskA::notifyParameters()` once on every affected instance before admitting
   more frame executions. Do not notify merely because a new phase starts.

### End cold path (shutdown order)

1. After every `TaskA::execute()` finishes, call `TaskA::stop()` once on each
   instance.
2. Call `TaskA::unload()` once on each instance.

### The `execute()` model for a one-task graph

The graph can dispatch a frame to `TaskA` when:

1. The frame is ready.
2. A `TaskA` instance is free.
3. An eligible graph thread from the task's NUMA node is free.

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

Suppose one graph copy has six graph threads and must process 200 frames. That
graph copy creates six instances of each task type: six `TaskA` instances, six
`TaskB` instances, and six `TaskC` instances.

### Cold path (initialization order)

1. Construct every `TaskA`, `TaskB`, and `TaskC` instance. Immediately after
   constructing an instance, call its `registerParameters()` exactly once.
2. Define all initial parameter values.
3. Call `load()` once on every task instance. Each load observes and applies
   the initial values.
4. Call `start()` once on every task instance to begin one graph execution
   cycle.

### Hot path (run order)

1. For 200 frames, call each task's `execute()` method 200 times in total.
   Calls from different frames may overlap, but each individual frame must be
   processed in the order `TaskA -> TaskB -> TaskC`.
2. If parameters change at a quiescent boundary, call `notifyParameters()` on
   every affected instance before admitting more executions. A phase boundary
   alone does not trigger notification.

This produces 600 `execute()` calls: 200 for `TaskA`, 200 for `TaskB`, and 200
for `TaskC`.

### End cold path (shutdown order)

1. After all in-flight executions finish, call `stop()` once on every task
   instance.
2. Call `unload()` once on every task instance.

### The `execute()` model for a multi-task graph

The graph can dispatch a frame to its next task when:

1. The frame has completed its previous task, if any.
2. An instance of the required task type is free.
3. An eligible graph thread from the task's NUMA node is free.

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

Therefore, within a graph copy, frames, task instances, and NUMA-local graph
threads are scheduled independently. They are associated only for the duration
of one `execute()` call.

---

## Graph Structure

There are several layers from top to bottom:
1. Graph layer: This layers contain how to execute the task. Compilcate stuffs... The logic is described as above. We just simulate it in our test here. We cannot change it.
2. Task layer: Fix API like the API above and we can define these. It should be clean and easy reading.
3. Algo layer: Main Algo logic. Fully Flexiable inside.
