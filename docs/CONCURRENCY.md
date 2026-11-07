# Concurrency notes (Day 2 of the optimisation programme)

Ten items, 11–20 in `OPTIMIZATION.md`. This file records what each one turned out to be and
what the code now guarantees, so a later run does not have to re-derive it. Nothing here is a
device measurement: no phone or emulator was attached to this run. Everything below was
established by reading the code and by unit tests on the JVM.

## The model

`GenerationManager` owns the queue and is the only thing that talks to `LocalAIEngine`.

| Guard | Protects | Why it exists |
| --- | --- | --- |
| `queueLock` | `nextId`, `pending`, the append to the job list, the send to the work channel | id order and execution order are the same order |
| `activeLock` | `activeJob`, `activeTask` | a cancel cannot miss a worker that is starting, or target one that finished |
| `initMutex` | the load path | two callers that both find the engine unloaded create one set of sessions |
| `Mutex gate` (engine) | session construction, generation, release | release never blocks a thread, and cancellation frees the gate at once |

Job states are `QUEUED → RUNNING → (SUCCEEDED | FAILED | CANCELLED)`. A terminal state is
final; only `retry` re-opens a job, and only from a terminal state.

## Item by item

**11 — `cancelActive` cancels the real worker.** It already cancelled the coroutine that runs
the generation, not the drain loop. What was wrong was the window around it: the job was marked
`RUNNING` and the worker handle was stored in two separate steps, so a cancel landing in between
found no handle, cancelled nothing, and the generation ran to completion under a `CANCELLED`
label. The worker is now created `CoroutineStart.LAZY`, and the state change and the handle are
published in one critical section, then the worker is started. A cancel therefore either
cancels it before its first suspension point or sees no worker at all.

**12 — FIFO under concurrent enqueue.** Broken. `nextId++` on a plain `Long` is a
read-modify-write, so two threads could take the same id; and `_jobs.value = _jobs.value + Job(…)`
is a lost-update, so a concurrent enqueue could leave an id in the work channel with no job
record behind it — the drain loop skipped such a job silently. Both now happen under one lock,
together with the channel send. `nextId` is an `AtomicLong`. Test: 48 jobs enqueued from 8
threads, ids unique, and the engine sees them in id order.

**13 — retry cannot run a job twice.** Broken, in a way the "is it already queued" check did not
catch: a cancelled queued job could be re-sent to the work channel while its original id was
still buffered there, so the request ran twice. A `pending` set now guarantees at most one copy
of an id in the channel, and the terminal→`QUEUED` transition is made under the same lock as the
send. Test: two `retry` calls in a row produce exactly one extra run.

**14 — `Mutex` instead of `synchronized` around the generation.** Done. The engine held a
monitor for the whole diffusion pass, so `release()` — called from `ProcessLifecycleOwner.onStop`,
i.e. the main thread — blocked that thread until the pass ended. `LocalAIEngine.release()` is now
`suspend` and takes a `Mutex`, and the generation holds the same mutex, so a release suspends
instead of parking a thread, and cancelling the generating coroutine releases the gate
immediately (the between-steps `ensureActive` throws out of `withLock`).

**15 — no progress after the job is marked finished.** Guarded. The engine's progress callback
now writes through an `AtomicBoolean` that is set before the job's terminal state is published,
so a callback that arrives late cannot re-open the progress ring on the Studio screen. Test: the
captured callback is invoked after the job succeeded and `progress` stays null.

**16 — monotonic `engineStatus`.** Two changes. `EngineStatusReducer` refuses `Ready → Loading`,
and `ensureReady` adopts a live ready engine (`engine.status().isReady`) instead of painting
`Loading` and calling `initialize()` again. Every other transition — including `Ready →
Uninitialised` on release and `Ready → Failed` — is still reported, because those are true.

**17 — dispatcher audit.** Three real findings, all fixed:
* `ImageStore.save` (PNG encode plus the file write) ran on `Dispatchers.Default`, i.e. on the
  same pool as the next job's text encoding. It now runs on `Dispatchers.IO`, outside the engine
  gate — the decode result is returned from the gated section and saved after it.
* `ModelImporter.discardPrevious()` deleted a model directory from the main thread (it is called
  from the ViewModel's import coroutine). Now `suspend` + `Dispatchers.IO`.
* `ModelImporter.activeModelBytes()` walked the file system on the caller's dispatcher. Now
  `suspend` + `Dispatchers.IO`. (It has no callers today; item 81 will decide whether it goes.)
`OnnxDiffusionEngine` stays on `Dispatchers.Default` for the graph execution itself, which is
compute, and the repository was already entirely on `Dispatchers.IO`.

**18 — result events cannot be dropped.** Broken. The channel was `Channel.BUFFERED` (64) and
results were sent with `trySend`, so a burst of more than 64 jobs with no collector lost the
tail — no snackbar, no gallery refresh for those jobs. It is now `Channel.UNLIMITED`. The real
bound is the number of queued jobs, which is a queue policy (item 166) rather than a channel one.
Test: 100 jobs, no collector, 100 results received afterwards.

**19 — two `initialize()` calls racing.** Already handled, at both levels: the engine's
`initialiseLocked` runs under the gate and returns the current status if it is already ready, and
`GenerationManager.ensureReady` serialises callers on `initMutex`. Verified by test rather than
by argument: two concurrent `ensureReady()` calls on an unloaded engine produce exactly one
`initialize()`.

**20 — stress test.** Added: 50 queued jobs, exactly 50 terminal states, 50 engine runs, 50
recorded rows, and the progress ring closed at the end.

## What is not verified

* No device or emulator was available. Peak RSS, per-step timings, cancellation latency on real
  weights, and whether ART's collector behaves as expected after the removal of the forced GC
  (Day 1) are all unmeasured.
* The `Mutex` change is verified by compilation and by the JVM tests only. Its behaviour under a
  real ONNX Runtime generation — where the gated section runs for tens of seconds — has not been
  observed on a device.
* `Channel.UNLIMITED` trades a bound for correctness; if item 166 sets a queue cap, the channel
  is bounded by that cap in practice, not by itself.

## Tests

`./gradlew :app:testReleaseUnitTest` — 48 tests, 0 failures (34 before this day).
`GenerationQueueTest` (8) and `EngineStatusReducerTest` (6) are new.
