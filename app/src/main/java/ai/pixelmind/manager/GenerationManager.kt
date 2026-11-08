package ai.pixelmind.manager

import ai.pixelmind.data.GenerationRecorder
import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.engine.GenerationProgress
import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.GenerationResult
import ai.pixelmind.engine.LocalAIEngine
import ai.pixelmind.engine.ModelInfo
import ai.pixelmind.engine.describeFailure
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job as CoroutineJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Owns the generation queue. The UI submits a request and watches state; it never talks to
 * the inference engine directly and never learns that ONNX Runtime exists.
 *
 * One generation runs at a time — a second concurrent diffusion pass would double peak RAM and
 * push mid-range phones into an OOM kill, so extra requests queue instead.
 *
 * Concurrency model, all of it deliberate:
 *
 *  * `queueLock` guards id assignment, the job list and the work channel together, so the order
 *    jobs are given ids in is the order they are executed in.
 *  * `activeLock` guards the worker handle, so a cancel can never miss a worker that is starting
 *    and can never target one that has already finished.
 *  * `initMutex` serialises loading, so two callers that both find the engine unloaded create
 *    one set of sessions, not two.
 *  * Terminal job states are final. Only [retry] re-opens one.
 */
class GenerationManager(
    private val engine: LocalAIEngine,
    private val recorder: GenerationRecorder,
    private val scope: CoroutineScope
) {
    enum class JobState {
        QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED;

        /** A job in a terminal state is never moved again by the queue; only [retry] re-opens it. */
        val isTerminal: Boolean get() = this == SUCCEEDED || this == FAILED || this == CANCELLED
    }

    data class Job(
        val id: Long,
        val request: GenerationRequest,
        val state: JobState,
        val message: String? = null
    )

    private val _engineStatus = MutableStateFlow<EngineStatus>(EngineStatus.Uninitialised)
    val engineStatus: StateFlow<EngineStatus> = _engineStatus.asStateFlow()

    private val _jobs = MutableStateFlow<List<Job>>(emptyList())
    val jobs: StateFlow<List<Job>> = _jobs.asStateFlow()

    private val _progress = MutableStateFlow<GenerationProgress?>(null)
    val progress: StateFlow<GenerationProgress?> = _progress.asStateFlow()

    // There used to be a `lastResult: StateFlow<GenerationResult?>` here, written on every
    // finished job and read by nothing: the UI reacts to `resultEvents` instead. It held the last
    // GenerationResult — and on the failure path the Throwable with its stack trace — for the
    // rest of the process lifetime. A result is a one-shot event; the queue no longer keeps a
    // copy of one. Nothing in a result holds a Bitmap either: GeneratedImage carries the saved
    // File, and the decoded bitmap is recycled inside the engine.

    /**
     * Unbounded on purpose. With a bounded buffer, `trySend` silently dropped results whenever
     * the UI was not collecting (backgrounded, or simply not composed yet) and a burst of queued
     * jobs produced fewer snackbars and fewer gallery refreshes than there were jobs. The bound
     * that matters is the number of queued jobs, and that is a queue policy, not a channel one.
     */
    private val results = Channel<GenerationResult>(Channel.UNLIMITED)

    /** One-shot results for callers that want to react (snackbars, navigation). */
    val resultEvents = results.receiveAsFlow()

    private val work = Channel<Long>(Channel.UNLIMITED)
    private val initMutex = Mutex()

    /** Guards [nextId], [pending] and the append to the job list — one order for ids and sends. */
    private val queueLock = Any()

    /** Guards [activeJob] and [activeTask] so a cancel sees a consistent pair. */
    private val activeLock = Any()

    private var activeJob: Job? = null
    private var activeTask: CoroutineJob? = null

    private val nextId = AtomicLong(1)

    /** Ids currently sitting in [work], so a job can never be queued twice. */
    private val pending = HashSet<Long>()

    private val started = AtomicBoolean(false)

    /**
     * True once [releaseEngine] has taken effect, so a second `onStop` is a no-op.
     *
     * Starts true: nothing is loaded at construction, so the first background transition has
     * nothing to release. [ensureReady] clears it whenever it leaves a loaded engine behind.
     */
    private val engineReleased = AtomicBoolean(true)

    private val _modelInfo = MutableStateFlow<ModelInfo?>(null)
    val modelInfo: StateFlow<ModelInfo?> = _modelInfo.asStateFlow()

    /** Starts the queue drain. Idempotent: a second call would otherwise run a second drain. */
    /** Starts the queue drain. Idempotent: a second call would otherwise run a second drain. */
    fun start() {
        if (!started.compareAndSet(false, true)) return
        scope.launch {
            for (jobId in work) {
                val job = _jobs.value.firstOrNull { it.id == jobId } ?: continue
                runJob(job)
            }
        }
    }

    suspend fun ensureReady(): EngineStatus = initMutex.withLock {
        val current = _engineStatus.value
        if (current.isReady) return current
        // The engine can be loaded while the manager's view of it is not: a release that has not
        // finished unwinding, or a reload that completed a moment ago. Adopting the live status
        // skips the `Loading` state the UI would otherwise paint over a model that is already in
        // memory, and skips an initialise that would return immediately anyway.
        val live = engine.status()
        if (live.isReady) {
            setStatus(live)
            _modelInfo.value = engine.modelInfo()
            engineReleased.set(false)
            return live
        }
        setStatus(EngineStatus.Loading)
        val status = engine.initialize()
        setStatus(status)
        _modelInfo.value = engine.modelInfo()
        // Only a loaded engine is something a release has to undo.
        engineReleased.set(!status.isReady)
        status
    }

    /**
     * Queues a request and returns its id.
     *
     * Id assignment, the job-list append and the channel send happen under one lock. Without
     * that, two threads calling this at once could take the same id (a plain `nextId++` is a
     * read-modify-write) and could lose one another's list append, which left a job in the work
     * channel with no job record behind it — the drain loop then skipped it silently.
     */
    fun enqueue(request: GenerationRequest): Long {
        val id: Long
        synchronized(queueLock) {
            id = nextId.getAndIncrement()
            _jobs.update { it + Job(id, request, JobState.QUEUED) }
            pending.add(id)
            work.trySend(id)
        }
        return id
    }

    /**
     * Re-queues a finished job. A queued or running job is left alone, and [pending] makes sure
     * a job is only ever present in the work channel once — tapping Retry twice quickly used to
     * put the same id in the channel twice and run the request back to back.
     */
    fun retry(jobId: Long) {
        synchronized(queueLock) {
            val job = _jobs.value.firstOrNull { it.id == jobId } ?: return
            if (!job.state.isTerminal) return
            if (!pending.add(jobId)) return
            _jobs.update { list ->
                list.map {
                    if (it.id == jobId) it.copy(state = JobState.QUEUED, message = null) else it
                }
            }
            work.trySend(jobId)
        }
    }

    /**
     * Cancels the worker that is actually running, then records the job as cancelled.
     *
     * The handle is read under the same lock [runJob] writes it under, so this either sees the
     * worker that is about to start (and cancels it before its first suspension point) or sees
     * nothing because no worker exists — never a handle to a coroutine that already finished.
     */
    fun cancelActive() {
        synchronized(activeLock) {
            val job = activeJob ?: return
            activeTask?.cancel()
            finishJob(job.id, JobState.CANCELLED)
        }
    }

    fun clearFinished() {
        _jobs.update { list -> list.filter { !it.state.isTerminal } }
    }

    /**
     * Drops the loaded model and builds it again — used after importing weights, and after a
     * failed import when the previous model has to come back.
     */
    suspend fun reloadEngine(): EngineStatus {
        synchronized(activeLock) { if (activeJob != null) return _engineStatus.value }
        engine.release()
        _modelInfo.value = null
        setStatus(EngineStatus.Uninitialised)
        return ensureReady()
    }

    /**
     * Drops the loaded sessions when the app leaves the foreground. Safe to call repeatedly: a
     * second `onStop` without an intervening load does nothing, where it used to queue another
     * `engine.release()` behind the engine gate for no reason.
     */
    fun releaseEngine() {
        if (!engineReleased.compareAndSet(false, true)) return
        // A job that is mid-flight when the app backgrounds is being cancelled by this call, so it
        // is recorded as cancelled rather than left RUNNING forever (the engine will report "not
        // running" for it a moment later, and the failure card would then blame the user's prompt).
        val inFlight = synchronized(activeLock) {
            val job = activeJob
            activeTask?.cancel()
            job
        }
        if (inFlight != null) {
            finishJob(
                inFlight.id,
                JobState.CANCELLED,
                "Cancelled because the app went to the background."
            )
        }
        _progress.value = null
        setStatus(EngineStatus.Uninitialised)
        _modelInfo.value = null
        // The engine's gate is held by an in-flight generation until it reaches its next
        // cancellation check, so this call must not block. The caller is
        // ProcessLifecycleOwner.onStop(), i.e. the main thread.
        scope.launch { engine.release() }
    }

    private suspend fun runJob(job: Job) {
        // The id has left the channel, so a retry may queue it again from here on.
        synchronized(queueLock) { pending.remove(job.id) }

        // A job queued before the app was backgrounded reaches the engine after its sessions have
        // been released, and used to fail with "open the app again to reload the model" while the
        // app was open and could reload it. Loading first is what the caller would have had to do.
        val status = ensureReady()
        if (!status.isReady) {
            val reason = when (status) {
                is EngineStatus.WeightsMissing -> status.detail
                is EngineStatus.Failed -> status.detail
                EngineStatus.Loading -> "The local engine is still loading the model."
                EngineStatus.Uninitialised ->
                    "The local engine is not running. Open the app again to reload the model."
                is EngineStatus.Ready -> "The local engine is not ready."
            }
            finishJob(job.id, JobState.FAILED, reason)
            _progress.value = null
            results.trySend(GenerationResult.Failed(reason))
            return
        }

        // Created lazy and published before it is started: the state change to RUNNING and the
        // worker handle are set under one lock, so cancelActive either cancels this worker before
        // its first suspension point or finds no worker at all. It used to be possible to mark a
        // job RUNNING, then have a cancel land in the gap before the handle was stored — the
        // cancel did nothing and the generation ran to completion under a "cancelled" label.
        val task = scope.launch(start = CoroutineStart.LAZY) { execute(job) }
        synchronized(activeLock) {
            val queued = _jobs.value.firstOrNull { it.id == job.id }?.state == JobState.QUEUED
            if (!queued) {
                task.cancel()
                _progress.value = null
                return
            }
            _jobs.update { list ->
                list.map { if (it.id == job.id) it.copy(state = JobState.RUNNING) else it }
            }
            activeJob = job
            activeTask = task
        }
        task.start()
        try {
            task.join()
        } finally {
            // Cleared here, not inside the task: a task cancelled before its body ever starts
            // never reaches a finally block of its own, and a leaked activeJob made
            // reloadEngine() a silent no-op for the rest of the process lifetime.
            synchronized(activeLock) {
                if (activeTask === task) {
                    activeTask = null
                    activeJob = null
                }
            }
        }
    }

    private suspend fun execute(job: Job) {
        // Progress is forwarded only while this execution is live. A callback that arrives after
        // the job has been marked finished used to re-open the progress ring on the Studio screen
        // for a job the UI had already reported as done.
        val finished = AtomicBoolean(false)
        val result: GenerationResult = try {
            engine.generate(job.request) { p -> if (!finished.get()) _progress.value = p }
        } catch (c: CancellationException) {
            finished.set(true)
            _progress.value = null
            finishJob(job.id, JobState.CANCELLED)
            throw c
        } catch (t: Throwable) {
            Log.e(TAG, "Generation threw", t)
            GenerationResult.Failed(describeFailure(t), t)
        }
        finished.set(true)
        _progress.value = null
        when (result) {
            is GenerationResult.Success -> {
                // A gallery row that fails to write must not turn a generated image into a
                // failed job: the PNG is already on disk and the result is real.
                try {
                    recorder.record(job.request, result.image)
                } catch (c: CancellationException) {
                    throw c
                } catch (t: Throwable) {
                    Log.e(TAG, "Could not record the generation", t)
                }
                finishJob(job.id, JobState.SUCCEEDED)
            }
            is GenerationResult.Failed -> finishJob(job.id, JobState.FAILED, result.reason)
            GenerationResult.Cancelled -> finishJob(job.id, JobState.CANCELLED)
        }
        results.trySend(result)
    }

    /**
     * Moves a job to a terminal state, unless it is already in one.
     *
     * The guard is what stops a cancelled job being relabelled: cancellation marks the job while
     * the engine is still unwinding, and the result that arrives afterwards must not overwrite it.
     */
    private fun finishJob(id: Long, state: JobState, message: String? = null) {
        _jobs.update { list ->
            list.map {
                if (it.id == id && !it.state.isTerminal) it.copy(state = state, message = message)
                else it
            }
        }
    }

    private fun setStatus(next: EngineStatus) {
        _engineStatus.value = EngineStatusReducer.next(_engineStatus.value, next)
    }

    private companion object {
        const val TAG = "GenerationManager"
    }
}
