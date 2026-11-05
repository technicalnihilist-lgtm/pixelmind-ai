package ai.pixelmind.manager

import ai.pixelmind.data.PixelMindRepository
import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.engine.GenerationProgress
import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.GenerationResult
import ai.pixelmind.engine.LocalAIEngine
import ai.pixelmind.engine.ModelInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job as CoroutineJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Owns the generation queue. The UI submits a request and watches state; it never talks to
 * the inference engine directly and never learns that ONNX Runtime exists.
 *
 * One generation runs at a time — a second concurrent diffusion pass would double peak RAM and
 * push mid-range phones into an OOM kill, so extra requests queue instead.
 */
class GenerationManager(
    private val engine: LocalAIEngine,
    private val repository: PixelMindRepository,
    private val scope: CoroutineScope
) {
    enum class JobState { QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED }

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

    private val _lastResult = MutableStateFlow<GenerationResult?>(null)
    val lastResult: StateFlow<GenerationResult?> = _lastResult.asStateFlow()

    private val results = Channel<GenerationResult>(Channel.BUFFERED)
    /** One-shot results for callers that want to react (snackbars, navigation). */
    val resultEvents = results.receiveAsFlow()

    private val work = Channel<Long>(Channel.UNLIMITED)
    private val initMutex = Mutex()
    private var activeJob: Job? = null
    private var activeTask: CoroutineJob? = null
    private var nextId = 1L

    /**
     * True once [releaseEngine] has taken effect, so a second `onStop` is a no-op.
     *
     * Starts true: nothing is loaded at construction, so the first background transition has
     * nothing to release. [ensureReady] clears it whenever it leaves a loaded engine behind.
     */
    private val engineReleased = AtomicBoolean(true)

    private val _modelInfo = MutableStateFlow<ModelInfo?>(null)
    val modelInfo: StateFlow<ModelInfo?> = _modelInfo.asStateFlow()

    fun start() {
        scope.launch {
            for (jobId in work) {
                val job = _jobs.value.firstOrNull { it.id == jobId } ?: continue
                runJob(job)
            }
        }
    }

    suspend fun ensureReady(): EngineStatus = initMutex.withLock {
        if (_engineStatus.value.isReady) return _engineStatus.value
        _engineStatus.value = EngineStatus.Loading
        val status = engine.initialize()
        _engineStatus.value = status
        _modelInfo.value = engine.modelInfo()
        // Only a loaded engine is something a release has to undo.
        engineReleased.set(!status.isReady)
        status
    }

    fun enqueue(request: GenerationRequest): Long {
        val id = nextId++
        _jobs.value = _jobs.value + Job(id, request, JobState.QUEUED)
        work.trySend(id)
        return id
    }

    fun retry(jobId: Long) {
        val job = _jobs.value.firstOrNull { it.id == jobId } ?: return
        // A queued job is already in the work channel; enqueueing it twice ran the same request
        // back to back once the first pass finished.
        if (job.state == JobState.RUNNING || job.state == JobState.QUEUED) return
        _jobs.value = _jobs.value.map { if (it.id == jobId) it.copy(state = JobState.QUEUED, message = null) else it }
        work.trySend(jobId)
    }

    fun cancelActive() {
        activeTask?.cancel()
        activeJob?.let { job ->
            _jobs.value = _jobs.value.map {
                if (it.id == job.id) it.copy(state = JobState.CANCELLED) else it
            }
        }
    }

    fun clearFinished() {
        _jobs.value = _jobs.value.filter {
            it.state == JobState.QUEUED || it.state == JobState.RUNNING
        }
    }

    /**
     * Drops the loaded model and builds it again — used after importing weights, and after a
     * failed import when the previous model has to come back.
     */
    suspend fun reloadEngine(): EngineStatus {
        if (activeJob != null) return _engineStatus.value
        engine.release()
        _modelInfo.value = null
        _engineStatus.value = EngineStatus.Uninitialised
        return ensureReady()
    }

    /**
     * Drops the loaded sessions when the app leaves the foreground. Safe to call repeatedly: a
     * second `onStop` without an intervening load does nothing, where it used to queue another
     * `engine.release()` behind the engine lock for no reason.
     */
    fun releaseEngine() {
        if (!engineReleased.compareAndSet(false, true)) return
        // A job that is mid-flight when the app backgrounds is being cancelled by this call, so it
        // is recorded as cancelled rather than left RUNNING forever (the engine will report "not
        // running" for it a moment later, and the failure card would then blame the user's prompt).
        val inFlight = activeJob
        activeTask?.cancel()
        if (inFlight != null) {
            updateJob(
                inFlight.id,
                JobState.CANCELLED,
                "Cancelled because the app went to the background."
            )
        }
        _progress.value = null
        _engineStatus.value = EngineStatus.Uninitialised
        _modelInfo.value = null
        // engine.release() takes the engine lock, which an in-flight generation holds until it
        // reaches the next cancellation check — a whole diffusion step. The caller is
        // ProcessLifecycleOwner.onStop(), i.e. the main thread, so the close has to happen off it.
        scope.launch { engine.release() }
    }

    private suspend fun runJob(job: Job) {
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
            updateJob(job.id, JobState.FAILED, reason)
            _progress.value = null
            results.trySend(GenerationResult.Failed(reason))
            return
        }
        // A job cancelled while it sat in the queue (the queue is drained one id at a time, so a
        // cancel can land between the enqueue and this line) must not start running afterwards.
        if (_jobs.value.firstOrNull { it.id == job.id }?.state == JobState.CANCELLED) {
            _progress.value = null
            return
        }
        updateJob(job.id, JobState.RUNNING)
        activeJob = job
        _lastResult.value = null
        val task = scope.launch { execute(job) }
        activeTask = task
        try {
            task.join()
        } finally {
            // Cleared here, not inside the task: a task cancelled before its body ever starts
            // never reaches a finally block of its own, and a leaked activeJob made
            // reloadEngine() a silent no-op for the rest of the process lifetime.
            activeTask = null
            activeJob = null
        }
    }

    private suspend fun execute(job: Job) {
        val result = engine.generate(job.request) { p -> _progress.value = p }
        _progress.value = null
        when (result) {
            is GenerationResult.Success -> {
                repository.record(job.request, result.image)
                updateJob(job.id, JobState.SUCCEEDED)
            }
            is GenerationResult.Failed -> updateJob(job.id, JobState.FAILED, result.reason)
            GenerationResult.Cancelled -> updateJob(job.id, JobState.CANCELLED)
        }
        _lastResult.value = result
        // trySend, not send: with no collector attached (app in the background, screen not
        // composed) a full buffer would suspend here and stall the whole queue forever.
        results.trySend(result)
    }

    private fun updateJob(id: Long, state: JobState, message: String? = null) {
        _jobs.value = _jobs.value.map {
            if (it.id == id) it.copy(state = state, message = message) else it
        }
    }
}
