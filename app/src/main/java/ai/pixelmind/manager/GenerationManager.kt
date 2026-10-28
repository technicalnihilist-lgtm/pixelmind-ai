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
        if (job.state == JobState.RUNNING) return
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

    fun releaseEngine() {
        activeTask?.cancel()
        engine.release()
        _engineStatus.value = EngineStatus.Uninitialised
    }

    private suspend fun runJob(job: Job) {
        updateJob(job.id, JobState.RUNNING)
        activeJob = job
        _lastResult.value = null
        activeTask = scope.launch {
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
            results.send(result)
            activeJob = null
        }
        activeTask?.join()
    }

    private fun updateJob(id: Long, state: JobState, message: String? = null) {
        _jobs.value = _jobs.value.map {
            if (it.id == id) it.copy(state = state, message = message) else it
        }
    }
}
