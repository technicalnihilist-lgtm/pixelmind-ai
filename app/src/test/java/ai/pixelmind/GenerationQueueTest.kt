package ai.pixelmind

import ai.pixelmind.data.GenerationRecorder
import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.engine.GeneratedImage
import ai.pixelmind.engine.GenerationProgress
import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.GenerationResult
import ai.pixelmind.engine.GenerationStage
import ai.pixelmind.engine.LocalAIEngine
import ai.pixelmind.engine.ModelInfo
import ai.pixelmind.manager.GenerationManager
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicInteger

/**
 * The queue's concurrency contract, exercised on a real dispatcher but with no device and no
 * ONNX Runtime: ordering under concurrent enqueue, single-run retries, cancellation of the real
 * worker, result delivery with no collector, progress after a job finished, and one initialise
 * per load.
 */
class GenerationQueueTest {

    private val info = ModelInfo(
        name = "test-model",
        version = "1.0",
        architecture = "lcm",
        precision = "int8",
        bytesOnDisk = 0L,
        license = "test",
        stepsRecommended = 4,
        maxResolution = 512
    )

    private val ready = EngineStatus.Ready(info, "test")

    private lateinit var scope: CoroutineScope

    @Before
    fun setUp() {
        scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    }

    @After
    fun tearDown() {
        scope.cancel()
    }

    private fun request(prompt: String) = GenerationRequest(prompt = prompt, steps = 1, guidance = 1.0f)

    private fun ok(request: GenerationRequest) = GenerationResult.Success(
        GeneratedImage(File("/tmp/pixelmind-test/${request.prompt}.png"), 8, 8, 7L, 1L)
    )

    /**
     * Engine double. Records what it was asked to run, in the order it was asked, and notes
     * whether a cancellation actually reached it.
     */
    private inner class FakeEngine(
        initialStatus: EngineStatus,
        /** What [initialize] settles on. Defaults to a ready engine. */
        private val initResult: EngineStatus = ready,
        private val initDelayMs: Long = 0L,
        private val swallowCancellation: Boolean = false,
        private val body: suspend (GenerationRequest, (GenerationProgress) -> Unit) -> GenerationResult =
            { req, onProgress ->
                onProgress(GenerationProgress(GenerationStage.DIFFUSING, 1, 1, 0L))
                ok(req)
            }
    ) : LocalAIEngine {

        @Volatile var liveStatus: EngineStatus = initialStatus
        val runs = AtomicInteger(0)
        val initialiseCalls = AtomicInteger(0)
        val order: MutableList<String> = Collections.synchronizedList(mutableListOf())
        @Volatile var lastProgress: ((GenerationProgress) -> Unit)? = null
        @Volatile var sawCancellation = false

        override suspend fun initialize(): EngineStatus {
            initialiseCalls.incrementAndGet()
            if (initDelayMs > 0) delay(initDelayMs)
            liveStatus = initResult
            return initResult
        }

        override fun status(): EngineStatus = liveStatus

        override fun modelInfo(): ModelInfo? = (liveStatus as? EngineStatus.Ready)?.info

        override suspend fun generate(
            request: GenerationRequest,
            onProgress: (GenerationProgress) -> Unit
        ): GenerationResult {
            runs.incrementAndGet()
            order += request.prompt
            lastProgress = onProgress
            return try {
                body(request, onProgress)
            } catch (c: CancellationException) {
                sawCancellation = true
                // The worst case for the queue: an engine that reports success on the way out.
                if (swallowCancellation) ok(request) else throw c
            }
        }

        override suspend fun release() {
            liveStatus = EngineStatus.Uninitialised
        }
    }

    private class FakeRecorder : GenerationRecorder {
        val recorded = AtomicInteger(0)
        override suspend fun record(request: GenerationRequest, image: GeneratedImage) {
            recorded.incrementAndGet()
        }
    }

    /** Waits for [predicate] to hold, failing the test rather than hanging. */
    private fun awaitTrue(what: String, timeoutMs: Long = 20_000, predicate: () -> Boolean) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            if (predicate()) return
            Thread.sleep(5)
        }
        throw AssertionError("timed out waiting for $what")
    }

    @Test
    fun `concurrent enqueue gives unique ids and runs them in id order`() = runBlocking {
        val engine = FakeEngine(ready)
        val manager = GenerationManager(engine, FakeRecorder(), scope)
        manager.start()

        val threads = 8
        val perThread = 6
        val total = threads * perThread
        val prompts = Collections.synchronizedMap(mutableMapOf<Long, String>())
        val start = CountDownLatch(1)
        val workers = (0 until threads).map { t ->
            Thread {
                start.await()
                repeat(perThread) { i ->
                    val prompt = "prompt-$t-$i"
                    prompts[manager.enqueue(request(prompt))] = prompt
                }
            }.also { it.start() }
        }
        start.countDown()
        workers.forEach { it.join() }

        awaitTrue("every job to reach a terminal state") {
            manager.jobs.value.size == total && manager.jobs.value.all { it.state.isTerminal }
        }

        assertEquals("every enqueue must get its own id", total, prompts.size)
        assertEquals(total, manager.jobs.value.size)
        val expectedOrder = prompts.keys.sorted().map { prompts.getValue(it) }
        assertEquals("jobs must run in the order they were given ids", expectedOrder, engine.order.toList())
    }

    @Test
    fun `retry tapped twice does not run the job twice`() {
        val engine = FakeEngine(ready, body = { _, _ -> GenerationResult.Failed("model refused") })
        val manager = GenerationManager(engine, FakeRecorder(), scope)
        manager.start()

        val id = manager.enqueue(request("a house"))
        awaitTrue("the first attempt to fail") { manager.jobs.value.single().state.isTerminal }
        assertEquals(GenerationManager.JobState.FAILED, manager.jobs.value.single().state)
        assertEquals(1, engine.runs.get())

        manager.retry(id)
        manager.retry(id)
        awaitTrue("the retry to run") { engine.runs.get() >= 2 && manager.jobs.value.single().state.isTerminal }
        Thread.sleep(100) // a third run would have shown up by now
        assertEquals("one retry, one run", 2, engine.runs.get())
    }

    @Test
    fun `fifty queued jobs produce exactly fifty terminal states`() {
        val engine = FakeEngine(ready, body = { req, onProgress ->
            onProgress(GenerationProgress(GenerationStage.DIFFUSING, 1, 1, 0L))
            delay(1)
            ok(req)
        })
        val recorder = FakeRecorder()
        val manager = GenerationManager(engine, recorder, scope)
        manager.start()

        repeat(50) { manager.enqueue(request("job-$it")) }
        awaitTrue("all fifty jobs to finish") {
            manager.jobs.value.size == 50 && manager.jobs.value.all { it.state.isTerminal }
        }

        val jobs = manager.jobs.value
        assertEquals(50, jobs.size)
        assertEquals("no job may be left queued or running", 50, jobs.count { it.state.isTerminal })
        assertEquals(50, jobs.count { it.state == GenerationManager.JobState.SUCCEEDED })
        assertEquals(50, engine.runs.get())
        assertEquals(50, recorder.recorded.get())
        assertNull("the ring must be closed once the queue is empty", manager.progress.value)
    }

    @Test
    fun `every result is delivered even when nothing is collecting`() = runBlocking {
        // More than the 64 slots the channel used to buffer: with trySend on a bounded channel
        // the tail of this burst was dropped and the UI never saw those jobs finish.
        val engine = FakeEngine(ready)
        val manager = GenerationManager(engine, FakeRecorder(), scope)
        manager.start()

        repeat(100) { manager.enqueue(request("job-$it")) }
        awaitTrue("all hundred jobs to finish") {
            manager.jobs.value.size == 100 && manager.jobs.value.all { it.state.isTerminal }
        }

        val received = withTimeout(10_000) { manager.resultEvents.take(100).toList() }
        assertEquals(100, received.size)
        assertTrue(received.all { it is GenerationResult.Success })
    }

    @Test
    fun `progress arriving after the job finished is ignored`() {
        val engine = FakeEngine(ready)
        val manager = GenerationManager(engine, FakeRecorder(), scope)
        manager.start()

        manager.enqueue(request("a lighthouse"))
        awaitTrue("the job to succeed") {
            manager.jobs.value.singleOrNull()?.state == GenerationManager.JobState.SUCCEEDED
        }
        assertNull(manager.progress.value)

        engine.lastProgress?.invoke(GenerationProgress(GenerationStage.DIFFUSING, 3, 8, 42L))
        assertNull("a late callback must not reopen the progress ring", manager.progress.value)
    }

    @Test
    fun `cancelActive cancels the running worker and the job stays cancelled`() {
        val engine = FakeEngine(ready, swallowCancellation = true, body = { req, onProgress ->
            repeat(50) { step ->
                delay(20)
                onProgress(GenerationProgress(GenerationStage.DIFFUSING, step, 50, 0L))
            }
            ok(req)
        })
        val manager = GenerationManager(engine, FakeRecorder(), scope)
        manager.start()

        manager.enqueue(request("a red bicycle"))
        awaitTrue("the job to start running") {
            manager.jobs.value.singleOrNull()?.state == GenerationManager.JobState.RUNNING
        }
        // RUNNING is published under the same lock as the worker handle, but the worker itself is
        // started a moment later. Cancelling in that gap is legitimate — the lazy coroutine is
        // cancelled before its body runs, so nothing is generated and the engine is never entered
        // — and this test used to fail there, waiting 20 s for a cancellation the engine could not
        // see because it had never been called. Wait for the worker to be inside the engine, which
        // is the case this test is about.
        awaitTrue("the worker to enter the engine") { engine.runs.get() == 1 }

        manager.cancelActive()
        awaitTrue("the worker to observe the cancellation") { engine.sawCancellation }
        awaitTrue("the job to settle") { manager.jobs.value.single().state.isTerminal }

        assertEquals(
            "a terminal state must never be overwritten",
            GenerationManager.JobState.CANCELLED,
            manager.jobs.value.single().state
        )
    }

    @Test
    fun `a cancellation produces no result event for the UI to toast`() {
        val engine = FakeEngine(ready, body = { req, onProgress ->
            repeat(50) { step ->
                delay(20)
                onProgress(GenerationProgress(GenerationStage.DIFFUSING, step, 50, 0L))
            }
            ok(req)
        })
        val manager = GenerationManager(engine, FakeRecorder(), scope)
        manager.start()

        manager.enqueue(request("a quiet street"))
        awaitTrue("the worker to enter the engine") { engine.runs.get() == 1 }
        manager.cancelActive()
        awaitTrue("the job to settle") { manager.jobs.value.single().state.isTerminal }

        // Nothing may arrive on the result channel: the ViewModel shows a snackbar for whatever
        // comes through it, and a cancel the user asked for is not something to announce twice.
        val event = runBlocking { withTimeoutOrNull(1_000) { manager.resultEvents.first() } }
        assertNull("cancellation must not raise a message", event)
        assertEquals(GenerationManager.JobState.CANCELLED, manager.jobs.value.single().state)
    }

    @Test
    fun `a failure caused by missing weights is not offered for retry`() {
        val engine = FakeEngine(
            initialStatus = EngineStatus.Uninitialised,
            initResult = EngineStatus.WeightsMissing("no weights are installed")
        )
        val manager = GenerationManager(engine, FakeRecorder(), scope)
        manager.start()

        val id = manager.enqueue(request("a hill"))
        awaitTrue("the job to fail") { manager.jobs.value.single().state.isTerminal }

        val job = manager.jobs.value.single()
        assertEquals(GenerationManager.JobState.FAILED, job.state)
        assertFalse("nothing to retry without weights", job.retryable)

        manager.retry(id)
        Thread.sleep(100)
        assertEquals("retry must not re-run a job that cannot succeed", 0, engine.runs.get())
    }

    @Test
    fun `an ordinary generation failure stays retryable`() {
        val engine = FakeEngine(ready, body = { _, _ -> GenerationResult.Failed("the UNet failed") })
        val manager = GenerationManager(engine, FakeRecorder(), scope)
        manager.start()

        manager.enqueue(request("a tower"))
        awaitTrue("the job to fail") { manager.jobs.value.single().state.isTerminal }

        val job = manager.jobs.value.single()
        assertEquals(GenerationManager.JobState.FAILED, job.state)
        assertTrue("a second attempt may succeed", job.retryable)
    }

    @Test
    fun `two callers that find the engine unloaded create one set of sessions`() = runBlocking {
        val engine = FakeEngine(EngineStatus.Uninitialised, initDelayMs = 100)
        val manager = GenerationManager(engine, FakeRecorder(), scope)

        val first = async { manager.ensureReady() }
        val second = async { manager.ensureReady() }

        assertTrue(first.await().isReady)
        assertTrue(second.await().isReady)
        assertEquals("initialise must run once, not once per caller", 1, engine.initialiseCalls.get())
    }

    @Test
    fun `an engine that is already loaded is adopted without a load`() = runBlocking {
        val engine = FakeEngine(ready)
        val manager = GenerationManager(engine, FakeRecorder(), scope)

        val status = manager.ensureReady()

        assertTrue(status.isReady)
        assertEquals("no second load over a live engine", 0, engine.initialiseCalls.get())
        assertEquals(ready, manager.engineStatus.value)
    }
}
