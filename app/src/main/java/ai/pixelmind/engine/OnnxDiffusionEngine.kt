package ai.pixelmind.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.pixelmind.util.ImageStore
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Random
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.CoroutineContext
import kotlin.math.min

/**
 * The on-device text-to-image pipeline.
 *
 * Four ONNX graphs, all executed locally through ONNX Runtime:
 *
 *   CLIP text encoder -> UNet (LCM distilled, int8) -> VAE decoder -> PNG
 *
 * Nothing here touches the network, and nothing here is stubbed: if the weights are absent the
 * engine reports [EngineStatus.WeightsMissing] and refuses to produce an image rather than
 * returning a placeholder.
 */
class OnnxDiffusionEngine(
    private val context: Context,
    private val locator: ModelLocator,
    private val memoryGuard: MemoryGuard,
    private val imageStore: ImageStore
) : LocalAIEngine {

    @Volatile
    private var currentStatus: EngineStatus = EngineStatus.Uninitialised

    @Volatile
    private var modelInfo: ModelInfo? = null

    private var env: OrtEnvironment? = null
    private var sessionOptions: OrtSession.SessionOptions? = null
    private var textEncoder: OrtSession? = null
    private var unet: OrtSession? = null
    private var vaeDecoder: OrtSession? = null
    private var manifest: ModelManifest? = null

    /**
     * The CLIP vocabulary, parsed on first use rather than here. See [TokenizerLoader]: the
     * sessions are built at app start and again after every background transition, and parsing
     * ~97,000 vocabulary entries at each of those points was startup work for a generation the
     * user may never ask for.
     */
    private var tokenizerLoader: TokenizerLoader? = null
    private var accelerator: String = "CPU"

    private val scheduler = LcmScheduler()

    /**
     * Serialises session construction, generation and release.
     *
     * A [Mutex] rather than a monitor: a generation holds this for the whole diffusion pass, and a
     * monitor made `release()` block whichever thread called it (the main thread, from
     * `ProcessLifecycleOwner.onStop`) until the pass finished. With a mutex the release suspends
     * instead, and — more importantly — cancelling the generating coroutine releases the gate at
     * once, because the between-steps cancellation check throws out of `withLock`.
     */
    private val gate = Mutex()
    private val native = NativeScheduler

    /**
     * Set when sessions are built, cleared by the first generation.
     *
     * The warm-up exists so the first Generate does not absorb the delegates' lazy compilation.
     * Running it at session creation meant paying it at app start — and again on every return
     * from background — whether or not the user ever generated anything. It now runs inside the
     * first pass instead, on exactly the three graphs that pass will use.
     */
    private val warmUpPending = AtomicBoolean(false)

    override fun status(): EngineStatus = currentStatus

    override fun modelInfo(): ModelInfo? = modelInfo

    override suspend fun initialize(): EngineStatus = withContext(Dispatchers.Default) {
        // Building sessions takes seconds and hundreds of megabytes, so it is serialised against
        // release() and against a second concurrent initialise. Without that, a background
        // transition arriving mid-load (release() runs while the sessions are being built) leaves
        // one set of sessions open and unreachable, and the status flips back to Ready afterwards.
        gate.withLock { initialiseLocked() }
    }

    private fun initialiseLocked(): EngineStatus {
        if (currentStatus.isReady) return currentStatus
        currentStatus = EngineStatus.Loading
        return try {
            when (val resolution = locator.resolve()) {
                is ModelLocator.Resolution.Missing -> {
                    currentStatus = EngineStatus.WeightsMissing(resolution.detail)
                }
                is ModelLocator.Resolution.Corrupt -> {
                    currentStatus = EngineStatus.Failed(resolution.detail)
                }
                // Also a Failed state, but with its own sentence: nothing is truncated and nothing
                // has to be re-imported, so telling the user the weights are damaged would be
                // false. The pill reads "Engine error" for both, the detail tells them apart.
                is ModelLocator.Resolution.Unreadable -> {
                    currentStatus = EngineStatus.Failed(resolution.detail)
                }
                is ModelLocator.Resolution.Available -> {
                    val located = resolution.located
                    manifest = located.manifest
                    buildSessions(located.dir, located.manifest)
                    // No requireNotNull here. A missing ModelInfo is a state the engine can reach
                    // (a manifest whose numbers do not add up leaves it unset), and turning that
                    // into an NPE from inside the initialiser surfaced as a failure card reading
                    // "null" with the sessions still open and unreachable.
                    val info = modelInfo
                    currentStatus = if (info != null) {
                        EngineStatus.Ready(info, accelerator)
                    } else {
                        releaseInternal()
                        EngineStatus.Failed(
                            "The model opened but produced no metadata. Its manifest is incomplete."
                        )
                    }
                }
            }
            currentStatus
        } catch (t: Throwable) {
            // Logged before it is mapped: an OutOfMemoryError has no message of its own, so the
            // stack trace is the only record of which allocation failed.
            Log.e(TAG, "Engine initialisation failed", t)
            releaseInternal()
            currentStatus = EngineStatus.Failed(describeFailure(t))
            currentStatus
        }
    }

    private fun buildSessions(dir: File, m: ModelManifest) {
        val runtimeEnv = OrtEnvironment.getEnvironment()
        env = runtimeEnv
        val cores = Runtime.getRuntime().availableProcessors()

        // Try the best delegate first, then fall back. A driver that accepts the session options
        // can still fail while building the graph (unsupported ops are common with NNAPI), so the
        // decision has to be made on session creation, not on addNnapi() returning without error.
        // XNNPACK first, on purpose. NNAPI looks attractive on paper, but its delegate recompiles
        // for every new input shape and pays a per-node dispatch cost — on a graph with Gather,
        // ReduceMean and Resize nodes that turned a sub-second pass into minutes. XNNPACK runs
        // the graph in-process with optimised CPU kernels and no shape recompilation.
        val attempts: List<Pair<String, (OrtSession.SessionOptions) -> Unit>> = listOf(
            "XNNPACK" to { opts: OrtSession.SessionOptions ->
                opts.addXnnpack(mapOf("intra_op_num_threads" to min(4, cores).toString()))
            },
            "NNAPI" to { opts: OrtSession.SessionOptions -> opts.addNnapi() },
            "CPU" to { _: OrtSession.SessionOptions -> }
        )

        var lastError: Throwable? = null
        for ((name, configure) in attempts) {
            val options = OrtSession.SessionOptions()
            var encoder: OrtSession? = null
            var unetSession: OrtSession? = null
            var vaeSession: OrtSession? = null
            try {
                options.setIntraOpNumThreads(min(4, cores))
                options.setInterOpNumThreads(1)
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                configure(options)

                // Timed per graph. Session creation is the dominant part of engine start-up and
                // it is wildly uneven between graphs — the UNet is an order of magnitude slower to
                // open than the decoder — so a single total would not say where the seconds went.
                // Each graph is named as it opens, so a session that fails to build reports
                // *which* graph failed instead of "the engine could not start".
                encoder = createTimed(runtimeEnv, File(dir, m.textEncoder.file), options, GraphName.TEXT_ENCODER)
                unetSession = createTimed(runtimeEnv, File(dir, m.unet.file), options, GraphName.UNET)
                vaeSession = createTimed(runtimeEnv, File(dir, m.vaeDecoder.file), options, GraphName.VAE_DECODER)

                textEncoder = encoder
                unet = unetSession
                vaeDecoder = vaeSession
                sessionOptions = options
                accelerator = name
                lastError = null
                break
            } catch (t: Throwable) {
                // An OOM here is not "this accelerator is unavailable": the next attempt allocates
                // the same sessions and fails the same way, so it is logged at error level and
                // stops the fallback chain instead of being retried twice more.
                if (t is OutOfMemoryError) Log.e(TAG, "Out of memory building the $name sessions", t)
                else Log.w(TAG, "Accelerator $name unavailable, falling back", t)
                runCatching { encoder?.close() }
                runCatching { unetSession?.close() }
                runCatching { vaeSession?.close() }
                runCatching { options.close() }
                lastError = t
                if (t is OutOfMemoryError) break
            }
        }
        lastError?.let { throw it }

        // Parsed by the first generation, not here: see TokenizerLoader.
        tokenizerLoader = TokenizerLoader(
            vocab = File(dir, m.tokenizer.vocabFile),
            merges = File(dir, m.tokenizer.mergesFile),
            maxLength = m.tokenizer.maxLength
        )
        // Same reasoning for the graphs: warmed by the first pass, not by opening the app.
        warmUpPending.set(true)

        modelInfo = ModelInfo(
            name = m.name,
            version = m.version,
            architecture = m.architecture,
            precision = m.precision,
            bytesOnDisk = m.totalBytes,
            license = m.license,
            stepsRecommended = 4,
            maxResolution = 640
        )
        // Once per session, not once per generation: the accelerator is chosen while the sessions
        // are built and never changes afterwards, and a line per generation would bury it.
        Log.i(
            TAG,
            "Sessions ready for ${m.name} ${m.version}: accelerator=$accelerator, " +
                "intra-op=${min(4, cores)} threads, inter-op=1, ${cores} cores"
        )
    }

    override suspend fun generate(
        request: GenerationRequest,
        onProgress: (GenerationProgress) -> Unit
    ): GenerationResult = withContext(Dispatchers.Default) {
        if (!currentStatus.isReady) {
            return@withContext GenerationResult.Failed(notReadyReason(currentStatus))
        }
        val weightsMb = (manifest?.totalBytes ?: 0L) / 1_048_576.0
        val fitted = memoryGuard.fitToDevice(request, weightsMb)
            ?: return@withContext GenerationResult.Failed(
                "This device does not have enough free memory to run the model safely. Close " +
                    "other apps, choose the Fast quality setting, and try again."
            )
        if (fitted != request) {
            Log.i(TAG, "Request scaled to ${fitted.width}x${fitted.height} @ ${fitted.steps} steps")
        }

        // Captured before the gate: the gated section is a plain lambda, so this is how the
        // between-steps cancellation check survives inside it.
        val jobContext = coroutineContext
        val started = SystemClock.elapsedRealtime()
        val seed = fitted.resolvedSeed()

        val pass = try {
            gate.withLock { runPass(jobContext, fitted, seed, started, onProgress) }
        } catch (c: CancellationException) {
            Pass.Cancelled
        } catch (t: Throwable) {
            Log.e(TAG, "Generation failed", t)
            Pass.Failed(describeFailure(t))
        }

        when (pass) {
            Pass.Cancelled -> GenerationResult.Cancelled
            is Pass.Failed -> GenerationResult.Failed(pass.reason)
            is Pass.Decoded -> {
                // PNG encode and the write are I/O, not compute. They used to run on the Default
                // pool, which put a full-resolution encode in front of the next job's text
                // encoding and made the queue look stalled after the ring had reached 97%.
                //
                // The bitmap is recycled on every exit from this branch, and the SAVING progress
                // callback is inside the try for a reason: it runs on the caller's dispatcher and
                // a cancellation delivered there used to skip the `finally` that covered only the
                // save, leaving a full-resolution ARGB_8888 bitmap alive until the next GC.
                val file = try {
                    onProgress(
                        GenerationProgress(GenerationStage.SAVING, fitted.steps, fitted.steps, elapsed(started))
                    )
                    withContext(Dispatchers.IO) { imageStore.save(pass.bitmap, pass.seed) }
                } catch (c: CancellationException) {
                    throw c
                } catch (t: Throwable) {
                    Log.e(TAG, "Could not write the PNG", t)
                    return@withContext GenerationResult.Failed(describeFailure(t), t)
                } finally {
                    pass.bitmap.recycle()
                }
                onProgress(
                    GenerationProgress(GenerationStage.DONE, fitted.steps, fitted.steps, elapsed(started))
                )
                val decodeDone = SystemClock.elapsedRealtime()
                GenerationResult.Success(
                    image = GeneratedImage(
                        file = file,
                        width = fitted.width,
                        height = fitted.height,
                        seed = pass.seed,
                        durationMs = elapsed(started)
                    ),
                    timings = RunTimings(
                        accelerator = accelerator,
                        textEncodeMs = pass.textEncodeMs,
                        diffusionMs = pass.diffusionMs,
                        decodeMs = decodeDone - pass.diffusionDone,
                        steps = fitted.steps,
                        width = fitted.width,
                        height = fitted.height
                    )
                )
            }
        }
    }

    /**
     * Everything that needs the sessions, run under the gate. Returns the decoded bitmap rather
     * than saving it: the save is I/O and belongs outside the section that holds the gate.
     */
    private fun runPass(
        jobContext: CoroutineContext,
        fitted: GenerationRequest,
        seed: Long,
        started: Long,
        onProgress: (GenerationProgress) -> Unit
    ): Pass {
        // The snapshot is taken once and used for the whole pass, so a release() landing
        // mid-run cannot leave a half-null field set behind; if any piece is missing the pass
        // fails with a sentence instead of an NPE from a requireNotNull.
        val sessions = loadedSessions()
        if (!currentStatus.isReady || sessions == null) {
            return Pass.Failed(notReadyReason(currentStatus))
        }
        val m = sessions.manifest

        onProgress(GenerationProgress(GenerationStage.PREPARING, 0, fitted.steps, 0))

        // The graphs are warmed by the first generation that needs them, not by opening the app.
        // Only one pass can be here at a time (the caller holds the gate), and the flag is cleared
        // whatever the outcome: a warm-up that failed leaves the real error to surface from the
        // pass itself, exactly as it did when it ran at session creation.
        if (warmUpPending.compareAndSet(true, false)) {
            warmUp(sessions.env, m)
        }

        // ---- 1-2. Text conditioning and the diffusion loop -------------------------------
        //
        // Both live in one call so the two text embeddings become unreachable the moment the
        // last step has been taken. With classifier-free guidance on they are 2 x 77 x 768
        // floats, the largest Kotlin-side buffers of a pass, and the decoder does not need them:
        // holding them across a full-resolution VAE decode was memory kept for nothing. The
        // OnnxTensor wrappers were already closed per call, so what this frees is the copied-out
        // float data rather than a leaked tensor handle.
        val diffusion = diffuse(jobContext, sessions, fitted, seed, started, onProgress)

        // ---- 3. Decode -----------------------------------------------------------------
        onProgress(GenerationProgress(GenerationStage.DECODING, fitted.steps, fitted.steps, elapsed(started)))
        val diffusionDone = SystemClock.elapsedRealtime()
        val decodeStarted = SystemClock.elapsedRealtime()
        val bitmap = try {
            decodeLatents(
                env = sessions.env,
                session = sessions.vaeDecoder,
                m = m,
                buffers = diffusion.buffers,
                width = fitted.width,
                height = fitted.height,
                jobContext = jobContext
            )
        } finally {
            // The decoder is the last reader of the latent, so the whole working set goes here —
            // on the cancelled and failed paths as well as the successful one.
            diffusion.buffers.release()
        }
        Log.i(TAG, "vae decode took ${SystemClock.elapsedRealtime() - decodeStarted} ms")

        return Pass.Decoded(
            bitmap = bitmap,
            seed = seed,
            textEncodeMs = diffusion.textEncodeMs,
            diffusionMs = diffusion.diffusionMs,
            diffusionDone = diffusionDone
        )
    }

    /** What the denoising half of a pass produced, and the buffers it produced it in. */
    private class Diffusion(
        val buffers: DiffusionBuffers,
        val textEncodeMs: Long,
        val diffusionMs: Long
    )

    /**
     * Text conditioning plus the whole denoising loop.
     *
     * The caller keeps the returned buffers (the decoder still needs the latent) and releases
     * them; everything else allocated here — both embeddings, the timestep array, the RNG — is
     * unreachable as soon as this returns.
     */
    private fun diffuse(
        jobContext: CoroutineContext,
        sessions: LoadedSessions,
        fitted: GenerationRequest,
        seed: Long,
        started: Long,
        onProgress: (GenerationProgress) -> Unit
    ): Diffusion {
        val m = sessions.manifest
        val env = sessions.env
        val latentH = fitted.height / 8
        val latentW = fitted.width / 8
        val latentSize = m.latentChannels * latentH * latentW
        val buffers = DiffusionBuffers(latentSize)
        try {
            onProgress(GenerationProgress(GenerationStage.ENCODING_TEXT, 0, fitted.steps, elapsed(started)))
            val textStarted = SystemClock.elapsedRealtime()
            // Parsed here, on the first generation, and reused by every later one.
            val tokenizer = sessions.tokenizerLoader.get()
            val condEmbedding = runTextEncoder(
                env, sessions.textEncoder, m, tokenizer, fitted.conditionalText()
            )
            val uncondEmbedding = if (fitted.guidance > 1.05f) {
                runTextEncoder(
                    env, sessions.textEncoder, m, tokenizer, fitted.negativePrompt.trim()
                )
            } else null
            val textDone = SystemClock.elapsedRealtime()
            Log.i(TAG, "text encoding took ${textDone - textStarted} ms")

            val timesteps = scheduler.timesteps(fitted.steps)
            val latent = buffers.latent
            val rng = Random(seed)
            val scale = scheduler.initialNoiseScale(timesteps[0]).toFloat()
            for (i in latent.indices) latent[i] = rng.nextGaussian().toFloat() * scale

            // Every buffer the loop touches is allocated once, above, and reused by every step:
            // `scaled` is the VAE-scaled copy the UNet reads, and the two eps buffers are filled
            // in place by runUnet, which reads the graph's output straight into them instead of
            // allocating a fresh FloatArray for each of the two calls per step.
            val eps = buffers.conditionalEps
            var stepIndex = 0
            while (stepIndex < timesteps.size) {
                jobContext.ensureActive() // cancellation is honoured between steps
                val t = timesteps[stepIndex]
                val prev = if (stepIndex + 1 < timesteps.size) timesteps[stepIndex + 1] else -1

                runUnet(env, sessions.unet, m, latent, buffers.scaled, t, latentH, latentW, condEmbedding, eps)
                if (uncondEmbedding != null) {
                    val epsUncond = buffers.unconditionalEps
                    runUnet(
                        env, sessions.unet, m, latent, buffers.scaled, t, latentH, latentW,
                        uncondEmbedding, epsUncond
                    )
                    val g = fitted.guidance
                    for (i in eps.indices) eps[i] = epsUncond[i] + g * (eps[i] - epsUncond[i])
                }
                native.step(latent, eps, t, prev, scheduler)
                stepIndex += 1
                Log.i(TAG, "step $stepIndex/${timesteps.size} done at ${elapsed(started)} ms")
                onProgress(GenerationProgress(GenerationStage.DIFFUSING, stepIndex, fitted.steps, elapsed(started)))
            }

            val diffusionDone = SystemClock.elapsedRealtime()
            return Diffusion(buffers, textDone - textStarted, diffusionDone - textDone)
        } catch (t: Throwable) {
            // Cancelled between steps, or a graph failed. The buffers are not handed to anyone on
            // this path, so they are dropped here rather than left for the collector to find.
            buffers.release()
            throw t
        }
    }

    /** What the gated half of a generation produced. */
    private sealed interface Pass {
        data class Failed(val reason: String) : Pass
        data object Cancelled : Pass
        data class Decoded(
            val bitmap: Bitmap,
            val seed: Long,
            val textEncodeMs: Long,
            val diffusionMs: Long,
            val diffusionDone: Long
        ) : Pass
    }

    /**
     * Runs each graph once with tiny inputs. Delegates compile lazily on first execution, so
     * without this the first Generate of the session absorbs that cost and looks like a hang.
     * Failures here are logged, not fatal: a device that cannot run the warm-up will report the
     * real error during generation.
     */
    private fun warmUp(env: OrtEnvironment, m: ModelManifest) {
        val started = SystemClock.elapsedRealtime()
        val seq = m.tokenizer.maxLength
        val side = 8L
        val latentSize = (m.latentChannels * side * side).toInt()
        try {
            val ids = LongArray(seq) { 49406L }
            OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, seq.toLong())).use { tensor ->
                textEncoder?.run(mapOf(m.textEncoder.inputNames[0] to tensor))?.use { result ->
                    result.get(m.textEncoder.outputName).orElse(null)
                }
            }
            val embedding = FloatArray(seq * 768)
            val latent = FloatArray(latentSize)
            OnnxTensor.createTensor(env, FloatBuffer.wrap(embedding), longArrayOf(1, seq.toLong(), 768)).use { emb ->
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(latent), longArrayOf(1, m.latentChannels.toLong(), side, side)
                ).use { sample ->
                    OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(500L)), longArrayOf(1)).use { step ->
                        unet?.run(
                            mapOf(
                                m.unet.inputNames[0] to sample,
                                m.unet.inputNames[1] to step,
                                m.unet.inputNames[2] to emb
                            )
                        )?.use { result -> result.get(m.unet.outputName).orElse(null) }
                    }
                }
            }
            OnnxTensor.createTensor(
                env, FloatBuffer.wrap(latent), longArrayOf(1, m.latentChannels.toLong(), side, side)
            ).use { latentTensor ->
                vaeDecoder?.run(mapOf(m.vaeDecoder.inputNames[0] to latentTensor))?.use { result ->
                    result.get(m.vaeDecoder.outputName).orElse(null)
                }
            }
            Log.i(TAG, "warm-up finished in ${SystemClock.elapsedRealtime() - started} ms ($accelerator)")
        } catch (t: Throwable) {
            // A warm-up that runs out of memory means the real generation will too; that is worth
            // an error-level line rather than the same warning as an unsupported operator.
            if (t is OutOfMemoryError) Log.e(TAG, "warm-up ran out of memory", t)
            else Log.w(TAG, "warm-up failed; generation will still be attempted", t)
        }
    }

    // ---------------------------------------------------------------------------------------
    // Graph execution
    // ---------------------------------------------------------------------------------------

    private fun runTextEncoder(
        env: OrtEnvironment,
        session: OrtSession,
        m: ModelManifest,
        tokenizer: ClipTokenizer,
        text: String
    ): FloatArray = inGraph(GraphName.TEXT_ENCODER) {
        val ids = tokenizer.encode(text)
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { tensor ->
            session.run(mapOf(m.textEncoder.inputNames[0] to tensor)).use { result ->
                val out = result.get(m.textEncoder.outputName)
                    .orElseThrow { IllegalStateException("Text encoder produced no output") } as OnnxTensor
                out.floatBuffer.copyOut()
            }
        }
    }

    private fun runUnet(
        env: OrtEnvironment,
        session: OrtSession,
        m: ModelManifest,
        latent: FloatArray,
        scaled: FloatArray,
        timestep: Int,
        latentH: Int,
        latentW: Int,
        embedding: FloatArray,
        dest: FloatArray
    ) = inGraph(GraphName.UNET) {
        require(scaled.size == latent.size) { "scratch buffer must match the latent size" }
        require(dest.size == latent.size) { "output buffer must match the latent size" }
        for (i in latent.indices) scaled[i] = latent[i] * LcmScheduler.LATENT_SCALE
        val inputs = HashMap<String, OnnxTensor>()
        try {
            inputs[m.unet.inputNames.getOrElse(0) { "sample" }] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(scaled),
                longArrayOf(1, m.latentChannels.toLong(), latentH.toLong(), latentW.toLong())
            )
            inputs[m.unet.inputNames.getOrElse(1) { "timestep" }] = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(longArrayOf(timestep.toLong())),
                longArrayOf(1)
            )
            inputs[m.unet.inputNames.getOrElse(2) { "encoder_hidden_states" }] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(embedding),
                longArrayOf(1, (embedding.size / 768).toLong(), 768)
            )
            session.run(inputs).use { result ->
                val out = result.get(m.unet.outputName)
                    .orElseThrow { IllegalStateException("UNet produced no output") } as OnnxTensor
                // Read into the caller's buffer rather than copying out into a new array: ONNX
                // Runtime hands back a FloatBuffer over the tensor's own memory, so this is the
                // one copy the JVM cannot avoid and not the second one it used to make.
                out.floatBuffer.copyInto(dest, "UNet", latentH, latentW, m.latentChannels)
            }
        } finally {
            inputs.values.forEach { runCatching { it.close() } }
        }
    }

    /**
     * Copies a graph's float payload into [dest], refusing a payload of the wrong length.
     *
     * The size check is not defensive padding: a UNet whose output does not match the latent
     * used to index past the end of the buffer several statements later, which surfaced as an
     * `ArrayIndexOutOfBoundsException` from the scheduler rather than as the model mismatch it is.
     */
    private fun FloatBuffer.copyInto(
        dest: FloatArray,
        graph: String,
        latentH: Int,
        latentW: Int,
        channels: Int
    ) {
        val available = remaining()
        if (available != dest.size) {
            throw IllegalStateException(
                "$graph returned $available values, but a ${channels}x${latentH}x${latentW} tensor " +
                    "needs ${dest.size}. This model's tensor contract does not match the pipeline; " +
                    "import a model exported for PixelMind."
            )
        }
        duplicate().get(dest)
    }

    /**
     * Decodes the latent to an ARGB bitmap.
     *
     * Two things keep this from holding memory the request no longer needs: the channel planes
     * are read straight out of the decoder's output buffer instead of being copied into a
     * 3 x H x W FloatArray first, and cancellation is checked before the full-resolution pixel
     * array and the bitmap are allocated — a job cancelled during the (uninterruptible) decoder
     * call used to go on to build both of them.
     */
    private fun decodeLatents(
        env: OrtEnvironment,
        session: OrtSession,
        m: ModelManifest,
        buffers: DiffusionBuffers,
        width: Int,
        height: Int,
        jobContext: CoroutineContext
    ): Bitmap = inGraph(GraphName.VAE_DECODER) {
        val latent = buffers.latent
        val unscaled = buffers.decodeScratch
        for (i in latent.indices) unscaled[i] = latent[i] / LcmScheduler.LATENT_SCALE

        val plane = width * height
        // Nothing below can be interrupted once the decoder starts, so the last chance to give
        // up cheaply is now, before a megabyte of pixels and a bitmap are built.
        jobContext.ensureActive()
        val argb = IntArray(plane)

        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(unscaled),
            longArrayOf(1, m.latentChannels.toLong(), (height / 8).toLong(), (width / 8).toLong())
        ).use { tensor ->
            session.run(mapOf(m.vaeDecoder.inputNames[0] to tensor)).use { result ->
                val out = result.get(m.vaeDecoder.outputName)
                    .orElseThrow { IllegalStateException("VAE decoder produced no output") } as OnnxTensor
                val pixels = out.floatBuffer
                val base = pixels.position()
                // The decoder's output is trusted to be 3 x H x W. It is not always: a model with
                // a different VAE scale factor produces a differently sized buffer, and indexing
                // it blindly crashed the process with an ArrayIndexOutOfBounds instead of
                // reporting the mismatch.
                if (pixels.remaining() < plane * 3) {
                    throw IllegalStateException(
                        "VAE decoder returned ${pixels.remaining()} values for a ${width}x$height " +
                            "image (expected ${plane * 3}). This model's VAE does not match the 8x " +
                            "latent grid."
                    )
                }
                for (i in 0 until plane) {
                    val r = toByte(pixels.get(base + i))
                    val g = toByte(pixels.get(base + plane + i))
                    val b = toByte(pixels.get(base + 2 * plane + i))
                    argb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
        Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun toByte(value: Float): Int {
        val normalised = (value * 0.5f + 0.5f).coerceIn(0f, 1f)
        return (normalised * 255f + 0.5f).toInt()
    }

    override suspend fun release() {
        gate.withLock {
            releaseInternal()
            currentStatus = EngineStatus.Uninitialised
        }
    }

    private fun releaseInternal() {
        runCatching { unet?.close() }
        runCatching { textEncoder?.close() }
        runCatching { vaeDecoder?.close() }
        runCatching { sessionOptions?.close() }
        unet = null
        textEncoder = null
        vaeDecoder = null
        sessionOptions = null
        tokenizerLoader = null
        manifest = null
        warmUpPending.set(false)
    }

    /**
     * One consistent view of everything a generation needs, or null if the engine is not loaded.
     *
     * Every field is @Volatile-unsafe on its own (they are written under [lock] by buildSessions
     * and cleared by releaseInternal), so reading them together under the same lock is what makes
     * "the engine is ready" and "these sessions exist" the same statement.
     */
    private fun loadedSessions(): LoadedSessions? {
        val e = env ?: return null
        val m = manifest ?: return null
        val t = tokenizerLoader ?: return null
        val te = textEncoder ?: return null
        val u = unet ?: return null
        val v = vaeDecoder ?: return null
        return LoadedSessions(e, m, t, te, u, v)
    }

    private data class LoadedSessions(
        val env: OrtEnvironment,
        val manifest: ModelManifest,
        val tokenizerLoader: TokenizerLoader,
        val textEncoder: OrtSession,
        val unet: OrtSession,
        val vaeDecoder: OrtSession
    )

    /**
     * Opens one graph and logs how long it took.
     *
     * The line is per graph on purpose: session creation is the biggest single cost of engine
     * start-up and the three graphs are not comparable — a 600 MB UNet can take ten times what the
     * decoder takes — so the total alone would not locate the cost.
     */
    private fun createTimed(
        env: OrtEnvironment,
        file: File,
        options: OrtSession.SessionOptions,
        graph: GraphName
    ): OrtSession {
        val started = SystemClock.elapsedRealtime()
        val session = inGraph(graph) { env.createSession(file.absolutePath, options) }
        Log.i(TAG, "session created: ${file.name} in ${SystemClock.elapsedRealtime() - started} ms")
        return session
    }

    /**
     * Runs [block] and attributes any failure to [graph].
     *
     * Cancellation and an out-of-memory error pass through unwrapped: the first is not a failure,
     * and the second already has a sentence that names the setting to change — burying it inside
     * "The UNet failed:" would make it harder to act on, not easier.
     */
    private inline fun <T> inGraph(graph: GraphName, block: () -> T): T = try {
        block()
    } catch (c: CancellationException) {
        throw c
    } catch (o: OutOfMemoryError) {
        throw o
    } catch (f: GraphFailure) {
        throw f
    } catch (t: Throwable) {
        throw GraphFailure(graph, describeFailure(t), t)
    }

    private fun FloatBuffer.copyOut(): FloatArray {
        val copy = FloatArray(remaining())
        duplicate().get(copy)
        return copy
    }

    private fun elapsed(started: Long) = SystemClock.elapsedRealtime() - started

    /** One true sentence about why the engine cannot generate right now. */
    private fun notReadyReason(status: EngineStatus): String = when (status) {
        is EngineStatus.WeightsMissing -> status.detail
        is EngineStatus.Failed -> "The local engine could not start: ${status.detail}"
        EngineStatus.Loading -> "The local engine is still loading the model."
        EngineStatus.Uninitialised ->
            "The local engine is not running. Open the app again to reload the model."
        is EngineStatus.Ready -> "The local engine is not ready."
    }

    /**
     * The prompt the text encoder sees for the conditional pass: the user's words plus the
     * style's photographic/art direction. The negative prompt is *not* appended here — it forms
     * the unconditional pass instead, which is what classifier-free guidance subtracts.
     */
    private fun GenerationRequest.conditionalText(): String = buildString {
        append(prompt.trim())
        if (style.promptSuffix.isNotBlank()) append(", ").append(style.promptSuffix)
    }

    private companion object {
        const val TAG = "OnnxDiffusionEngine"
    }
}
