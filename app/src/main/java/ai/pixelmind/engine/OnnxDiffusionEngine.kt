package ai.pixelmind.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.pixelmind.util.ImageStore
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Random
import kotlin.coroutines.coroutineContext
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
    private var tokenizer: ClipTokenizer? = null
    private var accelerator: String = "CPU"

    private val scheduler = LcmScheduler()
    private val lock = Any()
    private val native = NativeScheduler

    override fun status(): EngineStatus = currentStatus

    override fun modelInfo(): ModelInfo? = modelInfo

    override suspend fun initialize(): EngineStatus = withContext(Dispatchers.Default) {
        if (currentStatus.isReady) return@withContext currentStatus
        currentStatus = EngineStatus.Loading
        try {
            when (val resolution = locator.resolve()) {
                is ModelLocator.Resolution.Missing -> {
                    currentStatus = EngineStatus.WeightsMissing(resolution.detail)
                    return@withContext currentStatus
                }
                is ModelLocator.Resolution.Corrupt -> {
                    currentStatus = EngineStatus.Failed(resolution.detail)
                    return@withContext currentStatus
                }
                is ModelLocator.Resolution.Available -> {
                    val located = resolution.located
                    manifest = located.manifest
                    buildSessions(located.dir, located.manifest)
                    currentStatus = EngineStatus.Ready(
                        info = requireNotNull(modelInfo),
                        accelerator = accelerator
                    )
                    currentStatus
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Engine initialisation failed", t)
            releaseInternal()
            currentStatus = EngineStatus.Failed(t.message ?: t.javaClass.simpleName)
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

                encoder = runtimeEnv.createSession(File(dir, m.textEncoder.file).absolutePath, options)
                unetSession = runtimeEnv.createSession(File(dir, m.unet.file).absolutePath, options)
                vaeSession = runtimeEnv.createSession(File(dir, m.vaeDecoder.file).absolutePath, options)

                textEncoder = encoder
                unet = unetSession
                vaeDecoder = vaeSession
                sessionOptions = options
                accelerator = name
                lastError = null
                break
            } catch (t: Throwable) {
                Log.w(TAG, "Accelerator $name unavailable, falling back", t)
                runCatching { encoder?.close() }
                runCatching { unetSession?.close() }
                runCatching { vaeSession?.close() }
                runCatching { options.close() }
                lastError = t
            }
        }
        lastError?.let { throw it }

        tokenizer = ClipTokenizer.fromFiles(
            vocab = File(dir, m.tokenizer.vocabFile),
            merges = File(dir, m.tokenizer.mergesFile),
            maxLength = m.tokenizer.maxLength
        )

        warmUp(runtimeEnv, m)

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
        Log.i(TAG, "Sessions ready ($accelerator) for ${m.name} ${m.version}")
    }

    override suspend fun generate(
        request: GenerationRequest,
        onProgress: (GenerationProgress) -> Unit
    ): GenerationResult = withContext(Dispatchers.Default) {
        if (!currentStatus.isReady) {
            return@withContext GenerationResult.Failed(
                (currentStatus as? EngineStatus.WeightsMissing)?.detail
                    ?: "The local engine is not ready (${currentStatus.javaClass.simpleName})."
            )
        }
        val weightsMb = (manifest?.totalBytes ?: 0L) / 1_048_576.0
        val fitted = memoryGuard.fitToDevice(request, weightsMb)
            ?: return@withContext GenerationResult.Failed(
                "This device does not have enough free memory to run the model safely."
            )
        if (fitted != request) {
            Log.i(TAG, "Request scaled to ${fitted.width}x${fitted.height} @ ${fitted.steps} steps")
        }

        synchronized(lock) {
            val started = SystemClock.elapsedRealtime()
            val seed = fitted.resolvedSeed()
            var latent: FloatArray? = null
            try {
                val m = requireNotNull(manifest)
                val tok = requireNotNull(tokenizer)
                val envLocal = requireNotNull(env)
                val enc = requireNotNull(textEncoder)
                val unetSession = requireNotNull(unet)
                val vae = requireNotNull(vaeDecoder)

                onProgress(GenerationProgress(GenerationStage.PREPARING, 0, fitted.steps, 0))

                // ---- 1. Text conditioning -------------------------------------------------
                onProgress(GenerationProgress(GenerationStage.ENCODING_TEXT, 0, fitted.steps, elapsed(started)))
                val textStarted = SystemClock.elapsedRealtime()
                val condEmbedding = runTextEncoder(envLocal, enc, m, tok, fitted.conditionalText())
                val uncondEmbedding = if (fitted.guidance > 1.05f) {
                    runTextEncoder(envLocal, enc, m, tok, fitted.negativePrompt.trim())
                } else null

                Log.i(TAG, "text encoding took ${SystemClock.elapsedRealtime() - textStarted} ms")

                // ---- 2. Diffusion loop ----------------------------------------------------
                val textDone = SystemClock.elapsedRealtime()
                val timesteps = scheduler.timesteps(fitted.steps)
                val latentH = fitted.height / 8
                val latentW = fitted.width / 8
                val latentSize = m.latentChannels * latentH * latentW

                val noise = FloatArray(latentSize)
                val rng = Random(seed)
                val scale = scheduler.initialNoiseScale(timesteps[0]).toFloat()
                for (i in noise.indices) noise[i] = (rng.nextGaussian().toFloat() * scale)
                latent = noise

                var stepIndex = 0
                while (stepIndex < timesteps.size) {
                    coroutineContext.ensureActive() // cancellation is honoured between steps
                    val t = timesteps[stepIndex]
                    val prev = if (stepIndex + 1 < timesteps.size) timesteps[stepIndex + 1] else -1

                    val eps = runUnet(envLocal, unetSession, m, latent!!, t, latentH, latentW, condEmbedding)
                    if (uncondEmbedding != null) {
                        val epsUncond = runUnet(envLocal, unetSession, m, latent!!, t, latentH, latentW, uncondEmbedding)
                        val g = fitted.guidance
                        for (i in eps.indices) eps[i] = epsUncond[i] + g * (eps[i] - epsUncond[i])
                    }
                    native.step(latent!!, eps, t, prev, scheduler)
                    stepIndex += 1
                    Log.i(TAG, "step $stepIndex/${timesteps.size} done at ${elapsed(started)} ms")
                    onProgress(GenerationProgress(GenerationStage.DIFFUSING, stepIndex, fitted.steps, elapsed(started)))
                }

                // ---- 3. Decode --------------------------------------------------------------
                onProgress(GenerationProgress(GenerationStage.DECODING, fitted.steps, fitted.steps, elapsed(started)))
                val diffusionDone = SystemClock.elapsedRealtime()
                val decodeStarted = SystemClock.elapsedRealtime()
                val bitmap = decodeLatents(envLocal, vae, m, latent!!, fitted.width, fitted.height)
                Log.i(TAG, "vae decode took ${SystemClock.elapsedRealtime() - decodeStarted} ms")
                latent = null

                // ---- 4. Persist -------------------------------------------------------------
                onProgress(GenerationProgress(GenerationStage.SAVING, fitted.steps, fitted.steps, elapsed(started)))
                val file = imageStore.save(bitmap, seed)
                bitmap.recycle()
                onProgress(GenerationProgress(GenerationStage.DONE, fitted.steps, fitted.steps, elapsed(started)))

                val decodeDone = SystemClock.elapsedRealtime()
                GenerationResult.Success(
                    image = GeneratedImage(
                        file = file,
                        width = fitted.width,
                        height = fitted.height,
                        seed = seed,
                        durationMs = elapsed(started)
                    ),
                    timings = RunTimings(
                        accelerator = accelerator,
                        textEncodeMs = textDone - textStarted,
                        diffusionMs = diffusionDone - textDone,
                        decodeMs = decodeDone - diffusionDone,
                        steps = fitted.steps,
                        width = fitted.width,
                        height = fitted.height
                    )
                )
            } catch (c: kotlinx.coroutines.CancellationException) {
                GenerationResult.Cancelled
            } catch (t: Throwable) {
                Log.e(TAG, "Generation failed", t)
                GenerationResult.Failed(t.message ?: t.javaClass.simpleName, t)
            } finally {
                latent = null
                // Force the collectors to hand memory back before the next request arrives.
                System.gc()
            }
        }
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
            Log.w(TAG, "warm-up failed; generation will still be attempted", t)
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
    ): FloatArray {
        val ids = tokenizer.encode(text)
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { tensor ->
            session.run(mapOf(m.textEncoder.inputNames[0] to tensor)).use { result ->
                val out = result.get(m.textEncoder.outputName)
                    .orElseThrow { IllegalStateException("Text encoder produced no output") } as OnnxTensor
                return out.floatBuffer.copyOut()
            }
        }
    }

    private fun runUnet(
        env: OrtEnvironment,
        session: OrtSession,
        m: ModelManifest,
        latent: FloatArray,
        timestep: Int,
        latentH: Int,
        latentW: Int,
        embedding: FloatArray
    ): FloatArray {
        val scaled = FloatArray(latent.size) { latent[it] * LcmScheduler.LATENT_SCALE }
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
                return out.floatBuffer.copyOut()
            }
        } finally {
            inputs.values.forEach { runCatching { it.close() } }
        }
    }

    private fun decodeLatents(
        env: OrtEnvironment,
        session: OrtSession,
        m: ModelManifest,
        latent: FloatArray,
        width: Int,
        height: Int
    ): Bitmap {
        val unscaled = FloatArray(latent.size) { latent[it] / LcmScheduler.LATENT_SCALE }
        val pixels = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(unscaled),
            longArrayOf(1, m.latentChannels.toLong(), (height / 8).toLong(), (width / 8).toLong())
        ).use { tensor ->
            session.run(mapOf(m.vaeDecoder.inputNames[0] to tensor)).use { result ->
                val out = result.get(m.vaeDecoder.outputName)
                    .orElseThrow { IllegalStateException("VAE decoder produced no output") } as OnnxTensor
                out.floatBuffer.copyOut()
            }
        }

        val plane = width * height
        val argb = IntArray(plane)
        for (i in 0 until plane) {
            val r = toByte(pixels[i])
            val g = toByte(pixels[plane + i])
            val b = toByte(pixels[2 * plane + i])
            argb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun toByte(value: Float): Int {
        val normalised = (value * 0.5f + 0.5f).coerceIn(0f, 1f)
        return (normalised * 255f + 0.5f).toInt()
    }

    override fun release() {
        synchronized(lock) {
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
        tokenizer = null
        manifest = null
    }

    private fun FloatBuffer.copyOut(): FloatArray {
        val copy = FloatArray(remaining())
        duplicate().get(copy)
        return copy
    }

    private fun elapsed(started: Long) = SystemClock.elapsedRealtime() - started

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
