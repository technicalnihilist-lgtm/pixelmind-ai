package ai.pixelmind.engine

import android.graphics.Bitmap
import java.io.File

/** Which pipeline stage is currently running, so the UI can narrate honestly. */
enum class GenerationStage { PREPARING, ENCODING_TEXT, DIFFUSING, DECODING, SAVING, DONE }

data class GenerationProgress(
    val stage: GenerationStage,
    val step: Int,
    val totalSteps: Int,
    val elapsedMs: Long
) {
    /** 0f..1f across the whole pipeline, not just the diffusion loop. */
    val fraction: Float
        get() {
            val stageBase = when (stage) {
                GenerationStage.PREPARING -> 0f
                GenerationStage.ENCODING_TEXT -> 0.05f
                GenerationStage.DIFFUSING -> 0.10f
                GenerationStage.DECODING -> 0.90f
                GenerationStage.SAVING -> 0.97f
                GenerationStage.DONE -> 1f
            }
            if (stage != GenerationStage.DIFFUSING) return stageBase
            val within = if (totalSteps == 0) 0f else step.toFloat() / totalSteps
            return 0.10f + within * 0.80f
        }
}

/** Where the wall-clock time went, measured on the device. Shown in the UI, not just logged. */
data class RunTimings(
    val accelerator: String,
    val textEncodeMs: Long,
    val diffusionMs: Long,
    val decodeMs: Long,
    val steps: Int,
    val width: Int,
    val height: Int
) {
    val perStepMs: Long get() = if (steps > 0) diffusionMs / steps else 0
    val totalMs: Long get() = textEncodeMs + diffusionMs + decodeMs
}

data class GeneratedImage(
    val file: File,
    val width: Int,
    val height: Int,
    val seed: Long,
    val durationMs: Long
)

sealed interface GenerationResult {
    data class Success(val image: GeneratedImage, val timings: RunTimings? = null) : GenerationResult
    data class Failed(val reason: String, val cause: Throwable? = null) : GenerationResult
    data object Cancelled : GenerationResult
}

data class ModelInfo(
    val name: String,
    val version: String,
    val architecture: String,
    val precision: String,
    val bytesOnDisk: Long,
    val license: String,
    val stepsRecommended: Int,
    val maxResolution: Int
)

sealed interface EngineStatus {
    data object Uninitialised : EngineStatus
    data object Loading : EngineStatus
    data class Ready(val info: ModelInfo, val accelerator: String) : EngineStatus
    /** Weights are absent or fail checksum. Never faked into a "ready" state. */
    data class WeightsMissing(val detail: String) : EngineStatus
    data class Failed(val detail: String) : EngineStatus

    val isReady: Boolean get() = this is Ready
}

/** The graphs a failure can be attributed to, so a message can say which one broke. */
internal enum class GraphName(val label: String) {
    TEXT_ENCODER("text encoder"),
    UNET("UNet"),
    VAE_DECODER("VAE decoder")
}

/**
 * A failure that belongs to one graph.
 *
 * ONNX Runtime reports a bad session, a rejected input shape and an unsupported operator with the
 * same kind of native string, so the old failure card said "generation failed" without saying
 * where. Wrapping at the call site is the only place that knows which graph was running.
 */
internal class GraphFailure(
    val graph: GraphName,
    val reason: String,
    cause: Throwable? = null
) : Exception(reason, cause)

/**
 * Turns a thrown failure into one sentence a user can act on.
 *
 * `t.message ?: "unknown"` was used everywhere, which meant an [OutOfMemoryError] — whose message
 * is null on ART — reached the failure card as the bare word "OutOfMemoryError", and an
 * allocation failure deep inside ONNX Runtime reached it as whatever native string happened to be
 * in flight. The mapping is deliberately small: it only claims things that are true.
 *
 * Two properties are load-bearing and tested: no branch can leak a filesystem path (a native
 * message routinely carries one, and the UI has no business showing `/data/user/0/...`), and an
 * out-of-memory failure names the setting to change rather than just saying "memory".
 */
internal fun describeFailure(t: Throwable): String = when (t) {
    is GraphFailure -> "The ${t.graph.label} failed: ${redactPaths(t.reason)}"
    is OutOfMemoryError -> OUT_OF_MEMORY_SENTENCE
    is SecurityException ->
        "The app is not allowed to read the model files. Reinstall the app to restore them."
    is java.io.FileNotFoundException ->
        "A model file could not be opened. It may have been removed; reimport the model."
    is org.json.JSONException ->
        "A model file is not valid JSON, so it could not be read. Reimport the model."
    is java.io.IOException ->
        "A file could not be read or written. Check that this device has free space and try again."
    else -> redactPaths(t.message?.takeIf { it.isNotBlank() } ?: t.javaClass.simpleName)
}

/**
 * The out-of-memory sentence, with the setting that actually reduces the working set.
 *
 * "Fast" is not a guess: [QualityTier.FAST] runs 4 steps at a 0.75 resolution scale, which is the
 * smallest configuration the pipeline supports, so it is the one thing the user can change that
 * makes the next attempt cheaper.
 */
internal const val OUT_OF_MEMORY_SENTENCE =
    "This device ran out of memory while running the model. Close other apps, choose the Fast " +
        "quality setting, and try again."

/**
 * Removes filesystem paths from a message.
 *
 * Any absolute path with at least two segments is replaced. A single-segment token like "/8" or a
 * fraction is left alone, so ordinary sentences survive untouched.
 */
internal fun redactPaths(text: String): String =
    ABSOLUTE_PATH.replace(text, "<file>")

private val ABSOLUTE_PATH = Regex("""(?:file://)?(?:/[A-Za-z0-9._\-]+){2,}/?""")

/**
 * The only surface the rest of the app is allowed to touch. Everything about ONNX Runtime,
 * tensor layouts and quantisation lives behind this interface, so the UI layer stays light
 * and the backend can be swapped without touching a single Composable.
 */
interface LocalAIEngine {
    suspend fun initialize(): EngineStatus
    fun status(): EngineStatus
    fun modelInfo(): ModelInfo?

    /**
     * Runs a full text-to-image pass. Suspends; cancellation is cooperative and checked
     * between diffusion steps so a cancelled job releases its tensors immediately.
     */
    suspend fun generate(
        request: GenerationRequest,
        onProgress: (GenerationProgress) -> Unit
    ): GenerationResult

    /**
     * Frees sessions, tensors and native arenas. Called when the app goes to background.
     *
     * Suspending rather than blocking: the engine serialises a generation against this call, so a
     * blocking release would park the caller's thread (on the main thread, for a whole diffusion
     * step) waiting for the gate. Suspending lets the caller be cancelled or moved instead.
     */
    suspend fun release()
}
