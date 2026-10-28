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

data class GeneratedImage(
    val file: File,
    val width: Int,
    val height: Int,
    val seed: Long,
    val durationMs: Long
)

sealed interface GenerationResult {
    data class Success(val image: GeneratedImage) : GenerationResult
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

    /** Frees sessions, tensors and native arenas. Called when the app goes to background. */
    fun release()
}
