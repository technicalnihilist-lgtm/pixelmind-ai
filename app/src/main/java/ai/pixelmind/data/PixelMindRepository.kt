package ai.pixelmind.data

import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.GeneratedImage
import ai.pixelmind.util.ImageStore
import ai.pixelmind.util.StorageUsage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File

/** Single source of truth for gallery content, favourites and prompt history. */
class PixelMindRepository(
    private val db: PixelMindDb,
    private val imageStore: ImageStore,
    private val appDataDir: File
) : GenerationRecorder {
    private val _generations = MutableStateFlow<List<GenerationRecord>>(emptyList())
    val generations: StateFlow<List<GenerationRecord>> = _generations.asStateFlow()

    private val _promptHistory = MutableStateFlow<List<String>>(emptyList())
    val promptHistory: StateFlow<List<String>> = _promptHistory.asStateFlow()

    suspend fun refresh() = withContext(Dispatchers.IO) {
        publish()
    }

    override suspend fun record(request: GenerationRequest, image: GeneratedImage) = withContext(Dispatchers.IO) {
        // One transaction, one commit: the gallery row and the prompt-history row are written
        // together or not at all.
        db.recordGeneration(
            GenerationRecord(
                id = 0,
                prompt = request.prompt.trim(),
                negativePrompt = request.negativePrompt.trim(),
                style = request.style.name,
                aspect = request.aspect.name,
                width = image.width,
                height = image.height,
                steps = request.steps,
                seed = image.seed,
                filePath = image.file.absolutePath,
                createdAt = System.currentTimeMillis(),
                durationMs = image.durationMs,
                favorite = false
            )
        )
        publish()
    }

    suspend fun toggleFavorite(record: GenerationRecord) = withContext(Dispatchers.IO) {
        db.setFavorite(record.id, !record.favorite)
        publish()
    }

    suspend fun delete(record: GenerationRecord) = withContext(Dispatchers.IO) {
        imageStore.delete(record.filePath)
        db.deleteGeneration(record.id)
        publish()
    }

    fun storageBytes(): Long = imageStore.totalBytes()

    /**
     * Everything the app keeps in its own private directory: the generated PNGs, the provisioned
     * weights, the database and its journal, the verification sidecar and any import staging left
     * mid-flight.
     *
     * This is what the Settings card's "Total app data" row shows. That row used to add the images
     * and the weights together and call the result "total app data", which understated it by
     * everything else in the directory — small next to 1.2 GB of weights, but not true.
     */
    fun appDataBytes(): Long = StorageUsage.treeBytes(appDataDir)

    private fun publish() {
        _generations.value = db.queryAll()
        _promptHistory.value = db.recentPrompts(PROMPT_HISTORY_LIMIT)
    }

    companion object {
        private const val PROMPT_HISTORY_LIMIT = 30
    }
}
