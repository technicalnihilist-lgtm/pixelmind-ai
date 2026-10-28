package ai.pixelmind.data

import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.GeneratedImage
import ai.pixelmind.util.ImageStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

/** Single source of truth for gallery content, favourites and prompt history. */
class PixelMindRepository(
    private val db: PixelMindDb,
    private val imageStore: ImageStore
) {
    private val _generations = MutableStateFlow<List<GenerationRecord>>(emptyList())
    val generations: StateFlow<List<GenerationRecord>> = _generations.asStateFlow()

    private val _promptHistory = MutableStateFlow<List<String>>(emptyList())
    val promptHistory: StateFlow<List<String>> = _promptHistory.asStateFlow()

    suspend fun refresh() = withContext(Dispatchers.IO) {
        _generations.value = queryAll()
        _promptHistory.value = db.recentPrompts(30)
    }

    suspend fun record(request: GenerationRequest, image: GeneratedImage) = withContext(Dispatchers.IO) {
        db.insertGeneration(
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
        db.rememberPrompt(request.prompt)
        _generations.value = queryAll()
        _promptHistory.value = db.recentPrompts(30)
    }

    suspend fun toggleFavorite(record: GenerationRecord) = withContext(Dispatchers.IO) {
        db.writableDatabase.execSQL(
            "UPDATE ${PixelMindDb.TABLE_GENERATIONS} SET favorite = ? WHERE id = ?",
            arrayOf(if (record.favorite) 0 else 1, record.id)
        )
        _generations.value = queryAll()
    }

    suspend fun delete(record: GenerationRecord) = withContext(Dispatchers.IO) {
        imageStore.delete(record.filePath)
        db.writableDatabase.delete(
            PixelMindDb.TABLE_GENERATIONS, "id = ?", arrayOf(record.id.toString())
        )
        _generations.value = queryAll()
    }

    fun storageBytes(): Long = imageStore.totalBytes()

    private fun queryAll(): List<GenerationRecord> =
        db.readableDatabase.rawQuery(
            "SELECT id, prompt, negative_prompt, style, aspect, width, height, steps, seed, " +
                "file_path, created_at, duration_ms, favorite " +
                "FROM ${PixelMindDb.TABLE_GENERATIONS} ORDER BY created_at DESC",
            null
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(
                        GenerationRecord(
                            id = cursor.getLong(0),
                            prompt = cursor.getString(1),
                            negativePrompt = cursor.getString(2),
                            style = cursor.getString(3),
                            aspect = cursor.getString(4),
                            width = cursor.getInt(5),
                            height = cursor.getInt(6),
                            steps = cursor.getInt(7),
                            seed = cursor.getLong(8),
                            filePath = cursor.getString(9),
                            createdAt = cursor.getLong(10),
                            durationMs = cursor.getLong(11),
                            favorite = cursor.getInt(12) == 1
                        )
                    )
                }
            }
        }
}
