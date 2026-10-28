package ai.pixelmind.data

/** One row of local history. Everything here was produced on this device. */
data class GenerationRecord(
    val id: Long,
    val prompt: String,
    val negativePrompt: String,
    val style: String,
    val aspect: String,
    val width: Int,
    val height: Int,
    val steps: Int,
    val seed: Long,
    val filePath: String,
    val createdAt: Long,
    val durationMs: Long,
    val favorite: Boolean
)
