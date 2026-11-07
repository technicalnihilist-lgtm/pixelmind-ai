package ai.pixelmind.data

import ai.pixelmind.engine.GeneratedImage
import ai.pixelmind.engine.GenerationRequest

/**
 * The one thing the queue needs from storage: turn a finished generation into a gallery row.
 *
 * Kept as an interface so the queue can be exercised without a real SQLite database — the
 * ordering and cancellation guarantees in [ai.pixelmind.manager.GenerationManager] are about the
 * queue, not about Android, and a test that needs a device to prove them proves them late.
 */
interface GenerationRecorder {
    suspend fun record(request: GenerationRequest, image: GeneratedImage)
}
