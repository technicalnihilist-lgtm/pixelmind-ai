package ai.pixelmind.engine

import ai.pixelmind.util.StorageUsage

/**
 * The free-space arithmetic for a model import, kept pure so it can be tested.
 *
 * A model import is the largest write the app ever performs: the unpacked tree lands in a staging
 * directory while the model it is going to replace is still on disk (it is kept for rollback), so
 * the space needed is roughly *twice* a model, not one. Running out halfway through used to mean a
 * bare `ENOSPC` from a stream write, surfaced as whatever the exception said, after minutes of
 * copying.
 */
internal object ImportSpace {

    /**
     * Room kept free after the import: the database, the next generated image and the journal all
     * need somewhere to go, and a device with zero bytes free behaves badly everywhere else too.
     */
    const val HEADROOM_BYTES = 32L * 1024 * 1024

    /**
     * Bytes that must be free before unpacking starts.
     *
     * [declaredBytes] is the size the document provider reports for the selection, or a
     * non-positive value when it does not report one — a `content://` document from some providers
     * has no known length until it is read. In that case the current model's size stands in for the
     * import's, which is the honest estimate available: a model is not meaningfully smaller or
     * larger than the one it replaces without the user knowing.
     */
    fun requiredBytes(declaredBytes: Long, installedBytes: Long): Long {
        val installed = installedBytes.coerceAtLeast(0L)
        val incoming = if (declaredBytes > 0) declaredBytes else installed
        return incoming + installed + HEADROOM_BYTES
    }

    /** Null when [required] fits in [free], otherwise a sentence naming both figures. */
    fun shortfall(required: Long, free: Long): String? {
        if (free >= required) return null
        val needed = required - free
        return "Not enough free space: this import needs about " +
            "${StorageUsage.formatBytes(required)} free, and only " +
            "${StorageUsage.formatBytes(free)} is available. Free up at least " +
            "${StorageUsage.formatBytes(needed)} and import again."
    }
}
