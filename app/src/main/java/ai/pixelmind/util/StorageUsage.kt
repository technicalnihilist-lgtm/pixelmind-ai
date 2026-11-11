package ai.pixelmind.util

import java.io.File
import java.util.Locale

/**
 * The arithmetic and the formatting behind the Settings storage card.
 *
 * It lives here, as pure functions over `Long`s and a directory, so the numbers the card shows can
 * be pinned by a unit test — the card itself cannot be, and "the total is the sum of its parts" is
 * exactly the kind of claim that quietly stops being true.
 */
object StorageUsage {

    /**
     * Human-readable byte count, in the same units and precision the Settings card has always used.
     *
     * The locale is fixed to [Locale.US] on purpose. `String.format` without one uses the device
     * locale, so the same 22.2 MB read "22,2 MB" on a comma-decimal device and, on a locale with
     * its own digit shapes, changed the digits as well. Storage figures are technical values, not
     * prose, and the card's other numbers are ASCII.
     */
    fun formatBytes(bytes: Long): String = when {
        bytes >= 1_073_741_824 -> String.format(Locale.US, "%.2f GB", bytes / 1_073_741_824.0)
        bytes >= 1_048_576 -> String.format(Locale.US, "%.1f MB", bytes / 1_048_576.0)
        bytes >= 1024 -> String.format(Locale.US, "%.0f KB", bytes / 1024.0)
        else -> "$bytes B"
    }

    /**
     * Total size of every regular file under [root], or 0 when it does not exist.
     *
     * A directory that is being written to while it is walked can change under the iterator; a
     * vanished file is skipped rather than allowed to abort the total.
     */
    fun treeBytes(root: File): Long {
        if (!root.exists()) return 0L
        var total = 0L
        root.walkTopDown()
            .onFail { _, _ -> }
            .filter { it.isFile }
            .forEach { file -> total += runCatching { file.length() }.getOrDefault(0L) }
        return total
    }

    /**
     * The part of the app's private storage that is neither a generated image nor model weights:
     * the database, its journal, the verification sidecar, import staging.
     */
    fun otherBytes(appDataBytes: Long, imageBytes: Long, modelBytes: Long): Long =
        (appDataBytes - imageBytes - modelBytes).coerceAtLeast(0L)
}
