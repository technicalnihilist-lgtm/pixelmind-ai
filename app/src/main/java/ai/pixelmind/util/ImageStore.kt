package ai.pixelmind.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import java.io.File
import java.io.FileOutputStream

/**
 * Writes generated PNGs to app-private storage and keeps a small decode cache so the gallery
 * scrolls without re-decoding from disk on every frame.
 */
class ImageStore(private val context: Context) {

    private val root: File
        get() = File(context.filesDir, "generations").apply { mkdirs() }

    private val previewCache = object : LruCache<String, Bitmap>(cacheSizeKb()) {
        override fun sizeOf(key: String, value: Bitmap): Int = value.byteCount / 1024
    }

    fun save(bitmap: Bitmap, seed: Long): File {
        // Two generations with the same seed inside the same millisecond used to resolve to the
        // same name, so the second silently overwrote the first one's file.
        val stamp = System.currentTimeMillis()
        var file = File(root, "pm_${stamp}_$seed.png")
        var attempt = 1
        while (file.exists()) {
            file = File(root, "pm_${stamp}_${seed}_${attempt++}.png")
        }
        // Compressed straight into a staging file and renamed into place. A process kill partway
        // through the encode used to leave a truncated "pm_....png" in the gallery directory that
        // looked like a real image to every reader (totalBytes, count, a stale database row) but
        // decoded to nothing. Only the rename makes it visible, and a rename is atomic.
        val staging = File(root, "${file.name}.part")
        try {
            FileOutputStream(staging).use { out ->
                if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                    throw java.io.IOException("the image could not be encoded as PNG")
                }
                out.flush()
            }
            if (!staging.renameTo(file)) {
                // Some filesystems refuse a rename onto a fresh path; the copy is still staged.
                staging.copyTo(file, overwrite = true)
                staging.delete()
            }
        } catch (t: Throwable) {
            runCatching { staging.delete() }
            throw t
        }
        sweepStaging()
        return file
    }

    /**
     * Removes staging files abandoned by a previous process death.
     *
     * Generation is serialised by [ai.pixelmind.manager.GenerationManager] — one save at a time —
     * so any `.part` file present at the start of a save belongs to a run that never finished.
     * Nothing else in the app writes into this directory.
     */
    private fun sweepStaging() {
        root.listFiles { f -> f.isFile && f.name.endsWith(".part") }
            ?.forEach { runCatching { it.delete() } }
    }

    fun load(path: String, maxEdge: Int = 1080): Bitmap? {
        previewCache.get(path)?.let { return it }
        val file = File(path)
        if (!file.exists()) return null
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        var sample = 1
        while (bounds.outWidth / sample > maxEdge || bounds.outHeight / sample > maxEdge) sample *= 2
        val bitmap = BitmapFactory.decodeFile(path, BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }) ?: return null
        previewCache.put(path, bitmap)
        return bitmap
    }

    fun loadThumbnail(path: String, maxEdge: Int = 320): Bitmap? = load(path, maxEdge)

    fun delete(path: String): Boolean {
        previewCache.remove(path)
        return File(path).delete()
    }

    fun totalBytes(): Long =
        root.listFiles { f -> f.isFile && !f.name.endsWith(".part") }?.sumOf { it.length() } ?: 0L

    fun count(): Int = root.listFiles { f -> f.isFile && !f.name.endsWith(".part") }?.size ?: 0

    private fun cacheSizeKb(): Int {
        val maxKb = (Runtime.getRuntime().maxMemory() / 1024).toInt()
        return maxKb / 8
    }
}
