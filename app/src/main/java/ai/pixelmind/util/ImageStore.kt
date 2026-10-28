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
        val file = File(root, "pm_${System.currentTimeMillis()}_$seed.png")
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        return file
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
        root.listFiles()?.sumOf { it.length() } ?: 0L

    fun count(): Int = root.listFiles()?.size ?: 0

    private fun cacheSizeKb(): Int {
        val maxKb = (Runtime.getRuntime().maxMemory() / 1024).toInt()
        return maxKb / 8
    }
}
