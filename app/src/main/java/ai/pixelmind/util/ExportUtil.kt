package ai.pixelmind.util

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File

/**
 * Copies a generated PNG into the device's public Pictures collection so it survives app
 * uninstall and shows up in other gallery apps.
 *
 * Scoped storage (API 29+) allows this without any permission. Below API 29 the platform
 * requires WRITE_EXTERNAL_STORAGE, which PixelMind deliberately does not request — there the
 * caller falls back to sharing the file instead.
 */
object ExportUtil {

    val canExportToGallery: Boolean
        get() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q

    fun exportToPictures(context: Context, file: File): Boolean {
        if (!canExportToGallery) return false
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, file.name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/PixelMind")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return false
        var published = false
        try {
            val out = resolver.openOutputStream(uri)
            if (out == null) return false
            out.use { stream ->
                file.inputStream().use { input -> input.copyTo(stream) }
            }
            published = resolver.update(
                uri,
                ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) },
                null,
                null
            ) > 0
            return published
        } catch (t: Throwable) {
            return false
        } finally {
            // A failed copy used to leave an IS_PENDING=1 row behind: an invisible, empty entry in
            // the user's Pictures collection that never cleaned itself up.
            if (!published) runCatching { resolver.delete(uri, null, null) }
        }
    }
}
