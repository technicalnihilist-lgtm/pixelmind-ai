package ai.pixelmind.util

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import java.io.File

/** Shares a generated PNG through the system chooser. Nothing is uploaded by the app itself. */
object ShareUtil {

    fun shareImage(context: Context, file: File) {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(intent, "Share image"))
    }

    fun sharePrompt(context: Context, prompt: String, seed: Long) {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, "$prompt\n\nseed: $seed · generated offline with PixelMind AI")
        }
        context.startActivity(Intent.createChooser(intent, "Share prompt"))
    }
}
