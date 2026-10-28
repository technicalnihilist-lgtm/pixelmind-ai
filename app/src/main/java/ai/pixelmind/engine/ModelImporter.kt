package ai.pixelmind.engine

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/**
 * Imports a model from the device's own storage — a `.zip`, or a set of files picked from the
 * system file manager.
 *
 * Uses the Storage Access Framework, so the app still declares **no permissions at all**: the user
 * grants access to one document, and that grant is all the app ever sees.
 *
 * The import is transactional. Files are unpacked into a staging directory, validated, then
 * swapped in; if the engine cannot load the result, the previous model is restored.
 */
class ModelImporter(private val context: Context) {

    sealed interface Result {
        data class Imported(val name: String, val version: String, val bytes: Long) : Result
        data class Failed(val reason: String) : Result
    }

    private val root: File get() = File(context.filesDir, MODELS_DIR)
    private val staging: File get() = File(context.filesDir, "$MODELS_DIR.staging")
    private val previous: File get() = File(context.filesDir, "$MODELS_DIR.previous")

    suspend fun importZip(uri: Uri): Result = withContext(Dispatchers.IO) {
        try {
            freshStaging()
            val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
                extractZip(input, staging)
            } ?: return@withContext Result.Failed("Could not open the selected file.")
            Log.i(TAG, "unpacked $bytes bytes into staging")
            finishImport()
        } catch (t: Throwable) {
            Log.e(TAG, "zip import failed", t)
            staging.deleteRecursively()
            Result.Failed(t.message ?: t.javaClass.simpleName)
        }
    }

    suspend fun importFiles(uris: List<Uri>): Result = withContext(Dispatchers.IO) {
        try {
            freshStaging()
            uris.forEach { uri ->
                val name = displayName(uri) ?: return@withContext Result.Failed(
                    "Could not read a file name from the selection."
                )
                val safe = sanitiseEntryPath(name)
                    ?: return@withContext Result.Failed("Unsafe file name: $name")
                val target = File(staging, safe)
                target.parentFile?.mkdirs()
                context.contentResolver.openInputStream(uri)?.use { input ->
                    target.outputStream().use { output -> input.copyTo(output, BUFFER) }
                } ?: return@withContext Result.Failed("Could not open $name.")
            }
            finishImport()
        } catch (t: Throwable) {
            Log.e(TAG, "file import failed", t)
            staging.deleteRecursively()
            Result.Failed(t.message ?: t.javaClass.simpleName)
        }
    }

    /** Drops the imported model so the bundled one is provisioned again on next load. */
    suspend fun restoreBundled(): Result = withContext(Dispatchers.IO) {
        val removed = root.deleteRecursively()
        staging.deleteRecursively()
        previous.deleteRecursively()
        if (removed) {
            Result.Imported("Bundled model", "—", 0L)
        } else {
            Result.Failed("Could not clear the imported model.")
        }
    }

    /** Puts the previous model back after a failed load. */
    suspend fun rollback(): Boolean = withContext(Dispatchers.IO) {
        if (!previous.exists()) return@withContext false
        root.deleteRecursively()
        previous.renameTo(root)
    }

    fun discardPrevious() {
        previous.deleteRecursively()
    }

    fun activeModelBytes(): Long = root.listFiles()?.sumOf { it.length() } ?: 0L

    // ---------------------------------------------------------------------------------------
    // internals
    // ---------------------------------------------------------------------------------------

    private fun freshStaging() {
        staging.deleteRecursively()
        staging.mkdirs()
    }

    private fun extractZip(input: InputStream, destination: File): Long {
        var total = 0L
        ZipInputStream(input.buffered()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                if (entry.isDirectory) {
                    zip.closeEntry()
                    continue
                }
                val safe = sanitiseEntryPath(entry.name)
                if (safe == null) {
                    Log.w(TAG, "skipping unsafe zip entry: ${entry.name}")
                    zip.closeEntry()
                    continue
                }
                val target = File(destination, safe)
                // Defence in depth: even after sanitising, refuse anything that resolves outside.
                if (!target.canonicalPath.startsWith(destination.canonicalPath + File.separator)) {
                    Log.w(TAG, "skipping out-of-tree entry: ${entry.name}")
                    zip.closeEntry()
                    continue
                }
                target.parentFile?.mkdirs()
                target.outputStream().use { output ->
                    val copied = zip.copyTo(output, BUFFER)
                    total += copied
                }
                if (total > MAX_IMPORT_BYTES) {
                    throw IllegalStateException("Archive is larger than 4 GB")
                }
                zip.closeEntry()
            }
        }
        return total
    }

    /** Validates the staged tree, writes a manifest if needed, then swaps it into place. */
    private fun finishImport(): Result {
        val files = staging.walkTopDown().filter { it.isFile }.toList()
        if (files.isEmpty()) return Result.Failed("The selection contained no files.")

        val manifestFile = files.firstOrNull { it.name == ModelManifest.FILE_NAME }
        val manifest: ModelManifest = if (manifestFile != null) {
            runCatching { ModelManifest.parse(manifestFile.readText()) }.getOrElse {
                return Result.Failed("model_manifest.json is unreadable: ${it.message}")
            }
        } else {
            val detected = detectComponents(files.map { it.relativeTo(staging).path })
                ?: return Result.Failed(
                    "Could not tell the graphs apart. Include unet, text_encoder and vae_decoder " +
                        ".onnx files, a clip/ folder with vocab.json and merges.txt, and " +
                        "model_manifest.json — or pick a .zip that has them."
                )
            val json = buildManifestJson(detected, files, staging)
                ?: return Result.Failed("Could not describe the imported files.")
            ModelManifest.parse(json)
        }

        // Every component the manifest names must exist, otherwise the engine will fail later.
        manifest.components.forEach { component ->
            if (!File(staging, component.file).exists()) {
                return Result.Failed("Missing ${component.file} inside the import.")
            }
        }

        // Swap: current -> previous, staging -> current.
        previous.deleteRecursively()
        if (root.exists() && !root.renameTo(previous)) {
            return Result.Failed("Could not move the current model aside.")
        }
        if (!staging.renameTo(root)) {
            previous.renameTo(root)
            return Result.Failed("Could not install the imported model.")
        }
        return Result.Imported(manifest.name, manifest.version, manifest.totalBytes)
    }

    private fun displayName(uri: Uri): String? {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst() && !cursor.isNull(0)) return cursor.getString(0)
            }
        return uri.lastPathSegment?.substringAfterLast('/')
    }

    companion object {
        private const val TAG = "ModelImporter"
        private const val MODELS_DIR = "models"
        private const val BUFFER = 1 shl 16
        private const val MAX_IMPORT_BYTES = 4L * 1024 * 1024 * 1024

        /**
         * Rejects absolute paths, parent traversal and empty names. Returns a path relative to the
         * model directory, or null when the entry must not be written.
         */
        internal fun sanitiseEntryPath(raw: String): String? {
            val normalised = raw.replace('\\', '/').trim()
            if (normalised.isEmpty()) return null
            if (normalised.startsWith("/")) return null
            if (normalised.contains(":")) return null // drive letters, stream wrappers
            val parts = normalised.split('/').filter { it.isNotEmpty() && it != "." }
            if (parts.isEmpty()) return null
            if (parts.any { it == ".." }) return null
            return parts.joinToString("/")
        }

        /**
         * Finds the five files the pipeline needs by name, which is how Hugging Face exports and
         * most `.zip` bundles lay them out: `unet/model.onnx`, `text_encoder/model.onnx`,
         * `vae_decoder/model.onnx`, plus the CLIP tokenizer files.
         */
        internal fun detectComponents(paths: List<String>): Detected? {
            fun pick(predicate: (String) -> Boolean): String? =
                paths.filter { it.lowercase().endsWith(".onnx") }
                    .filter(predicate)
                    .minByOrNull { it.length }

            val unet = pick { it.lowercase().contains("unet") }
            val textEncoder = pick {
                it.lowercase().contains("text_encoder") || it.lowercase().contains("text-encoder")
            }
            val vae = pick { it.lowercase().contains("vae_decoder") || it.lowercase().contains("vae-decoder") }
            val vocab = paths.firstOrNull { it.lowercase().endsWith("vocab.json") }
            val merges = paths.firstOrNull { it.lowercase().endsWith("merges.txt") }
            if (unet == null || textEncoder == null || vae == null || vocab == null || merges == null) {
                return null
            }
            return Detected(unet, textEncoder, vae, vocab, merges)
        }

        internal data class Detected(
            val unet: String,
            val textEncoder: String,
            val vaeDecoder: String,
            val vocab: String,
            val merges: String
        )

        /** Writes a manifest describing files that arrived without one. */
        internal fun buildManifestJson(
            detected: Detected,
            files: List<File>,
            root: File
        ): String? {
            fun component(relPath: String): JSONObject {
                val file = File(root, relPath)
                if (!file.exists()) return JSONObject()
                return JSONObject()
                    .put("file", relPath)
                    .put("bytes", file.length())
                    .put("sha256", sha256(file))
            }
            val components = JSONArray()
            listOf(
                detected.textEncoder, detected.unet, detected.vaeDecoder,
                detected.vocab, detected.merges
            ).forEach { components.put(component(it)) }

            return JSONObject()
                .put("name", "Imported model")
                .put("version", "imported")
                .put("architecture", "sd15-lcm")
                .put("precision", "unknown")
                .put("license", "see the source of the imported weights")
                .put("latent_channels", 4)
                .put("sample_size", 64)
                .put("components", components)
                .put(
                    "text_encoder",
                    JSONObject()
                        .put("file", detected.textEncoder)
                        .put("inputs", JSONArray().put("input_ids"))
                        .put("output", "last_hidden_state")
                )
                .put(
                    "unet",
                    JSONObject()
                        .put("file", detected.unet)
                        .put("inputs", JSONArray().put("sample").put("timestep").put("encoder_hidden_states"))
                        .put("output", "out_sample")
                )
                .put(
                    "vae_decoder",
                    JSONObject()
                        .put("file", detected.vaeDecoder)
                        .put("inputs", JSONArray().put("latent_sample"))
                        .put("output", "sample")
                )
                .put(
                    "tokenizer",
                    JSONObject()
                        .put("vocab", detected.vocab)
                        .put("merges", detected.merges)
                        .put("max_length", 77)
                        .put("bos", "<|startoftext|>")
                        .put("eos", "<|endoftext|>")
                        .put("pad", "<|endoftext|>")
                )
                .toString(2)
        }

        internal fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(BUFFER)
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
