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
        var installed = false
        try {
            // Checked before anything is unpacked: a model tree is unpacked into staging while
            // the model it replaces is still on disk for rollback, so an import needs room for
            // roughly two models.
            spaceCheck(declaredBytes(listOf(uri)))?.let { return@withContext Result.Failed(it) }
            freshStaging()
            val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
                extractZip(input, staging)
            } ?: return@withContext Result.Failed("Could not open the selected file.")
            Log.i(TAG, "unpacked $bytes bytes into staging")
            val result = finishImport()
            installed = result is Result.Imported
            result
        } catch (t: Throwable) {
            Log.e(TAG, "zip import failed", t)
            Result.Failed(failureReason(t))
        } finally {
            // Every failure path leaves through here, including the `return@withContext` ones that
            // used to skip the cleanup: a rejected import kept its half-unpacked tree — hundreds of
            // megabytes of staging — until the next import happened to clear it.
            if (!installed) discardStaging()
        }
    }

    suspend fun importFiles(uris: List<Uri>): Result = withContext(Dispatchers.IO) {
        var installed = false
        try {
            spaceCheck(declaredBytes(uris))?.let { return@withContext Result.Failed(it) }
            freshStaging()
            val seen = HashSet<String>()
            uris.forEach { uri ->
                val name = displayName(uri) ?: return@withContext Result.Failed(
                    "Could not read a file name from the selection."
                )
                val safe = sanitiseEntryPath(name)
                    ?: return@withContext Result.Failed("Unsafe file name: $name")
                // Picking files one by one strips their folders, so two graphs that are both
                // called model.onnx land on the same path and one silently overwrote the other.
                if (!seen.add(safe)) {
                    return@withContext Result.Failed(
                        "Two selected files are both named $safe. Import a .zip instead so the " +
                            "folder names are kept."
                    )
                }
                val target = File(staging, safe)
                target.parentFile?.mkdirs()
                context.contentResolver.openInputStream(uri)?.use { input ->
                    target.outputStream().use { output ->
                        copyLimited(input, output, MAX_IMPORT_BYTES)
                    }
                } ?: return@withContext Result.Failed("Could not open $name.")
            }
            val result = finishImport()
            installed = result is Result.Imported
            result
        } catch (t: Throwable) {
            Log.e(TAG, "file import failed", t)
            Result.Failed(failureReason(t))
        } finally {
            if (!installed) discardStaging()
        }
    }

    /** Drops the imported model so the bundled one is provisioned again on next load. */
    suspend fun restoreBundled(): Result = withContext(Dispatchers.IO) {
        val removed = root.deleteRecursively()
        staging.deleteRecursively()
        previous.deleteRecursively()
        invalidateVerification()
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

    /**
     * Drops the copy kept for rollback. Deleting a model directory is a few hundred megabytes of
     * unlink work, and this was called straight from the ViewModel's coroutine — i.e. on the main
     * thread — right after a successful import. It now runs on IO like every other file operation
     * in this class.
     */
    suspend fun discardPrevious() = withContext(Dispatchers.IO) {
        previous.deleteRecursively()
    }

    /** Bytes on disk for the active model. File-system walk, so it runs on IO. */
    suspend fun activeModelBytes(): Long = withContext(Dispatchers.IO) {
        root.listFiles()?.sumOf { it.length() } ?: 0L
    }

    // ---------------------------------------------------------------------------------------
    // internals
    // ---------------------------------------------------------------------------------------

    private fun freshStaging() {
        staging.deleteRecursively()
        staging.mkdirs()
    }

    /**
     * Removes everything an abandoned import wrote.
     *
     * The staging tree is the bulk of it, but a `.part` file can also be left inside the installed
     * tree if the swap was interrupted after `writeModelFile` staged a component, so both are swept.
     */
    private fun discardStaging() {
        runCatching { staging.deleteRecursively() }
        runCatching {
            root.walkTopDown()
                .onFail { _, _ -> }
                .filter { it.isFile && it.name.endsWith(".part") }
                .forEach { runCatching { it.delete() } }
        }
    }

    /**
     * Size the selection reports for itself, or a non-positive value when it reports none.
     *
     * `OpenableColumns.SIZE` is optional in the Storage Access Framework and several providers
     * leave it null, which is why [ImportSpace] has an estimate for that case rather than treating
     * an unknown size as zero.
     */
    private fun declaredBytes(uris: List<Uri>): Long {
        var total = 0L
        var known = false
        uris.forEach { uri ->
            val size = runCatching {
                context.contentResolver
                    .query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)
                    ?.use { cursor ->
                        if (cursor.moveToFirst() && !cursor.isNull(0)) cursor.getLong(0) else -1L
                    } ?: -1L
            }.getOrDefault(-1L)
            if (size > 0) {
                total += size
                known = true
            }
        }
        return if (known) total else -1L
    }

    /** Null when there is room for the import, otherwise the sentence to show instead. */
    private fun spaceCheck(declared: Long): String? {
        val free = runCatching { context.filesDir.usableSpace }.getOrDefault(Long.MAX_VALUE)
        val installed = root.walkTopDown()
            .onFail { _, _ -> }
            .filter { it.isFile }
            .sumOf { runCatching { it.length() }.getOrDefault(0L) }
        return ImportSpace.shortfall(ImportSpace.requiredBytes(declared, installed), free)
    }

    /**
     * The reason to show for a failed import.
     *
     * A full disk surfaces as an `IOException` whose message is the platform's own `ENOSPC` text,
     * which says nothing about what the user can do. This is the one failure that the pre-flight
     * check above cannot always catch — the provider may not declare a size, or something else may
     * fill the disk mid-copy — so it gets a sentence of its own.
     */
    private fun failureReason(t: Throwable): String {
        val message = t.message ?: t.javaClass.simpleName
        val outOfSpace = message.contains("ENOSPC", ignoreCase = true) ||
            message.contains("No space left", ignoreCase = true) ||
            message.contains("not enough space", ignoreCase = true)
        return if (outOfSpace) {
            "Not enough free space on this device to unpack the model. Free up some space and " +
                "import again — nothing was changed."
        } else {
            message
        }
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
                // The cap is enforced while copying, not after: a single zip entry can decompress
                // to far more than the limit, and the old check only ran once the whole entry had
                // already been written to the device.
                target.outputStream().use { output ->
                    total += copyLimited(zip, output, MAX_IMPORT_BYTES - total)
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
            val json = buildManifestJson(detected, staging)
                ?: return Result.Failed("Could not describe the imported files.")
            ModelManifest.parse(json)
        }

        // Every component the manifest names must exist and match its checksum, otherwise the
        // engine would fail at load time and the rollback would throw the import away anyway.
        manifest.components.forEach { component ->
            val staged = File(staging, component.file)
            if (!staged.exists()) {
                return Result.Failed("Missing ${component.file} inside the import.")
            }
            if (component.sha256.isNotBlank() &&
                !sha256(staged).equals(component.sha256, ignoreCase = true)
            ) {
                return Result.Failed(
                    "${component.file} does not match the checksum in its manifest — the file " +
                        "is truncated or was modified after the manifest was written."
                )
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
        // Cached digests describe the model that was just moved aside.
        invalidateVerification()
        return Result.Imported(manifest.name, manifest.version, manifest.totalBytes)
    }

    private fun invalidateVerification() {
        runCatching { File(context.filesDir, ModelLocator.VERIFIED_FILE).delete() }
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
         * Copies at most [budget] bytes, so an oversized stream is refused instead of filling the
         * disk.
         *
         * It neither opens nor closes either stream: the caller owns both and wraps them in `use`,
         * which is what keeps a failed import from holding a descriptor on the picked document.
         */
        internal fun copyLimited(input: InputStream, output: java.io.OutputStream, budget: Long): Long {
            val buffer = ByteArray(BUFFER)
            var written = 0L
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                written += read
                if (written > budget) throw IllegalStateException("Archive is larger than 4 GB")
                output.write(buffer, 0, read)
            }
            return written
        }

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
        internal fun buildManifestJson(detected: Detected, root: File): String? {
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
