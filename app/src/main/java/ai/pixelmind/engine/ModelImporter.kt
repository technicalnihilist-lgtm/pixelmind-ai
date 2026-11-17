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

    /**
     * A read failure on one file of the selection.
     *
     * An import is a loop over documents, and the loop used to report the last exception verbatim:
     * "open failed: EACCES" with no indication of which of the five picked files it was. The name
     * is the only thing that makes the message actionable.
     */
    internal class ImportFileFailure(val file: String, cause: Throwable) : Exception(cause)

    private val root: File get() = ModelSwap.root(context.filesDir)
    private val staging: File get() = ModelSwap.staging(context.filesDir)
    private val previous: File get() = ModelSwap.previous(context.filesDir)

    suspend fun importZip(uri: Uri): Result = withContext(Dispatchers.IO) {
        var installed = false
        try {
            // Checked before anything is unpacked: a model tree is unpacked into staging while
            // the model it replaces is still on disk for rollback, so an import needs room for
            // roughly two models.
            spaceCheck(declaredBytes(listOf(uri)))?.let { return@withContext Result.Failed(it) }
            recoverInterruptedSwap()
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
            recoverInterruptedSwap()
            freshStaging()
            val seen = HashSet<String>()
            uris.forEach { uri ->
                val raw = displayName(uri) ?: return@withContext Result.Failed(
                    "Could not read a file name from the selection."
                )
                // A document picked from a provider that keeps no extension (a download manager,
                // a chat app's cache) arrived as "unet" and was then rejected as a set of files
                // nobody could tell apart. The MIME type restores an extension for the tokenizer
                // files; for a graph there is nothing to recover it from, so the message says so
                // instead of blaming the archive.
                val name = nameWithExtension(raw, uri)
                    ?: return@withContext Result.Failed(
                        "\"$raw\" has no file extension, so it cannot be matched to a model " +
                            "component. Rename it to end in .onnx (or .json / .txt for the " +
                            "tokenizer files) and import again — or import a .zip, which keeps " +
                            "the original names."
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
                try {
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        target.outputStream().use { output ->
                            copyLimited(input, output, MAX_IMPORT_BYTES)
                        }
                    } ?: return@withContext Result.Failed(
                        "Could not open $name: the file manager would not return its contents."
                    )
                } catch (t: Throwable) {
                    // Attributed to the file it happened on, not to "the import".
                    throw ImportFileFailure(name, t)
                }
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
        ModelSwap.rollback(root, previous)
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

    /**
     * Puts back a tree left in `models.previous` by an import that was killed between its two
     * renames.
     *
     * Run before staging is touched, because the swap deletes `previous` on its way in: without
     * this, the first import after a killed one would delete the only surviving copy of the model.
     */
    private fun recoverInterruptedSwap() {
        if (ModelSwap.recover(root, previous)) {
            Log.i(TAG, "restored the previous model after an interrupted import")
            invalidateVerification()
        }
    }

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
    private fun failureReason(t: Throwable): String = describeImportFailure(t)

    /**
     * The name to stage a picked document under, with an extension it can be recognised by.
     *
     * Null when the display name carries no extension and the provider's MIME type does not imply
     * one — the caller turns that into a sentence naming the file.
     */
    private fun nameWithExtension(raw: String, uri: Uri): String? {
        val mime = runCatching { context.contentResolver.getType(uri) }.getOrNull()
        return withUsableExtension(raw, mime)
    }

    /** Validates the staged tree, then swaps it into place. */
    private fun finishImport(): Result {
        val result = installStaging(staging, root, previous)
        if (result is Result.Imported) invalidateVerification()
        return result
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

        private const val OUT_OF_SPACE_SENTENCE =
            "Not enough free space on this device to unpack the model. Free up some space and " +
                "import again — nothing was changed."

        /**
         * What a withdrawn grant looks like.
         *
         * [describeFailure] maps a `SecurityException` to "reinstall the app to restore them",
         * which is right for the app's own model directory and wrong here: an import reads a
         * document the user picked, and the Storage Access Framework can drop that grant while a
         * long copy is running (the picker's grant does not survive the provider restarting).
         * Telling a user to reinstall the app for that sends them to the wrong fix.
         */
        private const val ACCESS_LOST_SENTENCE =
            "The file manager withdrew access to the selection before the copy finished. Nothing " +
                "was changed — pick the files again and keep the file manager open until the " +
                "import completes."

        /**
         * What an archive with no graph in it is told, rather than "could not tell the graphs
         * apart": there is nothing to tell apart.
         */
        internal const val NO_GRAPHS_SENTENCE =
            "There are no .onnx files in this selection, so it contains no model graphs. Pick the " +
                ".zip that holds unet, text_encoder and vae_decoder .onnx files."

        /** Some graphs, but not the five components the pipeline needs. */
        internal const val GRAPHS_UNCLEAR_SENTENCE =
            "Could not tell the graphs apart. Include unet, text_encoder and vae_decoder .onnx " +
                "files, a clip/ folder with vocab.json and merges.txt, and model_manifest.json — " +
                "or pick a .zip that has them."

        /**
         * One sentence for a failed import, mapped rather than copied from the exception, and
         * naming the file when the failure belongs to one.
         *
         * The free-space case is checked across the cause chain because the platform wraps
         * `ENOSPC` inside an `IOException` whose own message says nothing about space; everything
         * else goes through [describeFailure], which also keeps absolute paths off the screen.
         */
        internal fun describeImportFailure(t: Throwable): String {
            val named = t as? ImportFileFailure
            val cause = if (named != null) (named.cause ?: named) else t
            val reason = when {
                isAccessLost(cause) -> ACCESS_LOST_SENTENCE
                isOutOfSpace(cause) -> OUT_OF_SPACE_SENTENCE
                else -> describeFailure(cause)
            }
            // The file name comes from the selection, so it is the user's own — never an
            // absolute path the app chose.
            return if (named == null) reason else "Could not read ${named.file} from the selection. $reason"
        }

        internal fun isOutOfSpace(t: Throwable): Boolean =
            generateSequence(t) { it.cause }.take(5).any { e ->
                val message = e.message ?: return@any false
                message.contains("ENOSPC", ignoreCase = true) ||
                    message.contains("No space left", ignoreCase = true) ||
                    message.contains("not enough space", ignoreCase = true)
            }

        /** True when the provider took the grant away, however deeply it was wrapped. */
        internal fun isAccessLost(t: Throwable): Boolean =
            generateSequence(t) { it.cause }.take(5).any { it is SecurityException }

        private const val BUFFER = 1 shl 16
        private const val MAX_IMPORT_BYTES = 4L * 1024 * 1024 * 1024

        /**
         * Ceiling for a single entry, separate from the ceiling for the whole archive.
         *
         * The total cap alone let one entry take the entire budget. 3.5 GB is chosen against the
         * documented exports rather than picked round: the largest component any supported export
         * has is the fp32 SD 1.5 UNet at 3.3 GB, so this refuses a decompression bomb while still
         * accepting every model the project documents.
         */
        private const val MAX_ENTRY_BYTES = 7L * 512 * 1024 * 1024

        /**
         * Copies at most [budget] bytes, so an oversized stream is refused instead of filling the
         * disk.
         *
         * It neither opens nor closes either stream: the caller owns both and wraps them in `use`,
         * which is what keeps a failed import from holding a descriptor on the picked document.
         */
        internal fun copyLimited(
            input: InputStream,
            output: java.io.OutputStream,
            budget: Long,
            entryBudget: Long = Long.MAX_VALUE,
            entryName: String = "This file"
        ): Long {
            val buffer = ByteArray(BUFFER)
            var written = 0L
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                written += read
                if (written > entryBudget) {
                    throw IllegalStateException(
                        "$entryName unpacks to more than ${entryBudget / (1L shl 30)} GB, which is " +
                            "larger than any single model component. Nothing was changed."
                    )
                }
                if (written > budget) throw IllegalStateException("Archive is larger than 4 GB")
                output.write(buffer, 0, read)
            }
            return written
        }

        /**
         * Where an archive entry is written, or null when it must not be written at all.
         *
         * Sanitising and the canonical-path check are one step on purpose: every zip-slip variant
         * that matters — a NUL byte, `..%2f`, a backslash path, an absolute path — has to end in
         * the same answer, and one function is what a test can assert that against.
         */
        internal fun safeTargetFor(destination: File, rawName: String): File? {
            val safe = sanitiseEntryPath(rawName) ?: return null
            val target = File(destination, safe)
            // Defence in depth: even after sanitising, refuse anything that resolves outside.
            val inside = runCatching {
                target.canonicalPath.startsWith(destination.canonicalPath + File.separator)
            }.getOrDefault(false)
            return if (inside) target else null
        }

        /**
         * [name] with a file extension, or null when neither the name nor the MIME type has one.
         *
         * The extension is the only thing that identifies a picked file as a graph, a vocabulary
         * or a merge table, so a name without one is completed from the provider's MIME type — or
         * reported, never guessed.
         */
        internal fun withUsableExtension(name: String, mimeType: String?): String? {
            if (name.substringAfterLast('/').contains('.')) return name
            val extension = extensionForMime(mimeType) ?: return null
            return name + extension
        }

        /** The extensions the importer recognises, keyed by the MIME types providers report. */
        internal fun extensionForMime(mimeType: String?): String? =
            when (mimeType?.substringBefore(';')?.trim()?.lowercase()) {
                "application/json", "text/json", "text/x-json" -> ".json"
                "text/plain" -> ".txt"
                else -> null
            }

        /**
         * Rejects absolute paths, parent traversal and empty names. Returns a path relative to the
         * model directory, or null when the entry must not be written.
         */
        internal fun sanitiseEntryPath(raw: String): String? {
            // A NUL byte truncates the name inside native code, so `unet.onnx\u0000.png` would be
            // judged by this function and opened by the file system as a different name.
            if (raw.any { it == '\u0000' }) return null
            val normalised = raw.replace('\\', '/').trim()
            if (normalised.isEmpty()) return null
            if (normalised.startsWith("/")) return null
            if (normalised.contains(":")) return null // drive letters, stream wrappers
            // A name that is still a traversal once its percent-escapes are decoded: `..%2f..%2fx`
            // is harmless as a literal file name, but it is what a tool that decodes entry names
            // before writing would turn back into `../../x`, so it is refused rather than kept.
            val decoded = decodePercentEscapes(normalised)
            if (decoded != null && decoded != normalised) {
                val decodedParts = decoded.replace('\\', '/').split('/')
                    .filter { it.isNotEmpty() && it != "." }
                if (decoded.startsWith("/") ||
                    decoded.contains(":") ||
                    decodedParts.any { it == ".." }
                ) {
                    return null
                }
            }
            val parts = normalised.split('/').filter { it.isNotEmpty() && it != "." }
            if (parts.isEmpty()) return null
            if (parts.any { it == ".." }) return null
            return parts.joinToString("/")
        }

        /**
         * Decodes `%XX` escapes, or null when the text is not valid percent-encoding.
         *
         * Null also covers `%00`: a name that decodes to a NUL is refused at the call site by
         * virtue of not being decodable at all.
         */
        private fun decodePercentEscapes(text: String): String? {
            if (!text.contains('%')) return text
            val out = StringBuilder(text.length)
            var i = 0
            while (i < text.length) {
                val c = text[i]
                if (c == '%') {
                    if (i + 2 >= text.length) return null
                    val code = text.substring(i + 1, i + 3).toIntOrNull(16) ?: return null
                    if (code == 0) return null
                    out.append(code.toChar())
                    i += 3
                } else {
                    out.append(c)
                    i++
                }
            }
            return out.toString()
        }

        /**
         * Unpacks a zip into [destination], returning the bytes written.
         *
         * Android-free, so the entry-name handling that decides what gets written to disk can be
         * tested against a real archive instead of a mocked picker.
         */
        internal fun extractZip(
            input: InputStream,
            destination: File,
            entryCap: Long = MAX_ENTRY_BYTES
        ): Long {
            var total = 0L
            val written = HashSet<String>()
            ZipInputStream(input.buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    if (entry.isDirectory) {
                        zip.closeEntry()
                        continue
                    }
                    val target = safeTargetFor(destination, entry.name)
                    if (target == null) {
                        Log.w(TAG, "skipping unsafe zip entry: ${entry.name}")
                        zip.closeEntry()
                        continue
                    }
                    // Duplicate names: the first entry wins and the rest are skipped, whatever order
                    // the archive lists them in. The old code let the last one overwrite the first,
                    // which made the installed file depend on archive order, left the overwritten
                    // bytes counted twice against the size budget, and could leave a longer first
                    // entry's tail behind the shorter second one.
                    val relative = target.relativeTo(destination).path
                    if (!written.add(relative)) {
                        Log.w(TAG, "skipping duplicate zip entry: ${entry.name}")
                        zip.closeEntry()
                        continue
                    }
                    target.parentFile?.mkdirs()
                    // Both caps are enforced while copying, not after: a single zip entry can
                    // decompress to far more than either limit, and a check that runs once the entry
                    // has been written has already filled the device by the time it fires.
                    try {
                        target.outputStream().use { output ->
                            total += copyLimited(
                                input = zip,
                                output = output,
                                budget = MAX_IMPORT_BYTES - total,
                                entryBudget = entryCap,
                                entryName = entry.name
                            )
                        }
                    } catch (t: Throwable) {
                        throw ImportFileFailure(entry.name, t)
                    }
                    zip.closeEntry()
                }
            }
            return total
        }

        /**
         * Validates the staged tree, then swaps it into place.
         *
         * Android-free on purpose — it takes a staging directory, the live directory and the one
         * kept for rollback — so the decisions it makes about what counts as a model, and what
         * counts as a broken one, can be tested against real files instead of a mocked Context.
         */
        internal fun installStaging(staging: File, root: File, previous: File): Result {
            var files = staging.walkTopDown().filter { it.isFile }.toList()
            if (files.isEmpty()) return Result.Failed("The selection contained no files.")

            // An archive made by zipping a folder — `zip -r model.zip mymodel/`, or any desktop
            // "Compress" — holds the whole model one level down. Everything below looks for the
            // graphs and the manifest at the root of the import, so it is moved up first.
            if (!stripSingleFolder(staging)) {
                return Result.Failed("Could not unpack the archive's folder.")
            }
            files = staging.walkTopDown().filter { it.isFile }.toList()

            val manifestFile = files.firstOrNull { it.name == ModelManifest.FILE_NAME }
            val manifest: ModelManifest = if (manifestFile != null) {
                runCatching { ModelManifest.parse(manifestFile.readText()) }.getOrElse {
                    return Result.Failed(
                        "The model_manifest.json in this import could not be read. " +
                            describeFailure(it)
                    )
                }
            } else {
                val paths = files.map { it.relativeTo(staging).path }
                val detected = detectComponents(paths)
                    ?: return Result.Failed(
                        // Two different failures, two different sentences: an archive with no
                        // graph in it at all is not the same mistake as one that has three of the
                        // five components, and "could not tell the graphs apart" described neither.
                        if (containsGraph(paths)) GRAPHS_UNCLEAR_SENTENCE else NO_GRAPHS_SENTENCE
                    )
                val json = buildManifestJson(detected, staging)
                    ?: return Result.Failed("Could not describe the imported files.")
                // Written into the staged tree, not just parsed. ModelLocator reads the manifest
                // from the model directory, so an import without one installed five graphs that
                // nothing could describe: the next resolve() found no manifest, provisioned the
                // bundled model over the top and the import vanished without an error.
                File(staging, ModelManifest.FILE_NAME).writeText(json)
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
            ModelSwap.install(root, staging, previous)?.let { return Result.Failed(it) }
            return Result.Imported(manifest.name, manifest.version, manifest.totalBytes)
        }

        /**
         * Moves the contents of a single top-level folder up to the root of the import.
         *
         * A model zipped as `zip -r model.zip mymodel/`, or by any desktop "Compress", arrives
         * with every file one level down. The importer looks for `unet/model.onnx` and a manifest
         * at the root, so those archives were rejected as unrecognisable even though they held
         * exactly the right files. Only a *single* common folder is stripped: an import with two
         * top-level folders is left alone, because choosing between them would be a guess.
         *
         * True when there was nothing to strip, or when the strip succeeded.
         */
        internal fun stripSingleFolder(staging: File): Boolean {
            val relatives = staging.walkTopDown()
                .filter { it.isFile }
                .map { it.relativeTo(staging).path }
                .toList()
            val folderName = singleTopFolder(relatives) ?: return true
            val folder = File(staging, folderName)
            if (!folder.isDirectory) return true
            val children = folder.listFiles() ?: return false
            for (child in children) {
                if (!child.renameTo(File(staging, child.name))) {
                    Log.w(TAG, "could not lift ${child.name} out of $folderName")
                    return false
                }
            }
            folder.delete()
            Log.i(TAG, "stripped the archive's top-level folder: $folderName")
            return true
        }

        /**
         * The one folder every file in [relatives] sits under, or null when there is nothing to
         * strip: an empty selection, a file already at the root, or two different top-level
         * folders.
         */
        internal fun singleTopFolder(relatives: List<String>): String? {
            if (relatives.isEmpty()) return null
            if (relatives.any { !it.contains('/') && !it.contains(File.separatorChar) }) return null
            val top = relatives.map { it.substringBefore('/').substringBefore(File.separatorChar) }
                .distinct()
            return top.singleOrNull()
        }

        /** True when the selection holds at least one ONNX graph. */
        internal fun containsGraph(paths: List<String>): Boolean =
            paths.any { it.lowercase().endsWith(".onnx") }

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
