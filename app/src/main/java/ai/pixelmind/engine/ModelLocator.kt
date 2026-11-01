package ai.pixelmind.engine

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * Finds the bundled weights and provisions them into app-private storage.
 *
 * Weights ship inside the `modelpack` Play Asset Delivery *install-time* pack, so they are
 * on disk the moment the app is installed. ONNX Runtime needs a real file path, so the first
 * launch copies them once into filesDir/models. That copy is a local file operation — there
 * is no network call anywhere in this class, and none is possible: the app holds no
 * INTERNET permission.
 */
class ModelLocator(private val context: Context) {

    private val modelDir: File get() = File(context.filesDir, "models")

    /** Records which components were hashed, and when, so a relaunch does not re-read 1.2 GB. */
    private val verifiedFile: File get() = File(context.filesDir, VERIFIED_FILE)

    data class Located(val dir: File, val manifest: ModelManifest)

    sealed interface Resolution {
        data class Available(val located: Located) : Resolution
        data class Missing(val detail: String) : Resolution
        data class Corrupt(val detail: String) : Resolution
    }

    fun resolve(): Resolution {
        val manifestFile = File(modelDir, ModelManifest.FILE_NAME)
        if (!manifestFile.exists()) {
            val provisioned = provisionFromBundle(force = false)
            if (!provisioned.isNullOrBlank()) return Resolution.Missing(provisioned)
        }
        if (!manifestFile.exists()) {
            return Resolution.Missing(
                "No model weights are installed. Build the weights with tools/export_model.py " +
                    "and place them in modelpack/src/main/assets/models."
            )
        }
        var manifest = readManifest(manifestFile)
            ?: return Resolution.Corrupt("model_manifest.json is unreadable.")

        var problem = verifyComponents(manifest)
        if (problem != null) {
            // A truncated or half-copied graph used to be terminal: resolve() reported Corrupt and
            // the app refused to run until the user cleared app data, even though the asset pack
            // still holds a good copy of exactly these files. Repair once, then judge the result.
            Log.w(TAG, "Model verification failed ($problem); attempting repair from the bundle")
            val repair = repairFromBundle(manifest)
            if (repair != null) {
                return Resolution.Corrupt("$problem. $repair")
            }
            manifest = readManifest(manifestFile)
                ?: return Resolution.Corrupt("model_manifest.json is unreadable after repair.")
            problem = verifyComponents(manifest)
            if (problem != null) return Resolution.Corrupt(problem)
            Log.i(TAG, "Model files repaired from the bundled copy")
        }
        return Resolution.Available(Located(modelDir, manifest))
    }

    private fun readManifest(file: File): ModelManifest? =
        runCatching { ModelManifest.parse(file.readText()) }.getOrNull()

    /**
     * Re-provisions only when the installed tree is *the bundled set* — same component names and
     * the same checksums the asset pack declares. An imported model that fails its own checksums is
     * left exactly where it is: silently replacing a user's weights with the bundled ones would be
     * worse than telling them the file is broken.
     */
    private fun repairFromBundle(installed: ModelManifest): String? {
        val bundled = runCatching {
            context.assets.open("models/${ModelManifest.FILE_NAME}").use {
                ModelManifest.parse(it.readBytes().decodeToString())
            }
        }.getOrNull() ?: return "No bundled copy is available to repair from."

        val installedSet = installed.components.map { it.file to it.sha256.lowercase() }.sortedBy { it.first }
        val bundledSet = bundled.components.map { it.file to it.sha256.lowercase() }.sortedBy { it.first }
        if (installedSet != bundledSet) {
            return "The installed weights are not the bundled set, so they were left untouched."
        }
        return provisionFromBundle(force = true)
    }

    /** Returns null on success, or a human-readable reason on failure. */
    private fun provisionFromBundle(force: Boolean): String? {
        val manifestBytes = runCatching { context.assets.open("models/${ModelManifest.FILE_NAME}").readBytes() }
            .getOrElse {
                return "Model weights are not present in this build."
            }
        modelDir.mkdirs()
        return try {
            val manifest = ModelManifest.parse(manifestBytes.decodeToString())
            manifest.components.forEach { component ->
                val target = File(modelDir, component.file)
                // A file of the right length is assumed complete — unless we are here to repair,
                // where length proves nothing and the copy has to be redone regardless.
                if (!force && target.exists() && component.bytes > 0 && target.length() == component.bytes) {
                    return@forEach
                }
                context.assets.open("models/${component.file}").use { input ->
                    writeModelFile(modelDir, component.file, input)
                }
                Log.i(TAG, "Provisioned ${component.file} (${component.bytes} bytes)")
            }
            File(modelDir, ModelManifest.FILE_NAME).writeBytes(manifestBytes)
            // The tree changed under us; any recorded hashes are now stale.
            runCatching { verifiedFile.delete() }
            null
        } catch (t: Throwable) {
            Log.e(TAG, "Provisioning failed", t)
            "Could not prepare model files: ${t.message ?: t.javaClass.simpleName}"
        }
    }

    /**
     * Checks every component against the manifest. Returns null when the tree is intact, or a
     * message naming the first problem.
     *
     * Hashing the full weight set costs seconds on a 1 GB model and used to happen on every engine
     * load — including every return from background, because the sessions are released on stop. The
     * digests are cached against (length, last-modified) instead, so only files that actually
     * changed are re-read. Any mismatch, truncation or rewrite invalidates the entry.
     */
    private fun verifyComponents(manifest: ModelManifest): String? {
        val cache = loadVerified()
        var dirty = false
        for (component in manifest.components) {
            val file = File(modelDir, component.file)
            if (!file.exists()) return "Missing component: ${component.file}"
            if (component.sha256.isBlank()) continue

            val length = file.length()
            val modified = file.lastModified()
            val cached = cache[component.file]
            if (cached != null &&
                cached.bytes == length &&
                cached.modified == modified &&
                cached.sha256.equals(component.sha256, ignoreCase = true)
            ) {
                continue
            }

            val actual = sha256(file)
            if (!actual.equals(component.sha256, ignoreCase = true)) {
                cache.remove(component.file)
                saveVerified(cache)
                return "Checksum mismatch for ${component.file}"
            }
            cache[component.file] = Verified(component.file, length, modified, component.sha256)
            dirty = true
        }
        if (dirty) saveVerified(cache)
        return null
    }

    private data class Verified(val file: String, val bytes: Long, val modified: Long, val sha256: String)

    private fun loadVerified(): MutableMap<String, Verified> {
        val out = HashMap<String, Verified>()
        runCatching {
            val root = JSONObject(verifiedFile.readText())
            root.keys().forEach { key ->
                val entry = root.getJSONObject(key)
                out[key] = Verified(
                    file = entry.optString("file", key),
                    bytes = entry.optLong("bytes", -1L),
                    modified = entry.optLong("modified", -1L),
                    sha256 = entry.optString("sha256", "")
                )
            }
        }
        return out
    }

    private fun saveVerified(entries: Map<String, Verified>) {
        runCatching {
            val root = JSONObject()
            entries.forEach { (key, value) ->
                root.put(
                    key,
                    JSONObject()
                        .put("file", value.file)
                        .put("bytes", value.bytes)
                        .put("modified", value.modified)
                        .put("sha256", value.sha256)
                )
            }
            verifiedFile.writeText(root.toString())
        }
    }

    fun storageBytes(): Long =
        modelDir.listFiles()?.sumOf { it.length() } ?: 0L

    fun deleteProvisionedWeights(): Boolean {
        runCatching { verifiedFile.delete() }
        return modelDir.deleteRecursively()
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1 shl 16)
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    internal companion object {
        const val TAG = "ModelLocator"

        /** Sidecar holding the (length, mtime, digest) of every component already verified. */
        const val VERIFIED_FILE = "models.verified.json"
    }
}

/**
 * Copies one asset into the model directory.
 *
 * Manifest entries are relative paths ("clip/vocab.json"), so the containing directory has to be
 * created before the stream is opened — omitting that is what made provisioning fail with ENOENT
 * on a real device. The write goes through a ".part" file and is renamed into place, so an
 * interrupted copy can never leave a truncated graph that later fails its checksum.
 */
internal fun writeModelFile(modelDir: File, relativePath: String, input: java.io.InputStream): File {
    val target = File(modelDir, relativePath)
    target.parentFile?.mkdirs()
    val staging = File(target.parentFile, "${target.name}.part")
    try {
        staging.outputStream().use { output -> input.copyTo(output, DEFAULT_BUFFER_SIZE * 8) }
        if (target.exists() && !target.delete()) {
            throw IllegalStateException("could not replace $relativePath")
        }
        if (!staging.renameTo(target)) {
            staging.copyTo(target, overwrite = true)
            staging.delete()
        }
    } catch (t: Throwable) {
        // A failed copy used to leave its ".part" behind in filesDir forever.
        runCatching { staging.delete() }
        throw t
    }
    return target
}
