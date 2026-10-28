package ai.pixelmind.engine

import android.content.Context
import android.util.Log
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

    data class Located(val dir: File, val manifest: ModelManifest)

    sealed interface Resolution {
        data class Available(val located: Located) : Resolution
        data class Missing(val detail: String) : Resolution
        data class Corrupt(val detail: String) : Resolution
    }

    fun resolve(): Resolution {
        val manifestFile = File(modelDir, ModelManifest.FILE_NAME)
        if (!manifestFile.exists()) {
            val provisioned = provisionFromBundle()
            if (!provisioned.isNullOrBlank()) return Resolution.Missing(provisioned)
        }
        if (!manifestFile.exists()) {
            return Resolution.Missing(
                "No model weights are installed. Build the weights with tools/export_model.py " +
                    "and place them in modelpack/src/main/assets/models."
            )
        }
        val manifest = runCatching { ModelManifest.parse(manifestFile.readText()) }
            .getOrElse { return Resolution.Corrupt("model_manifest.json is unreadable: ${it.message}") }

        manifest.components.forEach { component ->
            val f = File(modelDir, component.file)
            if (!f.exists()) return Resolution.Corrupt("Missing component: ${component.file}")
            if (component.sha256.isNotBlank() && !sha256(f).equals(component.sha256, ignoreCase = true)) {
                return Resolution.Corrupt("Checksum mismatch for ${component.file}")
            }
        }
        return Resolution.Available(Located(modelDir, manifest))
    }

    /** Returns null on success, or a human-readable reason on failure. */
    private fun provisionFromBundle(): String? {
        val manifestBytes = runCatching { context.assets.open("models/${ModelManifest.FILE_NAME}").readBytes() }
            .getOrElse {
                return "Model weights are not present in this build."
            }
        modelDir.mkdirs()
        return try {
            val manifest = ModelManifest.parse(manifestBytes.decodeToString())
            manifest.components.forEach { component ->
                val target = File(modelDir, component.file)
                if (target.exists() && target.length() == component.bytes) return@forEach
                context.assets.open("models/${component.file}").use { input ->
                    target.outputStream().use { output -> input.copyTo(output, DEFAULT_BUFFER_SIZE * 8) }
                }
                Log.i(TAG, "Provisioned ${component.file} (${component.bytes} bytes)")
            }
            File(modelDir, ModelManifest.FILE_NAME).writeBytes(manifestBytes)
            null
        } catch (t: Throwable) {
            Log.e(TAG, "Provisioning failed", t)
            "Could not provision weights: ${t.message}"
        }
    }

    fun storageBytes(): Long =
        modelDir.listFiles()?.sumOf { it.length() } ?: 0L

    fun deleteProvisionedWeights(): Boolean = modelDir.deleteRecursively()

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

    private companion object {
        const val TAG = "ModelLocator"
    }
}
