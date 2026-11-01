package ai.pixelmind

import ai.pixelmind.engine.ModelImporter
import ai.pixelmind.engine.ModelManifest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * The importer decides what gets written to disk from a name an archive or a file picker handed
 * us, so its path handling is the riskiest string code in the app. It is also the only place that
 * builds a manifest, which the engine then trusts.
 */
class ModelImporterTest {

    @get:Rule
    val temp = TemporaryFolder()

    // ---- entry path sanitising -------------------------------------------------------------

    @Test
    fun `parent traversal is rejected`() {
        assertNull(ModelImporter.sanitiseEntryPath("../escape.onnx"))
        assertNull(ModelImporter.sanitiseEntryPath("unet/../../escape.onnx"))
        assertNull(ModelImporter.sanitiseEntryPath(".."))
    }

    @Test
    fun `absolute paths and drive letters are rejected`() {
        assertNull(ModelImporter.sanitiseEntryPath("/etc/passwd"))
        assertNull(ModelImporter.sanitiseEntryPath("C:\\models\\unet.onnx"))
        assertNull(ModelImporter.sanitiseEntryPath("content://media/1"))
    }

    @Test
    fun `empty names are rejected`() {
        assertNull(ModelImporter.sanitiseEntryPath(""))
        assertNull(ModelImporter.sanitiseEntryPath("   "))
        assertNull(ModelImporter.sanitiseEntryPath("./"))
        assertNull(ModelImporter.sanitiseEntryPath("/"))
    }

    @Test
    fun `ordinary nested names are kept and normalised`() {
        assertEquals("clip/vocab.json", ModelImporter.sanitiseEntryPath("clip/vocab.json"))
        assertEquals("unet/model.onnx", ModelImporter.sanitiseEntryPath("unet\\model.onnx"))
        assertEquals("unet/model.onnx", ModelImporter.sanitiseEntryPath("./unet//model.onnx"))
        assertEquals("unet/model.onnx", ModelImporter.sanitiseEntryPath("  unet/model.onnx  "))
    }

    // ---- component detection ---------------------------------------------------------------

    @Test
    fun `a hugging face layout is detected by name`() {
        val detected = ModelImporter.detectComponents(
            listOf(
                "unet/model.onnx",
                "text_encoder/model.onnx",
                "vae_decoder/model.onnx",
                "clip/vocab.json",
                "clip/merges.txt",
                "README.md"
            )
        )
        assertNotNull(detected)
        assertEquals("unet/model.onnx", detected!!.unet)
        assertEquals("text_encoder/model.onnx", detected.textEncoder)
        assertEquals("vae_decoder/model.onnx", detected.vaeDecoder)
        assertEquals("clip/vocab.json", detected.vocab)
        assertEquals("clip/merges.txt", detected.merges)
    }

    @Test
    fun `an incomplete selection is not detected`() {
        assertNull(
            ModelImporter.detectComponents(
                listOf("unet/model.onnx", "clip/vocab.json", "clip/merges.txt")
            )
        )
    }

    // ---- generated manifest ----------------------------------------------------------------

    @Test
    fun `a generated manifest describes every staged file and parses back`() {
        val root = temp.newFolder("staging")
        val files = listOf(
            "unet/model.onnx",
            "text_encoder/model.onnx",
            "vae_decoder/model.onnx",
            "clip/vocab.json",
            "clip/merges.txt"
        )
        files.forEach { relative ->
            File(root, relative).apply { parentFile?.mkdirs() }.writeText("payload:$relative")
        }
        val detected = ModelImporter.detectComponents(files)
        assertNotNull(detected)

        val json = ModelImporter.buildManifestJson(detected!!, root)
        assertNotNull(json)

        val manifest = ModelManifest.parse(json!!)
        assertEquals("unet/model.onnx", manifest.unet.file)
        assertEquals("text_encoder/model.onnx", manifest.textEncoder.file)
        assertEquals("vae_decoder/model.onnx", manifest.vaeDecoder.file)
        assertEquals("clip/vocab.json", manifest.tokenizer.vocabFile)
        assertEquals("clip/merges.txt", manifest.tokenizer.mergesFile)
        assertEquals(5, manifest.components.size)
        // Every component must carry a real digest, otherwise the import would install weights
        // that nothing ever checks.
        manifest.components.forEach { component ->
            assertEquals(64, component.sha256.length)
            assertTrue(component.bytes > 0)
        }
        assertEquals(
            files.sumOf { File(root, it).length() },
            manifest.totalBytes
        )
    }

    @Test
    fun `digests in a generated manifest match the files they describe`() {
        val root = temp.newFolder("staging2")
        val paths = listOf(
            "unet/model.onnx",
            "text_encoder/model.onnx",
            "vae_decoder/model.onnx",
            "clip/vocab.json",
            "clip/merges.txt"
        )
        paths.forEach { File(root, it).apply { parentFile?.mkdirs() }.writeText("weights:$it") }
        val detected = ModelImporter.detectComponents(paths)!!
        val manifest = ModelManifest.parse(ModelImporter.buildManifestJson(detected, root)!!)

        val unet = File(root, "unet/model.onnx")
        val component = manifest.components.first { it.file == "unet/model.onnx" }
        assertEquals(ModelImporter.sha256(unet), component.sha256)
        assertEquals(unet.length(), component.bytes)
    }
}
