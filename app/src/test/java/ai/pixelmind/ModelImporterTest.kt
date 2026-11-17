package ai.pixelmind

import ai.pixelmind.engine.ModelImporter
import ai.pixelmind.engine.ModelManifest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.fail
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
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

    // ---- zip-slip variants -------------------------------------------------------------------

    @Test
    fun `a NUL byte in an entry name is refused`() {
        // Native code stops reading at the NUL, so this function and the file system would
        // otherwise disagree about which file is being written.
        assertNull(ModelImporter.sanitiseEntryPath("unet/model.onnx\u0000.png"))
        assertNull(ModelImporter.sanitiseEntryPath("\u0000"))
    }

    @Test
    fun `percent-encoded traversal is refused`() {
        assertNull(ModelImporter.sanitiseEntryPath("..%2f..%2fescape.onnx"))
        assertNull(ModelImporter.sanitiseEntryPath("%2e%2e/%2e%2e/escape.onnx"))
        assertNull(ModelImporter.sanitiseEntryPath("unet/%2e%2e/%2e%2e/escape.onnx"))
        assertNull(ModelImporter.sanitiseEntryPath("..%5c..%5cescape.onnx"))
        assertNull(ModelImporter.sanitiseEntryPath("%2fetc/passwd"))
    }

    @Test
    fun `backslash traversal is refused`() {
        assertNull(ModelImporter.sanitiseEntryPath("..\\..\\escape.onnx"))
        assertNull(ModelImporter.sanitiseEntryPath("unet\\..\\..\\escape.onnx"))
    }

    @Test
    fun `an ordinary percent sign is not mistaken for an escape`() {
        assertEquals("clip/100%valid.json", ModelImporter.sanitiseEntryPath("clip/100%valid.json"))
    }

    @Test
    fun `every refused name resolves to no target at all`() {
        val destination = temp.newFolder("tree")
        listOf(
            "../escape.onnx",
            "..%2fescape.onnx",
            "..\\escape.onnx",
            "unet/model.onnx\u0000.png",
            "/etc/passwd",
            "C:\\models\\unet.onnx"
        ).forEach { name ->
            assertNull("$name must not be written", ModelImporter.safeTargetFor(destination, name))
        }
        assertEquals(
            "unet/model.onnx",
            ModelImporter.safeTargetFor(destination, "unet/model.onnx")?.relativeTo(destination)?.path
        )
    }

    // ---- archive extraction ----------------------------------------------------------------

    private fun zipOf(entries: List<Pair<String, String>>): java.io.ByteArrayInputStream {
        val bytes = java.io.ByteArrayOutputStream()
        java.util.zip.ZipOutputStream(bytes).use { zip ->
            entries.forEach { (name, payload) ->
                zip.putNextEntry(java.util.zip.ZipEntry(name))
                zip.write(payload.toByteArray())
                zip.closeEntry()
            }
        }
        return java.io.ByteArrayInputStream(bytes.toByteArray())
    }

    @Test
    fun `a zip-slip archive writes nothing outside the destination`() {
        val destination = temp.newFolder("slip")
        val escapee = File(temp.root, "escape.txt")

        ModelImporter.extractZip(
            zipOf(
                listOf(
                    "../escape.txt" to "escaped",
                    "..%2fescape.txt" to "escaped",
                    "a/../../escape.txt" to "escaped",
                    "..\\escape.txt" to "escaped",
                    "unet/model.onnx" to "weights"
                )
            ),
            destination
        )

        assertFalse("nothing may land beside the tree", escapee.exists())
        assertEquals("weights", File(destination, "unet/model.onnx").readText())
        assertFalse(File(temp.root, "a").exists())
    }


    /**
     * A stored zip built by hand, because [java.util.zip.ZipOutputStream] refuses to write two
     * entries with the same name — and duplicate names are exactly what this needs to exercise.
     */
    private fun rawZip(entries: List<Pair<String, String>>): java.io.ByteArrayInputStream {
        val out = java.io.ByteArrayOutputStream()
        val central = java.io.ByteArrayOutputStream()
        var offset = 0L

        fun ByteArrayOutputStream.le32(value: Long) {
            for (i in 0 until 4) write(((value shr (8 * i)) and 0xFF).toInt())
        }
        fun ByteArrayOutputStream.le16(value: Int) {
            for (i in 0 until 2) write((value shr (8 * i)) and 0xFF)
        }

        val crc = java.util.zip.CRC32()
        entries.forEach { (name, payload) ->
            val data = payload.toByteArray()
            val nameBytes = name.toByteArray(Charsets.UTF_8)
            crc.reset()
            crc.update(data)

            val local = java.io.ByteArrayOutputStream()
            local.le32(0x04034b50L)
            local.le16(20) // version needed
            local.le16(0) // flags
            local.le16(0) // stored
            local.le16(0) // time
            local.le16(0) // date
            local.le32(crc.value)
            local.le32(data.size.toLong())
            local.le32(data.size.toLong())
            local.le16(nameBytes.size)
            local.le16(0)
            local.write(nameBytes)
            local.write(data)
            val localBytes = local.toByteArray()
            out.write(localBytes)

            central.le32(0x02014b50L)
            central.le16(20) // version made by
            central.le16(20) // version needed
            central.le16(0) // flags
            central.le16(0) // stored
            central.le16(0) // time
            central.le16(0) // date
            central.le32(crc.value)
            central.le32(data.size.toLong())
            central.le32(data.size.toLong())
            central.le16(nameBytes.size)
            central.le16(0) // extra
            central.le16(0) // comment
            central.le16(0) // disk
            central.le32(0) // internal attrs
            central.le32(0) // external attrs
            central.le32(offset)
            central.write(nameBytes)

            offset += localBytes.size
        }

        val centralBytes = central.toByteArray()
        out.write(centralBytes)
        out.le32(0x06054b50L)
        out.le16(0)
        out.le16(0)
        out.le16(entries.size)
        out.le16(entries.size)
        out.le32(centralBytes.size.toLong())
        out.le32(offset)
        out.le16(0)
        return java.io.ByteArrayInputStream(out.toByteArray())
    }

    @Test
    fun `duplicate entries resolve to the first one, whatever the archive order`() {
        val destination = temp.newFolder("dupes")
        val written = ModelImporter.extractZip(
            rawZip(
                listOf(
                    "unet/model.onnx" to "first",
                    "unet/model.onnx" to "a much longer second copy"
                )
            ),
            destination
        )

        // The first entry wins, so the result cannot depend on how the archive was built, and the
        // skipped copy is not counted against the size budget.
        assertEquals("first", File(destination, "unet/model.onnx").readText())
        assertEquals(5L, written)
    }

    @Test
    fun `a single entry over the per-entry cap is refused by name`() {
        val destination = temp.newFolder("bomb")
        val input = zipOf(listOf("unet/model.onnx" to "x".repeat(4096)))
        try {
            ModelImporter.extractZip(input, destination, entryCap = 1024)
            fail("expected the per-entry cap to refuse the archive")
        } catch (expected: ModelImporter.ImportFileFailure) {
            assertEquals("unet/model.onnx", expected.file)
        }
    }

    // ---- one top-level folder ---------------------------------------------------------------

    @Test
    fun `a single top-level folder is identified`() {
        assertEquals(
            "mymodel",
            ModelImporter.singleTopFolder(
                listOf("mymodel/unet/model.onnx", "mymodel/clip/vocab.json")
            )
        )
    }

    @Test
    fun `a tree with a file at the root has no folder to strip`() {
        assertNull(
            ModelImporter.singleTopFolder(
                listOf("model_manifest.json", "unet/model.onnx")
            )
        )
    }

    @Test
    fun `two top-level folders are left alone`() {
        assertNull(
            ModelImporter.singleTopFolder(
                listOf("unet/model.onnx", "clip/vocab.json")
            )
        )
    }

    // ---- installing a staged tree ----------------------------------------------------------

    private fun stageModel(dir: File, marker: String = "weights") {
        listOf(
            "unet/model.onnx",
            "text_encoder/model.onnx",
            "vae_decoder/model.onnx",
            "clip/vocab.json",
            "clip/merges.txt"
        ).forEach { relative ->
            File(dir, relative).apply { parentFile?.mkdirs() }.writeText("$marker:$relative")
        }
    }

    @Test
    fun `an archive that wraps the model in one folder is installed flat`() {
        val staging = temp.newFolder("wrapped")
        stageModel(File(staging, "pixelmind-model"))
        val root = File(temp.root, "models")
        val previous = File(temp.root, "models.previous")

        val result = ModelImporter.installStaging(staging, root, previous)

        assertTrue(result.toString(), result is ModelImporter.Result.Imported)
        assertTrue(File(root, "unet/model.onnx").exists())
        assertFalse("the wrapper folder must not survive", File(root, "pixelmind-model").exists())
        // Without the generated manifest in the tree the locator finds nothing to load and
        // provisions the bundled model over the import.
        assertTrue(File(root, "model_manifest.json").exists())
    }

    @Test
    fun `an archive with no ONNX in it says exactly that`() {
        val staging = temp.newFolder("empty")
        File(staging, "README.txt").writeText("not a model")
        val root = File(temp.root, "models")
        val previous = File(temp.root, "models.previous")

        val result = ModelImporter.installStaging(staging, root, previous)

        assertEquals(
            ModelImporter.NO_GRAPHS_SENTENCE,
            (result as ModelImporter.Result.Failed).reason
        )
        assertFalse("nothing may be installed", root.exists())
    }

    @Test
    fun `an archive with some of the graphs gets the other sentence`() {
        val staging = temp.newFolder("partial")
        File(staging, "unet/model.onnx").apply { parentFile?.mkdirs() }.writeText("w")

        val result = ModelImporter.installStaging(
            staging,
            File(temp.root, "models"),
            File(temp.root, "models.previous")
        )

        assertEquals(
            ModelImporter.GRAPHS_UNCLEAR_SENTENCE,
            (result as ModelImporter.Result.Failed).reason
        )
    }

    @Test
    fun `a manifest whose checksum does not match is refused and the installed model is kept`() {
        // A model is already installed and working.
        val root = File(temp.root, "models")
        stageModel(root, marker = "installed")
        val previous = File(temp.root, "models.previous")

        // The import arrives with a manifest, then one component changes underneath it.
        val staging = temp.newFolder("tampered")
        stageModel(staging, marker = "imported")
        val detected = ModelImporter.detectComponents(
            listOf(
                "unet/model.onnx",
                "text_encoder/model.onnx",
                "vae_decoder/model.onnx",
                "clip/vocab.json",
                "clip/merges.txt"
            )
        )!!
        File(staging, ModelManifest.FILE_NAME)
            .writeText(ModelImporter.buildManifestJson(detected, staging)!!)
        File(staging, "unet/model.onnx").appendText("tampered after the manifest was written")

        val result = ModelImporter.installStaging(staging, root, previous)

        val failed = result as ModelImporter.Result.Failed
        assertTrue(failed.reason, failed.reason.contains("does not match the checksum"))
        assertEquals(
            "installed:unet/model.onnx",
            File(root, "unet/model.onnx").readText()
        )
    }

    @Test
    fun `a matching manifest is installed and the previous model kept for rollback`() {
        val root = File(temp.root, "models")
        stageModel(root, marker = "installed")
        val previous = File(temp.root, "models.previous")

        val staging = temp.newFolder("good")
        stageModel(staging, marker = "imported")
        val detected = ModelImporter.detectComponents(
            listOf(
                "unet/model.onnx",
                "text_encoder/model.onnx",
                "vae_decoder/model.onnx",
                "clip/vocab.json",
                "clip/merges.txt"
            )
        )!!
        File(staging, ModelManifest.FILE_NAME)
            .writeText(ModelImporter.buildManifestJson(detected, staging)!!)

        val result = ModelImporter.installStaging(staging, root, previous)

        assertTrue(result.toString(), result is ModelImporter.Result.Imported)
        assertEquals("imported:unet/model.onnx", File(root, "unet/model.onnx").readText())
        assertEquals("installed:unet/model.onnx", File(previous, "unet/model.onnx").readText())
    }

    // ---- names without an extension --------------------------------------------------------

    @Test
    fun `a name without an extension takes one from the MIME type`() {
        assertEquals("vocab.json", ModelImporter.withUsableExtension("vocab", "application/json"))
        assertEquals("merges.txt", ModelImporter.withUsableExtension("merges", "text/plain"))
        assertEquals("unet.onnx", ModelImporter.withUsableExtension("unet.onnx", "text/plain"))
    }

    @Test
    fun `a graph with no extension and no MIME type has no name to use`() {
        assertNull(ModelImporter.withUsableExtension("unet", "application/octet-stream"))
        assertNull(ModelImporter.withUsableExtension("unet", null))
    }

    @Test
    fun `a MIME type with parameters is still understood`() {
        assertEquals("vocab.json", ModelImporter.withUsableExtension("vocab", "application/json; charset=utf-8"))
    }

    // ---- access revoked mid-import ---------------------------------------------------------

    @Test
    fun `a revoked grant is not reported as a broken app`() {
        val text = ModelImporter.describeImportFailure(
            ModelImporter.ImportFileFailure("unet.onnx", SecurityException("Permission Denial"))
        )
        assertTrue(text, text.contains("unet.onnx"))
        assertTrue(text, text.contains("withdrew access"))
        assertFalse("reinstalling is the wrong advice here", text.contains("Reinstall"))
    }
}
