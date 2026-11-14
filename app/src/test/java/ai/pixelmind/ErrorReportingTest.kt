package ai.pixelmind

import ai.pixelmind.engine.GraphFailure
import ai.pixelmind.engine.GraphName
import ai.pixelmind.engine.ModelImporter
import ai.pixelmind.engine.ModelMessages
import ai.pixelmind.engine.describeFailure
import ai.pixelmind.engine.redactPaths
import org.json.JSONException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.FileNotFoundException
import java.io.IOException

/**
 * Day 6's contract for what the user is told when something goes wrong.
 *
 * Every assertion here is about a sentence: which graph broke, which of the three "no model"
 * states the device is in, what to change after an out-of-memory failure, which file an import
 * failed on, and that none of them shows a filesystem path. Nothing in this file measures device
 * behaviour — there is no device.
 */
class ErrorReportingTest {

    // ---- 51: a failure names the graph that produced it -----------------------------------

    @Test
    fun `a graph failure names the graph`() {
        val text = describeFailure(
            GraphFailure(GraphName.UNET, "conv2d rejected the input shape", IllegalStateException())
        )
        assertTrue(text, text.contains("UNet"))
        assertTrue(text, text.contains("conv2d rejected the input shape"))
    }

    @Test
    fun `each graph reads differently`() {
        val texts = GraphName.entries.map { describeFailure(GraphFailure(it, "boom")) }
        assertEquals("three graphs, three sentences", texts.size, texts.toSet().size)
        assertTrue(texts.all { it.startsWith("The ") && it.endsWith(" failed: boom") })
        assertTrue(texts[0].contains("text encoder"))
        assertTrue(texts[2].contains("VAE decoder"))
    }

    @Test
    fun `a graph failure never repeats an absolute path from the native message`() {
        val native = IOException("/data/user/0/ai.pixelmind/files/models/unet.onnx: open failed: EACCES")
        val text = describeFailure(GraphFailure(GraphName.UNET, describeFailure(native), native))
        assertFalse(text, text.contains("/data/user"))
        assertFalse(text, text.contains("unet.onnx"))
    }

    // ---- 52: absent, unreadable and damaged are three different things --------------------

    @Test
    fun `the three model states read differently and give different advice`() {
        val all = listOf(
            ModelMessages.ABSENT,
            ModelMessages.EMPTY_MANIFEST,
            ModelMessages.BAD_MANIFEST,
            ModelMessages.BAD_MANIFEST_AFTER_REPAIR,
            ModelMessages.missingComponent("unet.onnx"),
            ModelMessages.checksumMismatch("unet.onnx")
        )
        assertEquals("no two states may share a sentence", all.size, all.toSet().size)

        assertTrue(ModelMessages.ABSENT.contains("No model weights are installed"))
        assertFalse("absence is not damage", ModelMessages.ABSENT.contains("damaged"))
        assertFalse("absence is not a read failure", ModelMessages.ABSENT.contains("could not be read"))

        assertTrue(ModelMessages.BAD_MANIFEST.contains("could not be read"))
        assertFalse("an unreadable manifest is not damage", ModelMessages.BAD_MANIFEST.contains("damaged"))

        assertTrue(ModelMessages.checksumMismatch("unet.onnx").contains("damaged"))
        assertTrue(ModelMessages.checksumMismatch("unet.onnx").contains("unet.onnx"))
        assertTrue(ModelMessages.missingComponent("vae_decoder.onnx").contains("vae_decoder.onnx"))
    }

    // ---- 53: out of memory names the setting to change ------------------------------------

    @Test
    fun `an out of memory failure names the lower setting`() {
        val text = describeFailure(OutOfMemoryError())
        assertTrue(text, text.contains("memory", ignoreCase = true))
        assertTrue("it must name the setting", text.contains("Fast"))
        assertFalse(text.contains("OutOfMemoryError"))
    }

    // ---- 57: mapped, human-readable reasons instead of a raw message ----------------------

    @Test
    fun `a raw IO message becomes a mapped reason`() {
        val text = describeFailure(IOException("/storage/emulated/0/Download/model.zip: EIO"))
        assertTrue(text, text.contains("file could not be read or written"))
        assertFalse("no path may survive", text.contains("/storage/emulated"))
    }

    @Test
    fun `a malformed manifest is reported as invalid JSON`() {
        val text = describeFailure(JSONException("Unterminated object at character 12"))
        assertTrue(text, text.contains("not valid JSON"))
    }

    @Test
    fun `a missing model file keeps its own advice`() {
        val text = describeFailure(FileNotFoundException("/data/user/0/ai.pixelmind/files/models/vae.onnx"))
        assertTrue(text, text.contains("reimport", ignoreCase = true))
        assertFalse(text, text.contains("/data/user"))
    }

    @Test
    fun `an unknown failure still says something, never the word unknown`() {
        val text = describeFailure(IllegalStateException())
        assertEquals("IllegalStateException", text)
        assertNotEquals("unknown", text)
    }

    // ---- 60: no error string leaks a filesystem path --------------------------------------

    @Test
    fun `redaction removes absolute paths and leaves ordinary text alone`() {
        assertEquals("<file>", redactPaths("/data/user/0/ai.pixelmind/files/models/unet.onnx"))
        assertEquals("<file>", redactPaths("file:///sdcard/Download/model.zip"))
        assertEquals(
            "expected 786432 values for a 512x512 image",
            redactPaths("expected 786432 values for a 512x512 image")
        )
        assertEquals("the ratio is 1/8 of the latent", redactPaths("the ratio is 1/8 of the latent"))
        assertEquals("no path here at all", redactPaths("no path here at all"))
    }

    @Test
    fun `a message that is only a path does not become an empty sentence`() {
        val text = describeFailure(IOException("/data/user/0/ai.pixelmind/files/models/x.onnx"))
        assertTrue(text.isNotBlank())
        assertFalse(text.contains("/data/user"))
    }

    // ---- 56: an import failure says which file -------------------------------------------

    @Test
    fun `an import failure names the file it happened on`() {
        val text = ModelImporter.describeImportFailure(
            ModelImporter.ImportFileFailure("unet.onnx", IOException("open failed: EACCES"))
        )
        assertTrue(text, text.contains("unet.onnx"))
        assertTrue(text, text.contains("Could not read"))
        assertFalse("no path may survive", text.contains("/data/"))
    }

    @Test
    fun `an import failure with no file still says something actionable`() {
        val text = ModelImporter.describeImportFailure(IllegalStateException("staging is empty"))
        assertEquals("staging is empty", text)
    }

    @Test
    fun `a full disk is reported as a full disk, wherever it was wrapped`() {
        val wrapped = IOException("copy failed", IOException("write failed: ENOSPC (No space left on device)"))
        val text = ModelImporter.describeImportFailure(
            ModelImporter.ImportFileFailure("vae_decoder.onnx", wrapped)
        )
        assertTrue(text, text.contains("vae_decoder.onnx"))
        assertTrue(text, text.contains("free space", ignoreCase = true))
    }
}
