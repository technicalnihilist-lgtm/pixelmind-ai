package ai.pixelmind

import ai.pixelmind.engine.writeModelFile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream

/**
 * `writeModelFile` is the provisioning helper: it is what copies a graph out of the asset pack
 * into app-private storage on first launch, and what a repair re-runs when a checksum fails.
 *
 * Its failure paths are the interesting ones. Before the ".part" staging file existed, a copy that
 * ran out of space or lost its asset stream left a truncated graph at the final path, and the next
 * launch reported "checksum mismatch" for a file the app had itself half-written. These tests pin
 * the two cases that matter: the bytes never appear at the final path until they are all there, and
 * no staging file is left behind when the copy fails.
 */
class ProvisioningTest {

    @get:Rule
    val temp = TemporaryFolder()

    private class TruncatedStream(private val payload: ByteArray, private val failAfter: Int) : InputStream() {
        private var read = 0
        override fun read(): Int {
            if (read >= failAfter) throw IOException("asset stream ended early")
            return payload[read++].toInt() and 0xFF
        }

        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (read >= failAfter) throw IOException("asset stream ended early")
            val n = minOf(len, failAfter - read, payload.size - read)
            if (n <= 0) throw IOException("asset stream ended early")
            System.arraycopy(payload, read, b, off, n)
            read += n
            return n
        }
    }

    @Test
    fun `a complete copy lands at the final path and leaves no staging file`() {
        val root = temp.newFolder("models")
        val payload = ByteArray(64 * 1024) { (it % 251).toByte() }

        val written = writeModelFile(root, "unet/model.onnx", ByteArrayInputStream(payload))

        assertEquals(File(root, "unet/model.onnx"), written)
        assertArrayEquals(payload, written.readBytes())
        assertEquals(listOf("model.onnx"), written.parentFile!!.list()!!.toList())
    }

    @Test
    fun `a nested path has its directory created`() {
        val root = temp.newFolder("models")
        writeModelFile(root, "text_encoder/model.onnx", ByteArrayInputStream(byteArrayOf(1, 2, 3)))
        assertTrue(File(root, "text_encoder/model.onnx").isFile)
    }

    @Test
    fun `a truncated asset leaves nothing behind and does not touch the existing graph`() {
        val root = temp.newFolder("models")
        val target = File(root, "unet/model.onnx")
        target.parentFile!!.mkdirs()
        val previous = "the previous, working graph".toByteArray()
        target.writeBytes(previous)

        try {
            writeModelFile(root, "unet/model.onnx", TruncatedStream(ByteArray(4096), failAfter = 1024))
            fail("a truncated asset must not be reported as provisioned")
        } catch (expected: IOException) {
            assertEquals("asset stream ended early", expected.message)
        }

        assertArrayEquals("the previous graph must survive a failed repair", previous, target.readBytes())
        assertEquals(
            "a failed copy must not leave its .part file behind",
            listOf("model.onnx"),
            target.parentFile!!.list()!!.toList()
        )
    }

    @Test
    fun `a truncated asset with no previous graph leaves no file at the final path`() {
        val root = temp.newFolder("models")

        try {
            writeModelFile(root, "vae/model.onnx", TruncatedStream(ByteArray(4096), failAfter = 10))
            fail("a truncated asset must not be reported as provisioned")
        } catch (expected: IOException) {
            // expected
        }

        assertFalse("no half-written graph may appear at the final path", File(root, "vae/model.onnx").exists())
        assertEquals(emptyList<String>(), File(root, "vae").list()!!.toList())
    }

    @Test
    fun `an unwritable model directory fails instead of pretending to provision`() {
        val root = temp.newFolder("models")
        val readOnly = File(root, "unet")
        assertTrue(readOnly.mkdirs())
        assertTrue("the test needs a directory it cannot write to", readOnly.setWritable(false))

        try {
            writeModelFile(root, "unet/model.onnx", ByteArrayInputStream(ByteArray(1024)))
            fail("writing into a read-only directory must fail")
        } catch (expected: Exception) {
            assertFalse("no graph may appear in a directory that rejected the write",
                File(readOnly, "model.onnx").exists())
            assertEquals(emptyList<String>(), readOnly.list()!!.toList())
        } finally {
            readOnly.setWritable(true)
        }
    }
}
