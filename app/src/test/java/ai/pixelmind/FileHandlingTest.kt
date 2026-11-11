package ai.pixelmind

import ai.pixelmind.engine.ModelImporter
import ai.pixelmind.util.moveAtomically
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.InputStream

/**
 * The two pieces of file handling that every write in the app goes through: the staging rename and
 * the bounded stream copy.
 *
 * Both writers — a generated PNG and a provisioned model component — used to fall back to
 * `copyTo(target)` when `renameTo` returned false. That fallback wrote straight into the visible
 * path, which is exactly the truncated-file hazard the staging file exists to prevent. And the
 * importer's copy is what holds the descriptor on the picked document, so its ownership rules are
 * pinned here too: it closes nothing it did not open, and a failure in the middle closes everything.
 */
class FileHandlingTest {

    @get:Rule
    val temp = TemporaryFolder()

    // ---- item 47: the staging rename --------------------------------------------------------

    @Test
    fun `a staged file lands under its final name with its bytes intact`() {
        val staging = temp.newFile("image.png.part").apply { writeBytes(byteArrayOf(1, 2, 3, 4)) }
        val target = File(temp.root, "image.png")

        moveAtomically(staging, target)

        assertTrue(target.exists())
        assertFalse("the staging file is gone", staging.exists())
        assertArrayEquals(byteArrayOf(1, 2, 3, 4), target.readBytes())
    }

    @Test
    fun `an existing target is replaced by the rename`() {
        val staging = temp.newFile("image.png.part").apply { writeBytes(byteArrayOf(9)) }
        val target = temp.newFile("image.png").apply { writeBytes(byteArrayOf(1, 2, 3)) }

        moveAtomically(staging, target)

        assertArrayEquals(byteArrayOf(9), target.readBytes())
    }

    @Test
    fun `a missing staging file fails and leaves the target untouched`() {
        val staging = File(temp.root, "image.png.part")
        val target = temp.newFile("image.png").apply { writeBytes(byteArrayOf(7)) }

        try {
            moveAtomically(staging, target)
            fail("expected the move to fail")
        } catch (expected: IOException) {
            assertTrue(expected.message!!.contains("image.png.part"))
        }
        assertArrayEquals(byteArrayOf(7), target.readBytes())
    }

    // ---- item 48: descriptor ownership in the importer's copy --------------------------------

    @Test
    fun `the bounded copy closes nothing it did not open`() {
        val input = TrackingStream(byteArrayOf(1, 2, 3))
        val output = ByteArrayOutputStream()

        val written = ModelImporter.copyLimited(input, output, budget = 16)

        assertEquals(3L, written)
        assertFalse("copyLimited must leave the caller's input open", input.closed)
        assertEquals(3, output.toByteArray().size)
    }

    @Test
    fun `the bounded copy stops at the budget`() {
        val input = ByteArrayInputStream(ByteArray(100))
        try {
            ModelImporter.copyLimited(input, ByteArrayOutputStream(), budget = 50)
            fail("expected the copy to refuse an oversized stream")
        } catch (expected: IllegalStateException) {
            assertTrue(expected.message!!.contains("larger than 4 GB"))
        }
    }

    @Test
    fun `two hundred imports of a file leak no descriptors`() {
        val source = temp.newFile("selection.bin").apply { writeBytes(ByteArray(4096) { it.toByte() }) }
        val target = File(temp.root, "staged.bin")

        // Warm up first: the JDK opens descriptors lazily on the first few reads, which would
        // otherwise be counted as a leak.
        repeat(5) { copyOnce(source, target) }
        val before = openDescriptors()

        repeat(200) { copyOnce(source, target) }

        val after = openDescriptors()
        assertTrue(
            "open descriptors grew from $before to $after across 200 copies",
            after <= before + 2
        )
    }

    private fun copyOnce(source: File, target: File) {
        // The importer's exact nesting: the caller owns both streams, and a failure anywhere inside
        // still closes them.
        source.inputStream().use { input ->
            target.outputStream().use { output ->
                ModelImporter.copyLimited(input, output, budget = 1L shl 20)
            }
        }
    }

    private fun openDescriptors(): Int = File("/proc/self/fd").list()?.size ?: 0

    private class TrackingStream(payload: ByteArray) : InputStream() {
        private val delegate = ByteArrayInputStream(payload)
        var closed = false
            private set

        override fun read(): Int = delegate.read()

        override fun read(b: ByteArray, off: Int, len: Int): Int = delegate.read(b, off, len)

        override fun close() {
            closed = true
            delegate.close()
        }
    }
}
