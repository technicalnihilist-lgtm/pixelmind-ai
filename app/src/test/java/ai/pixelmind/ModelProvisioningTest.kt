package ai.pixelmind

import ai.pixelmind.engine.writeModelFile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.File

/**
 * Regression test for the provisioning failure seen on a real device:
 *
 *   Could not provision weights: /data/user/0/ai.pixelmind/files/models/clip/vocab.json:
 *   open failed: ENOENT (No such file or directory)
 *
 * The manifest lists nested paths, and the containing directory was never created.
 */
class ModelProvisioningTest {

    @get:Rule
    val temp = TemporaryFolder()

    @Test
    fun `nested paths are created before the file is written`() {
        val root = temp.newFolder("models")
        val payload = """{"a": 1}""".toByteArray()

        val written = writeModelFile(root, "clip/vocab.json", ByteArrayInputStream(payload))

        assertTrue("file should exist", written.exists())
        assertEquals("clip/vocab.json", written.relativeTo(root).path.replace(File.separatorChar, '/'))
        assertEquals(payload.size.toLong(), written.length())
        assertTrue("parent directory should exist", File(root, "clip").isDirectory)
    }

    @Test
    fun `an existing file is replaced atomically`() {
        val root = temp.newFolder("models")
        val first = "one".toByteArray()
        val second = "two-two".toByteArray()

        writeModelFile(root, "clip/merges.txt", ByteArrayInputStream(first))
        val replaced = writeModelFile(root, "clip/merges.txt", ByteArrayInputStream(second))

        assertEquals(second.size.toLong(), replaced.length())
        assertEquals("two-two", replaced.readText())
        assertTrue("no staging file should be left behind", File(root, "clip/merges.txt.part").exists().not())
    }

    @Test
    fun `a single-level path still works`() {
        val root = temp.newFolder("models")
        val written = writeModelFile(root, "model_manifest.json", ByteArrayInputStream("{}".toByteArray()))
        assertTrue(written.exists())
    }
}
