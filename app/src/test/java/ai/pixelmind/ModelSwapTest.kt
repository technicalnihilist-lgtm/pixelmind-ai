package ai.pixelmind

import ai.pixelmind.engine.ModelSwap
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * The swap that installs an imported model, and the two ways it can be interrupted.
 *
 * These run against real directories rather than a mocked Context because the thing being tested
 * *is* a pair of file renames: a mock would answer the question by assertion instead of by moving
 * a file, and the interesting states are the ones that only exist on disk (a tree that is present
 * under one name and absent under another).
 */
class ModelSwapTest {

    @get:Rule
    val temp = TemporaryFolder()

    private fun tree(parent: File, marker: String): File {
        parent.mkdirs()
        File(parent, "unet").mkdirs()
        File(parent, "unet/model.onnx").writeText(marker)
        File(parent, "model_manifest.json").writeText("""{"name":"$marker"}""")
        return parent
    }

    private fun marker(dir: File): String = File(dir, "unet/model.onnx").readText()

    @Test
    fun `install moves the live tree aside and the staged tree in`() {
        val root = tree(File(temp.root, "models"), "old")
        val staging = tree(File(temp.root, "models.staging"), "new")
        val previous = File(temp.root, "models.previous")

        val failure = ModelSwap.install(root, staging, previous)

        assertEquals(null, failure)
        assertEquals("new", marker(root))
        assertEquals("old", marker(previous))
        assertFalse("staging must have been consumed by the rename", staging.exists())
    }

    @Test
    fun `install with nothing installed yet still works`() {
        val root = File(temp.root, "models")
        val staging = tree(File(temp.root, "models.staging"), "new")
        val previous = File(temp.root, "models.previous")

        assertEquals(null, ModelSwap.install(root, staging, previous))
        assertEquals("new", marker(root))
        assertFalse(previous.exists())
    }

    @Test
    fun `a swap interrupted between its two renames is undone on the next launch`() {
        // Exactly the state a killed process leaves: the live directory is gone, the user's tree
        // survives under the name kept for rollback.
        val root = File(temp.root, "models")
        val previous = tree(File(temp.root, "models.previous"), "mine")

        assertTrue(ModelSwap.recover(root, previous))
        assertEquals("mine", marker(root))
    }

    @Test
    fun `recovery does nothing when a live tree is present`() {
        val root = tree(File(temp.root, "models"), "live")
        val previous = tree(File(temp.root, "models.previous"), "older")

        assertFalse(ModelSwap.recover(root, previous))
        assertEquals("live", marker(root))
        assertEquals("older", marker(previous))
    }

    @Test
    fun `recovery does nothing when there is nothing to recover`() {
        val root = File(temp.root, "models")
        val previous = File(temp.root, "models.previous")

        assertFalse(ModelSwap.recover(root, previous))
        assertFalse(root.exists())
    }

    @Test
    fun `rollback puts the previous model back`() {
        val root = tree(File(temp.root, "models"), "old")
        val staging = tree(File(temp.root, "models.staging"), "new")
        val previous = File(temp.root, "models.previous")
        ModelSwap.install(root, staging, previous)

        assertTrue(ModelSwap.rollback(root, previous))
        assertEquals("old", marker(root))
        assertFalse(previous.exists())
    }

    @Test
    fun `rollback with nothing kept says so instead of deleting the live model`() {
        val root = tree(File(temp.root, "models"), "only")
        val previous = File(temp.root, "models.previous")

        assertFalse(ModelSwap.rollback(root, previous))
        assertEquals("only", marker(root))
    }
}
