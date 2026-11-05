package ai.pixelmind

import ai.pixelmind.engine.describeFailure
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.FileNotFoundException

/**
 * The failure card shows exactly one sentence. `t.message ?: "unknown"` was what filled it, and
 * an [OutOfMemoryError] has no message on ART, so an out-of-memory kill reached the user as the
 * bare class name — true, but not something anyone can act on. These tests keep the mapping
 * honest: every branch says something, and nothing silently degrades to "unknown".
 */
class FailureDescriptionTest {

    @Test
    fun `an out of memory error explains itself instead of naming the class`() {
        val text = describeFailure(OutOfMemoryError())
        assertTrue(text.contains("memory", ignoreCase = true))
        assertFalse(text.contains("OutOfMemoryError"))
    }

    @Test
    fun `a missing model file says what to do`() {
        val text = describeFailure(FileNotFoundException("/data/user/0/ai.pixelmind/files/models/unet.onnx"))
        assertTrue(text.contains("reimport", ignoreCase = true))
        assertFalse("no filesystem path may reach the UI", text.contains("/data/user"))
    }

    @Test
    fun `an ordinary failure keeps its own message`() {
        assertEquals("VAE decoder produced no output", describeFailure(IllegalStateException("VAE decoder produced no output")))
    }

    @Test
    fun `a message-less failure falls back to the class name rather than the word unknown`() {
        assertEquals("IllegalStateException", describeFailure(IllegalStateException()))
        assertEquals("IllegalStateException", describeFailure(IllegalStateException("   ")))
    }
}
