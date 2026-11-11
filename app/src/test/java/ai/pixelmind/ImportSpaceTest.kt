package ai.pixelmind

import ai.pixelmind.engine.ImportSpace
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The free-space arithmetic run before a model import.
 *
 * A model import is the largest write the app ever does, and it needs room for two models: the
 * incoming tree is unpacked into staging while the model it replaces is kept on disk for rollback.
 * These tests pin that, and pin the sentence shown when there is not enough room — it has to name
 * both figures and say what to do, because the previous behaviour was a raw `ENOSPC` after minutes
 * of copying.
 */
class ImportSpaceTest {

    @Test
    fun `a known import size needs room for both models and the headroom`() {
        val required = ImportSpace.requiredBytes(declaredBytes = 1_200_000_000, installedBytes = 1_200_000_000)
        assertEquals(1_200_000_000L * 2 + ImportSpace.HEADROOM_BYTES, required)
    }

    @Test
    fun `an unknown size is estimated from the installed model`() {
        assertEquals(600L * 2 + ImportSpace.HEADROOM_BYTES, ImportSpace.requiredBytes(-1, 600))
        assertEquals(0L + ImportSpace.HEADROOM_BYTES, ImportSpace.requiredBytes(0, 0))
    }

    @Test
    fun `a first import on an empty device only needs the incoming model`() {
        val required = ImportSpace.requiredBytes(declaredBytes = 500_000_000, installedBytes = 0)
        assertEquals(500_000_000L + ImportSpace.HEADROOM_BYTES, required)
    }

    @Test
    fun `nothing is reported when the import fits`() {
        assertNull(ImportSpace.shortfall(required = 100, free = 100))
        assertNull(ImportSpace.shortfall(required = 100, free = 101))
    }

    @Test
    fun `the shortfall message names the requirement, the free space and what to do`() {
        val message = ImportSpace.shortfall(required = 2_147_483_648, free = 536_870_912)
        assertTrue(message != null)
        val text = message!!
        assertTrue(text, text.contains("2.00 GB"))
        assertTrue(text, text.contains("512.0 MB"))
        assertTrue(text, text.contains("Free up at least"))
        assertTrue(text, text.contains("1.50 GB"))
        assertTrue(text, text.startsWith("Not enough free space"))
    }

    @Test
    fun `the message does not leak a filesystem path`() {
        val text = ImportSpace.shortfall(required = 2048, free = 1024)!!
        assertFalse(text, text.contains("/data/"))
        assertFalse(text, text.contains("/storage/"))
    }
}
