package ai.pixelmind

import ai.pixelmind.util.PreviewCachePolicy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The decode cache's sizing and keying rules.
 *
 * Pinned here because both were wrong in ways no test would have caught: the budget was a
 * fraction of a heap that no longer contains bitmap pixels, and the key was the file path, so
 * one decoded size answered requests for every other size.
 */
class PreviewCachePolicyTest {

    @Test
    fun `the budget is capped absolutely whatever the device reports`() {
        assertEquals(PreviewCachePolicy.MAX_KB, PreviewCachePolicy.budgetKb(1024, 16_384, false))
        assertEquals(PreviewCachePolicy.MAX_KB, PreviewCachePolicy.budgetKb(512, 8_192, false))
    }

    @Test
    fun `a low-RAM device gets half the share`() {
        val normal = PreviewCachePolicy.budgetKb(128, 1024, false)
        val lowRam = PreviewCachePolicy.budgetKb(128, 1024, true)
        assertEquals(normal / 2, lowRam)
        assertTrue(lowRam >= PreviewCachePolicy.MIN_KB)
    }

    @Test
    fun `the budget never falls below the floor or above the cap`() {
        val tiny = PreviewCachePolicy.budgetKb(16, 128, true)
        assertEquals(PreviewCachePolicy.MIN_KB, tiny)
        for (memoryClass in listOf(32, 64, 128, 192, 256, 384, 512, 1024)) {
            for (ram in listOf(256, 512, 1024, 2048, 4096, 8192, 16_384)) {
                for (lowRam in listOf(true, false)) {
                    val kb = PreviewCachePolicy.budgetKb(memoryClass, ram, lowRam)
                    assertTrue("$memoryClass/$ram/$lowRam gave $kb", kb in PreviewCachePolicy.MIN_KB..PreviewCachePolicy.MAX_KB)
                }
            }
        }
    }

    @Test
    fun `physical RAM can hold the budget down below the heap-class share`() {
        // A 512 MB heap class on a 1 GB device must not be given 64 MB of native bitmap memory.
        val kb = PreviewCachePolicy.budgetKb(512, 1024, false)
        assertTrue("$kb KB is too much for a 1 GB device", kb <= 12 * 1024)
    }

    @Test
    fun `sample sizes are powers of two that keep both edges under the requested maximum`() {
        assertEquals(1, PreviewCachePolicy.sampleSize(512, 512, 1080))
        assertEquals(2, PreviewCachePolicy.sampleSize(512, 512, 480))
        assertEquals(2, PreviewCachePolicy.sampleSize(512, 512, 320))
        assertEquals(2, PreviewCachePolicy.sampleSize(640, 384, 320))
        assertEquals(4, PreviewCachePolicy.sampleSize(1024, 1024, 320))
        assertEquals(1, PreviewCachePolicy.sampleSize(0, 0, 320)) // undecodable file
    }

    @Test
    fun `two sizes of the same file get different keys and share one prefix`() {
        val path = "/data/user/0/ai.pixelmind/files/generations/pm_1_2.png"
        val thumb = PreviewCachePolicy.key(path, 2)
        val preview = PreviewCachePolicy.key(path, 1)
        assertTrue(thumb != preview)
        assertTrue(thumb.startsWith(PreviewCachePolicy.prefix(path)))
        assertTrue(preview.startsWith(PreviewCachePolicy.prefix(path)))
        // A different file must not be caught by the same prefix.
        assertTrue(!PreviewCachePolicy.key("$path.other", 1).startsWith(PreviewCachePolicy.prefix(path)))
    }
}
