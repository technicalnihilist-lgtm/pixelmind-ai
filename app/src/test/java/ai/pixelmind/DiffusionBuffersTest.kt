package ai.pixelmind

import ai.pixelmind.engine.DiffusionBuffers
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The per-pass working set: allocated once, reused by every step, dropped on release.
 *
 * This is the JVM half of item 30. It can prove that a pass allocates a fixed number of arrays
 * and that a released pass holds none of them — it cannot prove anything about a device's RSS,
 * which needs a phone and real weights.
 */
class DiffusionBuffersTest {

    private val latentSize = 4 * 64 * 64 // 512x512 at a 4-channel, 8x-downsampled latent

    @Test
    fun `every buffer is a distinct array of the latent size`() {
        val buffers = DiffusionBuffers(latentSize)
        try {
            val all = listOf(
                buffers.latent,
                buffers.scaled,
                buffers.conditionalEps,
                buffers.unconditionalEps,
                buffers.decodeScratch
            )
            all.forEach { assertEquals(latentSize, it.size) }
            assertEquals("each buffer must be its own array", all.size, all.toSet().size)
            assertEquals(DiffusionBuffers.ARRAYS, all.size)
        } finally {
            buffers.release()
        }
    }

    @Test
    fun `reading a buffer twice returns the same array, so steps reuse it`() {
        val buffers = DiffusionBuffers(latentSize)
        try {
            assertSame(buffers.scaled, buffers.scaled)
            assertSame(buffers.conditionalEps, buffers.conditionalEps)
            // Two "steps" write through the same references: nothing is reallocated between them.
            val first = buffers.conditionalEps
            for (step in 1..8) {
                val eps = buffers.conditionalEps
                assertSame(first, eps)
                eps[0] = step.toFloat()
            }
            assertEquals(8f, buffers.conditionalEps[0], 0f)
        } finally {
            buffers.release()
        }
    }

    @Test
    fun `the working set is five latent-sized arrays of floats`() {
        val buffers = DiffusionBuffers(latentSize)
        try {
            assertEquals(latentSize * 4 * 5, buffers.bytes)
        } finally {
            buffers.release()
        }
    }

    @Test
    fun `release drops every buffer and is idempotent`() {
        val buffers = DiffusionBuffers(latentSize)
        val latent = buffers.latent
        assertEquals(latentSize, latent.size)
        buffers.release()
        assertTrue(buffers.isReleased)
        // A second release (a `finally` racing a cancellation handler) must not throw or
        // decrement the live count twice.
        buffers.release()
        for (read in listOf<() -> Any>(
            { buffers.latent },
            { buffers.scaled },
            { buffers.conditionalEps },
            { buffers.unconditionalEps },
            { buffers.decodeScratch }
        )) {
            try {
                read()
                fail("a released pass must not hand out its buffers")
            } catch (expected: IllegalStateException) {
                assertTrue(expected.message!!.contains("released"))
            }
        }
    }

    @Test
    fun `twenty cancelled passes leave no working set alive`() {
        val before = DiffusionBuffers.liveCount()
        repeat(20) {
            val buffers = DiffusionBuffers(latentSize)
            buffers.latent[0] = 1f // touch it, as a pass would
            buffers.release()
            assertTrue("pass $it still alive after release", buffers.isReleased)
        }
        assertEquals(
            "20 create/release cycles must not accumulate working sets",
            before,
            DiffusionBuffers.liveCount()
        )
    }

    @Test
    fun `a non-positive latent size is rejected rather than allocating nothing`() {
        try {
            DiffusionBuffers(0)
            fail("a zero-sized working set should not be constructible")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message!!.contains("positive"))
        }
    }
}
