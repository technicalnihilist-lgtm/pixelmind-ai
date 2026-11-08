package ai.pixelmind

import ai.pixelmind.engine.MemoryEstimate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the numbers quoted in `docs/MEASUREMENTS.md` and in the Settings-facing memory guard.
 *
 * These are *estimates* from the formula, not device measurements: the figures below are what the
 * app plans against, and the point of the test is that the plan cannot drift silently.
 */
class MemoryEstimateTest {

    private val fullWeightsMb = 640.0 // an int8 SD-1.5 UNet export
    private val tinyWeightsMb = 8.0   // the bundled verification model

    @Test
    fun `512x512 at each quality tier with full weights`() {
        // FAST 4 steps, BALANCED 6, QUALITY 8 — see QualityTier.
        assertEquals(1008, MemoryEstimate.peakMb(512, 512, 4, fullWeightsMb))
        assertEquals(1012, MemoryEstimate.peakMb(512, 512, 6, fullWeightsMb))
        assertEquals(1016, MemoryEstimate.peakMb(512, 512, 8, fullWeightsMb))
    }

    @Test
    fun `512x512 with the bundled verification model`() {
        assertEquals(121, MemoryEstimate.peakMb(512, 512, 4, tinyWeightsMb))
        assertEquals(125, MemoryEstimate.peakMb(512, 512, 6, tinyWeightsMb))
        assertEquals(129, MemoryEstimate.peakMb(512, 512, 8, tinyWeightsMb))
    }

    @Test
    fun `the non-square aspects are cheaper than 512x512`() {
        assertEquals(993, MemoryEstimate.peakMb(640, 384, 6, fullWeightsMb))
        assertEquals(1007, MemoryEstimate.peakMb(448, 576, 6, fullWeightsMb))
        assertTrue(MemoryEstimate.peakMb(384, 640, 8, fullWeightsMb) < MemoryEstimate.peakMb(512, 512, 8, fullWeightsMb))
    }

    @Test
    fun `the estimate grows with steps, weights and resolution`() {
        val base = MemoryEstimate.peakMb(512, 512, 4, fullWeightsMb)
        assertTrue(MemoryEstimate.peakMb(512, 512, 8, fullWeightsMb) > base)
        assertTrue(MemoryEstimate.peakMb(512, 512, 4, 1024.0) > base)
        assertTrue(MemoryEstimate.peakMb(640, 640, 4, fullWeightsMb) > base)
    }

    @Test
    fun `a toy graph is never assumed to be free`() {
        // The complexity term has a floor of 0.15, so a small model still budgets workspace.
        val tiny = MemoryEstimate.peakMb(512, 512, 4, 1.0)
        assertTrue("a 1 MB model still needs its activations", tiny > 60)
    }
}
