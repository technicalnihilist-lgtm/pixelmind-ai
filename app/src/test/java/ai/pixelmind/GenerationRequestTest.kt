package ai.pixelmind

import ai.pixelmind.engine.AspectRatio
import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.QualityTier
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GenerationRequestTest {

    @Test
    fun `every aspect ratio is a multiple of 64 and divisible by 8`() {
        AspectRatio.entries.forEach { ratio ->
            assertEquals("width ${ratio.width}", 0, ratio.width % 64)
            assertEquals("height ${ratio.height}", 0, ratio.height % 64)
            assertEquals(ratio.width / 8, ratio.latentWidth)
            assertEquals(ratio.height / 8, ratio.latentHeight)
        }
    }

    @Test
    fun `explicit seeds are preserved`() {
        val request = GenerationRequest(prompt = "test", seed = 1234L)
        assertEquals(1234L, request.resolvedSeed())
    }

    @Test
    fun `random seeds are positive and stable once resolved`() {
        val request = GenerationRequest(prompt = "test")
        val seed = request.resolvedSeed()
        assertTrue(seed >= 0)
    }

    @Test
    fun `quality tiers stay inside the step range the scheduler supports`() {
        QualityTier.entries.forEach { tier ->
            assertTrue("${tier.label} steps", tier.steps in 4..12)
            assertTrue("${tier.label} guidance", tier.guidance in 1.0f..3.0f)
        }
    }
}
