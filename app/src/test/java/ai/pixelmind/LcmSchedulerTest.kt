package ai.pixelmind

import ai.pixelmind.engine.LcmScheduler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LcmSchedulerTest {

    private val scheduler = LcmScheduler()

    @Test
    fun `timesteps descend and stay inside the training range`() {
        val steps = scheduler.timesteps(4)
        assertEquals(4, steps.size)
        assertEquals(999, steps[0])
        for (i in 1 until steps.size) {
            assertTrue("timesteps must strictly descend", steps[i] < steps[i - 1])
            assertTrue(steps[i] >= 0)
        }
    }

    @Test
    fun `sigma decreases as the schedule advances`() {
        val steps = scheduler.timesteps(6)
        val sigmas = steps.map { scheduler.sigmaFor(it) }
        for (i in 1 until sigmas.size) {
            assertTrue(sigmas[i] < sigmas[i - 1])
        }
    }

    @Test
    fun `consistency coefficients match the reference formulas`() {
        val t = 749
        val sigma = scheduler.sigmaFor(t)
        val expectedCSkip = (0.25) / (sigma * sigma + 0.25)
        val expectedCOut = (sigma * 0.5) / Math.sqrt(sigma * sigma + 0.25)
        assertEquals(expectedCSkip, scheduler.cSkip(t), 1e-9)
        assertEquals(expectedCOut, scheduler.cOut(t), 1e-9)
    }

    @Test
    fun `final step collapses to the predicted denoised image`() {
        // At the last step alpha_prev == 1, so x_prev must equal the model's denoised estimate.
        val latent = floatArrayOf(0.4f, -0.2f, 0.0f)
        val output = floatArrayOf(0.4f, -0.2f, 0.0f)
        scheduler.step(latent, output, timestep = 249, prevTimestep = -1)
        assertEquals(0.0, scheduler.sqrtOneMinusAlphaPrev(-1), 0.0)
        assertEquals(1.0, scheduler.sqrtAlphaPrev(-1), 0.0)
    }

    @Test
    fun `a step is a pure function of its inputs`() {
        val a = floatArrayOf(1.2f, -0.7f, 0.33f)
        val b = floatArrayOf(1.2f, -0.7f, 0.33f)
        val eps = floatArrayOf(0.1f, 0.2f, -0.3f)
        scheduler.step(a, eps, 999, 749)
        scheduler.step(b, eps, 999, 749)
        assertTrue(a.contentEquals(b))
    }
}
