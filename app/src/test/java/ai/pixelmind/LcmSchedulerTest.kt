package ai.pixelmind

import ai.pixelmind.engine.LcmScheduler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The LCM scheduler against an independent implementation of the reference formulas.
 *
 * The golden numbers in this file were produced by a Python implementation written from
 * `diffusers`' `LCMScheduler` (v0.27.2) — `set_timesteps`, `get_scalings_for_boundary_condition_discrete`
 * and `step` — not by running this class and pasting its output. That distinction is the whole
 * point: a test that compares the code to itself passes whatever the code does.
 *
 * What the comparison found, on 18 Nov 2026: `step` fed the boundary-condition blend (a *denoised*
 * estimate of x0) into a DDIM update that expects a *noise* estimate, so the last step — where
 * `alpha_prev == 1` — returned the noisy input latent unchanged. And `c_skip`/`c_out` were
 * evaluated at the DDPM noise level instead of the reference's `t * 10` scaling, which made
 * `c_out ~ 0.5` and halved the denoised prediction at the first step of every generation.
 */
class LcmSchedulerTest {

    private val scheduler = LcmScheduler()

    // ---- the schedule ------------------------------------------------------------------

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
    fun `the four, six and eight step schedules are the reference schedules`() {
        // k = 1000 / steps, t_i = 999 - i * k: the grid LCM was distilled on.
        assertEquals(listOf(999, 749, 499, 249), scheduler.timesteps(4).toList())
        assertEquals(listOf(999, 833, 667, 501, 335, 169), scheduler.timesteps(6).toList())
        assertEquals(listOf(999, 874, 749, 624, 499, 374, 249, 124), scheduler.timesteps(8).toList())
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
    fun `sigma matches the reference noise levels`() {
        assertEquals(14.614641229334, scheduler.sigmaFor(999), 1e-9)
        assertEquals(4.081729363724, scheduler.sigmaFor(749), 1e-9)
        assertEquals(1.612886194304, scheduler.sigmaFor(499), 1e-9)
        assertEquals(0.693204899639, scheduler.sigmaFor(249), 1e-9)
    }

    @Test
    fun `the initial noise scale matches the consistency model's`() {
        assertEquals(14.623191794616, scheduler.initialNoiseScale(999), 1e-9)
    }

    // ---- boundary conditions -----------------------------------------------------------

    @Test
    fun `boundary coefficients match the reference at the scaled timestep`() {
        // c_out ~ 1 and c_skip ~ 0: the boundary conditions are a passthrough for LCM-LoRA at
        // these timesteps, which is exactly what makes the denoised estimate equal to x0.
        assertEquals(0.999999998747, scheduler.cOut(999), 1e-9)
        assertEquals(0.999999979839, scheduler.cOut(249), 1e-9)
        assertEquals(2.505e-09, scheduler.cSkip(999), 1e-12)
        assertEquals(4.0322e-08, scheduler.cSkip(249), 1e-12)
    }

    @Test
    fun `the boundary conditions are a passthrough where LCM runs`() {
        // c_out -> 1 and c_skip -> 0 as the scaled timestep grows, which is why the denoised
        // estimate equals x0 at every timestep the schedule visits. At t = 0 they invert
        // (c_skip = 1, c_out = 0); LCM never visits 0, and the schedule test pins that.
        for (t in listOf(999, 874, 749, 624, 499, 374, 249, 124)) {
            assertTrue("c_out at t=$t", scheduler.cOut(t) > 0.9999)
            assertTrue("c_skip at t=$t", scheduler.cSkip(t) < 1e-6)
        }
        assertEquals(1.0, scheduler.cSkip(0), 0.0)
        assertEquals(0.0, scheduler.cOut(0), 0.0)
    }

    @Test
    fun `the consistency coefficients match the reference formulas`() {
        val t = 749
        val scaled = t * 10.0
        val expectedCSkip = 0.25 / (scaled * scaled + 0.25)
        val expectedCOut = scaled / Math.sqrt(scaled * scaled + 0.25)
        assertEquals(expectedCSkip, scheduler.cSkip(t), 1e-12)
        assertEquals(expectedCOut, scheduler.cOut(t), 1e-12)
    }

    // ---- the step ----------------------------------------------------------------------

    @Test
    fun `one step at 999 to 749 matches the reference`() {
        val latent = floatArrayOf(0.5f, -0.25f, 1.0f, -1.5f, 0.0f, 2.0f)
        val eps = floatArrayOf(0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f)
        scheduler.step(latent, eps, timestep = 999, prevTimestep = 749)
        val expected = doubleArrayOf(
            1.492255293239, -0.370170771216, 2.733872669543,
            -4.226127962782, -1.253189584679, 8.475400342314
        )
        for (i in expected.indices) {
            assertEquals("element $i", expected[i], latent[i].toDouble(), 1e-4)
        }
    }

    @Test
    fun `a full four, six and eight step chain matches the reference`() {
        val expected = mapOf(
            4 to doubleArrayOf(
                5.862942717645, -0.739275156061, 10.264421300116,
                -16.127364017762, -7.307320675872, 38.066412222326
            ),
            6 to doubleArrayOf(
                5.862942871407, -0.739275182170, 10.264421573792,
                -16.127364445198, -7.307320845111, 38.066413175850
            ),
            8 to doubleArrayOf(
                5.862943142416, -0.739275225052, 10.264422054062,
                -16.127365196478, -7.307321153853, 38.066414877371
            )
        )
        for ((steps, want) in expected) {
            val schedule = scheduler.timesteps(steps)
            val latent = floatArrayOf(0.5f, -0.25f, 1.0f, -1.5f, 0.0f, 2.0f)
            val eps = floatArrayOf(0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f)
            for (i in schedule.indices) {
                val prev = if (i + 1 < schedule.size) schedule[i + 1] else -1
                scheduler.step(latent, eps, schedule[i], prev)
            }
            for (i in want.indices) {
                assertEquals("$steps steps, element $i", want[i], latent[i].toDouble(), 1e-4)
            }
        }
    }

    @Test
    fun `the final step returns the denoised estimate, not the noisy latent`() {
        // With a zero model output the epsilon-parameterised x0 is x / sqrt(alpha_t), and the
        // boundary conditions leave it there. Before the fix this returned `x` itself — the last
        // step of every generation was a no-op, so the image never left the noise it started in.
        val x = 0.8f
        val latent = floatArrayOf(x)
        scheduler.step(latent, floatArrayOf(0f), timestep = 249, prevTimestep = -1)
        val alpha = Math.pow(scheduler.sqrtAlpha(249), 2.0)
        val x0 = x / Math.sqrt(alpha)
        val expected = scheduler.cOut(249) * x0 + scheduler.cSkip(249) * x
        assertEquals(expected, latent[0].toDouble(), 1e-6)
        assertTrue("the latent must change", Math.abs(latent[0] - x) > 1e-3)
        assertEquals(1.0, scheduler.sqrtAlphaPrev(-1), 0.0)
        assertEquals(0.0, scheduler.sqrtOneMinusAlphaPrev(-1), 0.0)
    }

    @Test
    fun `an intermediate step moves towards the prediction`() {
        // One step from t=999 with a zero prediction: x0 = x / sqrt(alpha), and the update must
        // land near sqrt(alpha_prev) * x0 rather than drifting away from it.
        val latent = floatArrayOf(1.0f)
        scheduler.step(latent, floatArrayOf(0f), timestep = 999, prevTimestep = 749)
        val x0 = 1.0 / scheduler.sqrtAlpha(999)
        assertEquals(scheduler.sqrtAlphaPrev(749) * x0, latent[0].toDouble(), 1e-4)
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

    @Test
    fun `a model output that is too short is refused`() {
        val latent = floatArrayOf(0.1f, 0.2f, 0.3f)
        val failure = runCatching { scheduler.step(latent, floatArrayOf(0.1f), 999, 749) }.exceptionOrNull()
        assertTrue("expected an explicit refusal, got $failure", failure is IllegalArgumentException)
    }

    @Test
    fun `every step count the app offers produces a valid schedule`() {
        for (steps in 1..12) {
            val timesteps = scheduler.timesteps(steps)
            assertEquals(steps, timesteps.size)
            assertEquals(999, timesteps.first())
            for (i in timesteps.indices) {
                assertTrue("timestep $i out of range", timesteps[i] in 0..999)
                if (i > 0) assertTrue("must strictly descend", timesteps[i] < timesteps[i - 1])
            }
        }
    }

    @Test
    fun `a zero step count does not divide by zero`() {
        assertEquals(1, scheduler.timesteps(0).size)
    }
}
