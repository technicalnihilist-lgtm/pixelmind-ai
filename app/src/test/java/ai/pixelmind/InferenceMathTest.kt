package ai.pixelmind

import ai.pixelmind.engine.AspectRatio
import ai.pixelmind.engine.ClassifierFreeGuidance
import ai.pixelmind.engine.ImageChannels
import ai.pixelmind.engine.LatentNoise
import ai.pixelmind.engine.LatentScale
import ai.pixelmind.engine.LcmScheduler
import ai.pixelmind.engine.MemoryGuard
import ai.pixelmind.engine.QualityTier
import ai.pixelmind.engine.TimestepInput
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.FloatBuffer
import java.util.Random

/**
 * The arithmetic between the graphs: guidance, the VAE scaling, the seeded noise, the decoder's
 * channel order and the timestep tensor.
 *
 * None of this needs a session, which is why it can be checked here at all — the engine that
 * calls it needs ONNX Runtime's Android natives, and those do not load on a JVM.
 */
class InferenceMathTest {

    // ---- classifier-free guidance ------------------------------------------------------

    @Test
    fun `guidance blends unconditional plus scale times the difference`() {
        val cond = floatArrayOf(2f, 4f)
        val uncond = floatArrayOf(1f, 1f)
        ClassifierFreeGuidance.blend(cond, uncond, scale = 1.5f)
        assertEquals(2.5f, cond[0], 1e-6f) // 1 + 1.5 * (2 - 1)
        assertEquals(5.5f, cond[1], 1e-6f) // 1 + 1.5 * (4 - 1)
    }

    @Test
    fun `a guidance scale of one returns the conditional prediction untouched`() {
        val cond = floatArrayOf(0.3f, -0.7f, 1.9f)
        val uncond = floatArrayOf(-2f, 3f, 0f)
        val original = cond.copyOf()
        ClassifierFreeGuidance.blend(cond, uncond, scale = 1f)
        for (i in cond.indices) assertEquals(original[i], cond[i], 1e-6f)
    }

    @Test
    fun `guidance can write into a separate buffer`() {
        val cond = floatArrayOf(1f, 2f)
        val uncond = floatArrayOf(0f, 0f)
        val dest = FloatArray(2)
        ClassifierFreeGuidance.blend(cond, uncond, scale = 2f, dest = dest)
        assertEquals(2f, dest[0], 1e-6f)
        assertEquals(4f, dest[1], 1e-6f)
        assertEquals("the source must be left alone", 1f, cond[0], 1e-6f)
    }

    @Test
    fun `mismatched guidance buffers are refused rather than read past the end`() {
        val failure = runCatching {
            ClassifierFreeGuidance.blend(floatArrayOf(1f, 2f), floatArrayOf(1f), scale = 1.5f)
        }.exceptionOrNull()
        assertTrue("expected a refusal, got $failure", failure is IllegalArgumentException)
    }

    // ---- the VAE scaling ---------------------------------------------------------------

    @Test
    fun `the latent scale is the SD-1_5 constant`() {
        assertEquals(0.18215f, LatentScale.FACTOR, 1e-9f)
        assertEquals(LcmScheduler.LATENT_SCALE, LatentScale.FACTOR, 0f)
    }

    @Test
    fun `scaling up and back down returns the original latent`() {
        val latent = floatArrayOf(0.5f, -1.25f, 3f, 0f)
        val scaled = FloatArray(4)
        val roundTrip = FloatArray(4)
        LatentScale.toModel(latent, scaled)
        LatentScale.fromModel(scaled, roundTrip)
        for (i in latent.indices) assertEquals(latent[i], roundTrip[i], 1e-6f)
    }

    @Test
    fun `the scaling is applied exactly once in each direction`() {
        // Applied twice the image comes out washed out; not at all, saturated. Both are silent,
        // so the factor is asserted rather than trusted: one multiply, one divide.
        val latent = floatArrayOf(2f)
        val scaled = FloatArray(1)
        LatentScale.toModel(latent, scaled)
        assertEquals(2f * 0.18215f, scaled[0], 1e-7f)
        assertNotEquals("twice would be a different number", 2f * 0.18215f * 0.18215f, scaled[0], 1e-7f)
        val back = FloatArray(1)
        LatentScale.fromModel(scaled, back)
        assertEquals(2f, back[0], 1e-6f)
    }

    // ---- seeded noise ------------------------------------------------------------------

    @Test
    fun `the same seed produces the same noise twice`() {
        val a = FloatArray(64)
        val b = FloatArray(64)
        LatentNoise.fill(a, seed = 123456789L, scale = 2f)
        LatentNoise.fill(b, seed = 123456789L, scale = 2f)
        assertTrue("seeded noise must be reproducible", a.contentEquals(b))
    }

    @Test
    fun `the noise is the seeded gaussian stream scaled by the initial sigma`() {
        val expected = FloatArray(8)
        val rng = Random(42L)
        for (i in expected.indices) expected[i] = rng.nextGaussian().toFloat() * 1.5f
        val actual = FloatArray(8)
        LatentNoise.fill(actual, seed = 42L, scale = 1.5f)
        assertTrue(actual.contentEquals(expected))
    }

    @Test
    fun `different seeds produce different noise`() {
        val a = FloatArray(64)
        val b = FloatArray(64)
        LatentNoise.fill(a, 1L, 1f)
        LatentNoise.fill(b, 2L, 1f)
        assertTrue("two seeds must not produce the same image", !a.contentEquals(b))
    }

    // ---- decoder channel order ---------------------------------------------------------

    @Test
    fun `plane zero is red, plane one green, plane two blue`() {
        // A pure-red pixel: plane 0 at 1.0, the other two at -1.0. Read as BGR this comes out
        // blue, which looks like a colour-grade problem rather than a bug.
        val plane = 2
        val buffer = FloatBuffer.wrap(
            floatArrayOf(
                1f, 1f, // red plane
                -1f, -1f, // green plane
                -1f, -1f // blue plane
            )
        )
        val argb = IntArray(plane)
        ImageChannels.packRgb(buffer, base = 0, plane = plane, dest = argb)
        assertEquals(0xFFFF0000.toInt(), argb[0])
        assertEquals(0xFFFF0000.toInt(), argb[1])
    }

    @Test
    fun `the alpha channel is opaque and channels are packed in the ARGB order`() {
        val buffer = FloatBuffer.wrap(floatArrayOf(0f, 0f, 0f)) // mid grey: 0.5 -> 128
        val argb = IntArray(1)
        ImageChannels.packRgb(buffer, base = 0, plane = 1, dest = argb)
        assertEquals(0xFF, (argb[0] ushr 24) and 0xFF)
        assertEquals(128, (argb[0] ushr 16) and 0xFF)
        assertEquals(128, (argb[0] ushr 8) and 0xFF)
        assertEquals(128, argb[0] and 0xFF)
    }

    @Test
    fun `decoder values are clamped and rounded, not wrapped`() {
        assertEquals(0, ImageChannels.toByte(-9f))
        assertEquals(255, ImageChannels.toByte(9f))
        assertEquals(0, ImageChannels.toByte(-1f))
        assertEquals(255, ImageChannels.toByte(1f))
        assertEquals(128, ImageChannels.toByte(0f))
    }

    @Test
    fun `a short decoder buffer is refused`() {
        val buffer = FloatBuffer.wrap(FloatArray(5))
        val failure = runCatching {
            ImageChannels.packRgb(buffer, base = 0, plane = 2, dest = IntArray(2))
        }.exceptionOrNull()
        assertTrue("expected a refusal, got $failure", failure is IllegalArgumentException)
    }

    // ---- the timestep tensor -----------------------------------------------------------

    @Test
    fun `the timestep is one int64 value`() {
        assertEquals(1, TimestepInput.SHAPE.size)
        assertEquals(1L, TimestepInput.SHAPE[0])
        assertEquals("INT64", TimestepInput.DTYPE)
        val values = TimestepInput.values(999)
        assertEquals(1, values.size)
        assertEquals(999L, values[0])
        // A LongArray is int64 on the wire; an IntArray would be int32 and the graph would reject it.
        assertEquals(999L, values[0])
    }

    // ---- aspect ratios -----------------------------------------------------------------

    @Test
    fun `every aspect ratio lands on the latent grid`() {
        for (aspect in AspectRatio.entries) {
            assertEquals("${aspect.label} width", 0, aspect.width % 8)
            assertEquals("${aspect.label} height", 0, aspect.height % 8)
            assertEquals("${aspect.label} latent width", aspect.width / 8, aspect.latentWidth)
            assertEquals("${aspect.label} latent height", aspect.height / 8, aspect.latentHeight)
        }
    }

    @Test
    fun `every aspect ratio is a multiple of 64 on both axes`() {
        // The enum documents this; the VAE only needs 8, but 64 keeps every resolution on the
        // same grid the model was exported at, so the tensor contract holds for all of them.
        for (aspect in AspectRatio.entries) {
            assertEquals("${aspect.label} width", 0, aspect.width % 64)
            assertEquals("${aspect.label} height", 0, aspect.height % 64)
        }
    }

    @Test
    fun `the decoder's expected output size matches the latent grid`() {
        for (aspect in AspectRatio.entries) {
            val plane = aspect.width * aspect.height
            val latentValues = aspect.latentWidth * aspect.latentHeight * 4
            assertEquals("${aspect.label} plane", plane, aspect.latentWidth * 8 * (aspect.latentHeight * 8))
            assertEquals("${aspect.label} decoder output", plane * 3, plane * ImageChannels.CHANNELS)
            assertTrue("${aspect.label} latent", latentValues > 0)
        }
    }

    @Test
    fun `a scaled-down resolution still lands on the latent grid`() {
        // MemoryGuard shrinks a request that does not fit; the shrink must stay a multiple of 8
        // or the decoder is handed a latent it cannot reproduce at the requested size.
        for (value in listOf(256, 384, 448, 512, 576, 640, 768, 1000, 100)) {
            val clamped = MemoryGuard.clampResolution(value)
            assertEquals("clampResolution($value)", 0, clamped % 8)
            assertTrue("clampResolution($value) = $clamped", clamped in MemoryGuard.MIN_EDGE..768)
        }
    }

    @Test
    fun `every quality tier's step count has a schedule`() {
        val scheduler = LcmScheduler()
        for (tier in QualityTier.entries) {
            val schedule = scheduler.timesteps(tier.steps)
            assertEquals("${tier.label} steps", tier.steps, schedule.size)
            assertEquals(999, schedule.first())
            assertTrue("${tier.label} must end above zero", schedule.last() > 0)
        }
    }
}
