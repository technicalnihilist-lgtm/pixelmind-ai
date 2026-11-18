package ai.pixelmind

import ai.pixelmind.engine.ImageChannels
import ai.pixelmind.engine.LatentScale
import ai.pixelmind.engine.LcmScheduler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.FloatBuffer
import kotlin.math.sin
import kotlin.math.tanh

/**
 * A golden image for the whole numeric chain, at 64x64 with a fixed latent.
 *
 * The chain under test is the one a real generation runs, minus ONNX Runtime: seed-free fixed
 * latent -> four LCM steps -> the 0.18215 unscale -> the decoder's three planes -> ARGB. The
 * expected pixels come from an independent Python implementation of the same steps, run
 * separately from this code, so a change in any one of them (the scheduler coefficients, the
 * scaling direction, the channel order, the rounding) moves the image and fails here.
 *
 * What this is *not*: it does not run the bundled ONNX model. `onnxruntime-android` ships arm64
 * natives only, so no graph executes on the JVM this test runs on, and a true golden image of the
 * real pipeline needs a device (see docs/VERIFICATION.md). The graphs are replaced by the
 * deterministic stand-in below — exact integer arithmetic, identical in both implementations —
 * so what is pinned is the Kotlin maths, which is the part that can regress silently.
 */
class GoldenPipelineTest {

    private val latentSide = 8
    private val channels = 4
    private val latentSize = channels * latentSide * latentSide
    private val imageSide = 64
    private val plane = imageSide * imageSide

    /** The UNet stand-in: a function of index and timestep, with a small dependence on x. */
    private fun standInEpsilon(x: FloatArray, timestep: Int): FloatArray =
        FloatArray(x.size) { i ->
            val pattern = ((i * 37 + timestep) % 101) / 101.0 - 0.5
            (0.5 * pattern + 0.1 * x[i]).toFloat()
        }

    /** The decoder stand-in: three planes of tanh-mapped values. */
    private fun standInDecode(latent: FloatArray): FloatBuffer {
        val pixels = FloatArray(ImageChannels.CHANNELS * plane)
        for (c in 0 until ImageChannels.CHANNELS) {
            for (p in 0 until plane) {
                val source = latent[(c * plane + p) % latentSize]
                pixels[c * plane + p] = tanh(0.02 * source + 0.2 * sin(p * 0.017 + c)).toFloat()
            }
        }
        return FloatBuffer.wrap(pixels)
    }

    /** The fixed starting latent: a spread of values with no RNG involved. */
    private fun initialLatent(): FloatArray =
        FloatArray(latentSize) { i -> ((i * 13 % 17) / 17.0 - 0.5).toFloat() }

    private fun render(): IntArray {
        val scheduler = LcmScheduler()
        val latent = initialLatent()
        val timesteps = scheduler.timesteps(4)
        for (i in timesteps.indices) {
            val prev = if (i + 1 < timesteps.size) timesteps[i + 1] else -1
            scheduler.step(latent, standInEpsilon(latent, timesteps[i]), timesteps[i], prev)
        }
        val unscaled = FloatArray(latentSize)
        LatentScale.fromModel(latent, unscaled)
        val argb = IntArray(plane)
        ImageChannels.packRgb(standInDecode(unscaled), base = 0, plane = plane, dest = argb)
        return argb
    }

    @Test
    fun `the rendered image matches the reference pixels`() {
        val argb = render()
        // Sampled pixels from the reference run; the full image is covered by the checksum below.
        val expected = mapOf(
            0 to 0xFF293637.toInt(),
            1 to 0xFFB2C3C4.toInt(),
            7 to 0xFF697C7B.toInt(),
            100 to 0xFF9B8D74.toInt(),
            1000 to 0xFF45495D.toInt(),
            2047 to 0xFF382D2F.toInt(),
            4095 to 0xFF465148.toInt()
        )
        for ((index, want) in expected) {
            assertPixel(want, argb[index], "pixel $index")
        }
    }

    @Test
    fun `the image's total brightness matches the reference`() {
        // A sum over all 4096 pixels: catches a shift everywhere that a sampled pixel could miss.
        val sum = render().sumOf { it.toLong() and 0xFFFFFFFFL }
        val expected = 17_556_722_089_640L
        val tolerance = plane * 2L * 3L // two levels per channel, four thousand pixels
        assertTrue(
            "image sum was $sum, reference is $expected",
            kotlin.math.abs(sum - expected) <= tolerance
        )
    }

    @Test
    fun `the image is neither blank nor saturated`() {
        // A golden test on a black rectangle would pass whatever the maths did.
        val argb = render()
        val distinct = argb.toSet().size
        assertTrue("only $distinct distinct pixels: the image is degenerate", distinct > 1000)
        assertTrue("the image is all black", argb.any { (it and 0xFFFFFF) != 0 })
        assertTrue("the image is all white", argb.any { (it and 0xFFFFFF) != 0xFFFFFF })
    }

    @Test
    fun `the chain is deterministic`() {
        assertTrue("two renders of the same latent must be identical", render().contentEquals(render()))
    }

    @Test
    fun `swapping red and blue changes the image`() {
        // Proves the golden pixels actually pin the channel order rather than the layout alone.
        val argb = render()
        val swapped = argb.map {
            val r = (it ushr 16) and 0xFF
            val b = it and 0xFF
            (it and 0xFF00FF00.toInt()) or (b shl 16) or r
        }.toIntArray()
        assertTrue("channel order is not pinned by the golden pixels", !argb.contentEquals(swapped))
    }

    private fun assertPixel(want: Int, got: Int, label: String) {
        // One level of tolerance per channel: the reference computes in doubles, this in floats.
        for (shift in intArrayOf(24, 16, 8, 0)) {
            val a = (want ushr shift) and 0xFF
            val b = (got ushr shift) and 0xFF
            assertEquals("$label channel at bit $shift", a.toDouble(), b.toDouble(), 1.0)
        }
    }
}
