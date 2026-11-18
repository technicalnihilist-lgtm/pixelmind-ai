package ai.pixelmind.engine

import java.nio.FloatBuffer
import java.util.Random

/**
 * The pure arithmetic of one diffusion pass, lifted out of [OnnxDiffusionEngine].
 *
 * Every function here is a total function of its arguments with no session, no bitmap and no
 * Android type in sight, which is the only way any of it can be checked on a JVM without a
 * device: the engine around it needs ONNX Runtime's Android natives, these five do not.
 */

/**
 * Classifier-free guidance: `uncond + scale * (cond - uncond)`.
 *
 * At `scale == 1` this is the conditional prediction unchanged, which is why the engine skips
 * the unconditional pass entirely below 1.05 rather than paying for a UNet call whose result the
 * blend would throw away.
 */
internal object ClassifierFreeGuidance {

    /** Blends in place into [conditional], which the caller owns and reuses every step. */
    fun blend(conditional: FloatArray, unconditional: FloatArray, scale: Float, dest: FloatArray = conditional) {
        require(unconditional.size == conditional.size) {
            "unconditional prediction has ${unconditional.size} values, conditional has ${conditional.size}"
        }
        require(dest.size == conditional.size) {
            "destination has ${dest.size} values, conditional has ${conditional.size}"
        }
        for (i in conditional.indices) {
            val u = unconditional[i]
            dest[i] = u + scale * (conditional[i] - u)
        }
    }
}

/**
 * The 0.18215 VAE scaling, in one place.
 *
 * The latent the sampler works on is *unscaled*; the UNet and the decoder both want the scaled
 * space. Applied twice (or not at all) the image comes out washed out or saturated, and neither
 * failure announces itself, so both directions live here and nowhere else.
 */
internal object LatentScale {

    const val FACTOR: Float = LcmScheduler.LATENT_SCALE

    /** Unscaled latent -> the space the UNet reads. */
    fun toModel(latent: FloatArray, dest: FloatArray) {
        require(dest.size >= latent.size) {
            "destination has ${dest.size} values for ${latent.size} latents"
        }
        for (i in latent.indices) dest[i] = latent[i] * FACTOR
    }

    /** The space the decoder emits -> back to the unscaled latent the sampler works in. */
    fun fromModel(latent: FloatArray, dest: FloatArray) {
        require(dest.size >= latent.size) {
            "destination has ${dest.size} values for ${latent.size} latents"
        }
        for (i in latent.indices) dest[i] = latent[i] / FACTOR
    }
}

/**
 * The initial latent: gaussian noise, scaled to the schedule's first step.
 *
 * `java.util.Random` is specified down to the algorithm, so a seed reproduces an image exactly
 * across runs and across devices. That is the whole point of showing the seed in the UI.
 */
internal object LatentNoise {

    fun fill(latent: FloatArray, seed: Long, scale: Float) {
        val rng = Random(seed)
        for (i in latent.indices) latent[i] = rng.nextGaussian().toFloat() * scale
    }
}

/**
 * The decoder's output planes -> the ARGB pixels a PNG is written from.
 *
 * The order is the contract: the ONNX VAE decoder emits CHW with **R, G, B** planes, which is
 * what `tools/export_model.py` produces (it converts to a torch tensor straight from the
 * pipeline's RGB output and adds no flip) and what the decoder stand-in in the golden test
 * assumes. Reading the planes as BGR swaps red and blue in every image — a bug that looks like a
 * colour-grading problem rather than a bug.
 */
internal object ImageChannels {

    const val CHANNELS = 3

    /**
     * Packs three consecutive float planes into [dest], one ARGB int per pixel.
     *
     * Reads straight out of the decoder's buffer — no 3 x H x W copy — because at 640x384 that
     * copy is 3 MB allocated on the same thread that just ran the decoder.
     */
    fun packRgb(pixels: FloatBuffer, base: Int, plane: Int, dest: IntArray) {
        require(dest.size >= plane) { "destination holds ${dest.size} pixels, plane is $plane" }
        val needed = base + CHANNELS * plane
        require(pixels.limit() >= needed) {
            "decoder buffer holds ${pixels.limit()} values, ${CHANNELS} planes of $plane need $needed"
        }
        for (i in 0 until plane) {
            val r = toByte(pixels.get(base + i))
            val g = toByte(pixels.get(base + plane + i))
            val b = toByte(pixels.get(base + 2 * plane + i))
            dest[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
    }

    /** A decoder value in [-1, 1] -> 0..255, rounding rather than truncating. */
    fun toByte(value: Float): Int {
        val normalised = (value * 0.5f + 0.5f).coerceIn(0f, 1f)
        return (normalised * 255f + 0.5f).toInt()
    }
}

/**
 * The UNet's timestep input: one value, int64.
 *
 * The graph rejects anything else, and a Float tensor (the tempting mistake, since every other
 * input is float) fails inside ONNX Runtime with a type error that says nothing about which
 * input was wrong. Kept here so the shape and the dtype are stated once.
 */
internal object TimestepInput {

    /** `[1]`: the graph's timestep input is a single value per call. */
    val SHAPE: LongArray = longArrayOf(1)

    fun values(timestep: Int): LongArray = longArrayOf(timestep.toLong())

    /** The dtype the graph must declare, in ONNX Runtime's own naming. */
    const val DTYPE = "INT64"
}
