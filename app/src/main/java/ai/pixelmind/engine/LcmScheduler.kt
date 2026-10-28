package ai.pixelmind.engine

import kotlin.math.sqrt

/**
 * Latent Consistency Model scheduler (Luo et al., 2023) for the SD-1.5 LCM distillation.
 *
 * LCM is what makes four-step generation viable on a phone CPU: instead of 25–50 DDIM steps
 * it jumps across the trajectory with a consistency update. The maths here is the reference
 * implementation, written against float arrays so it can be unit-tested without a GPU.
 */
class LcmScheduler(
    private val numTrainTimesteps: Int = 1000,
    private val betaStart: Double = 0.00085,
    private val betaEnd: Double = 0.012,
    private val sigmaData: Double = 0.5
) {
    private val alphasCumprod: DoubleArray = DoubleArray(numTrainTimesteps).also { arr ->
        val betas = DoubleArray(numTrainTimesteps) { i ->
            val t = i.toDouble() / (numTrainTimesteps - 1)
            val b = sqrt(betaStart) + t * (sqrt(betaEnd) - sqrt(betaStart))
            b * b // scaled_linear schedule
        }
        var cumulative = 1.0
        for (i in 0 until numTrainTimesteps) {
            cumulative *= (1.0 - betas[i])
            arr[i] = cumulative
        }
    }

    /** Evenly spaced timesteps, descending, ending above zero — LCM's `linspace` spacing. */
    fun timesteps(steps: Int): IntArray {
        val stepRatio = numTrainTimesteps / steps
        return IntArray(steps) { i -> numTrainTimesteps - 1 - i * stepRatio }
    }

    fun sigmaFor(timestep: Int): Double {
        val alpha = alphasCumprod[timestep.coerceIn(0, numTrainTimesteps - 1)]
        return sqrt((1 - alpha) / alpha)
    }

    fun initialNoiseScale(timestep: Int): Double {
        val sigma = sigmaFor(timestep)
        return sqrt(sigma * sigma + sigmaData * sigmaData)
    }

    fun cSkip(timestep: Int): Double {
        val sigma = sigmaFor(timestep)
        return (sigmaData * sigmaData) / (sigma * sigma + sigmaData * sigmaData)
    }

    fun cOut(timestep: Int): Double {
        val sigma = sigmaFor(timestep)
        return (sigma * sigmaData) / sqrt(sigma * sigma + sigmaData * sigmaData)
    }

    fun sqrtAlphaPrev(prevTimestep: Int): Double =
        if (prevTimestep < 0) 1.0 else sqrt(alphasCumprod[prevTimestep])

    fun sqrtOneMinusAlphaPrev(prevTimestep: Int): Double =
        if (prevTimestep < 0) 0.0 else sqrt(1 - alphasCumprod[prevTimestep])

    /**
     * One consistency step. [latent] and [modelOutput] are flat CHW float arrays;
     * the result is written back into [latent].
     */
    fun step(latent: FloatArray, modelOutput: FloatArray, timestep: Int, prevTimestep: Int) {
        val sigma = sigmaFor(timestep)
        val sigmaPrev = if (prevTimestep < 0) 0.0 else sigmaFor(prevTimestep)
        val alphaPrev = if (prevTimestep < 0) 1.0 else alphasCumprod[prevTimestep]

        val cSkip = cSkip(timestep)
        val cOut = cOut(timestep)
        val sqrtAlphaPrev = sqrt(alphaPrev)
        val sqrtOneMinus = sqrt(1 - alphaPrev)

        for (i in latent.indices) {
            val x = latent[i]
            val eps = cOut * modelOutput[i] + cSkip * x
            latent[i] = ((x - sqrtOneMinus * eps) / sqrtAlphaPrev).toFloat()
        }
    }

    companion object {
        /** SD-1.5 VAE scaling factor: latents live in this scaled space. */
        const val LATENT_SCALE = 0.18215f
    }
}
