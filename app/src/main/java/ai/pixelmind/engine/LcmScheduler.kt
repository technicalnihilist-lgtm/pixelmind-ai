package ai.pixelmind.engine

import kotlin.math.sqrt

/**
 * Latent Consistency Model scheduler (Luo et al., 2023) for the SD-1.5 LCM distillation.
 *
 * LCM is what makes four-step generation viable on a phone CPU: instead of 25–50 DDIM steps
 * it jumps across the trajectory with a consistency update. The maths here is the reference
 * implementation, written against float arrays so it can be unit-tested without a GPU.
 *
 * "Reference" means `diffusers`' `LCMScheduler` (v0.27.2) with the settings LCM-LoRA ships with,
 * checked against an independent implementation of those formulas — see
 * `LcmSchedulerReferenceTest`. Three deliberate departures, all documented where they happen:
 *
 *  * the multistep sampler does **not** inject fresh noise between steps (item 74 requires a
 *    seed to reproduce an image exactly, and a phone app has no business being non-deterministic);
 *  * the timestep schedule is the paper's uniform `k = 1000 / steps` grid rather than diffusers'
 *    subset of a 50-step origin schedule — both are LCM schedules, this one is the distilled one;
 *  * [initialNoiseScale] is the consistency model's `sqrt(sigma^2 + sigma_data^2)`.
 */
class LcmScheduler(
    private val numTrainTimesteps: Int = 1000,
    private val betaStart: Double = 0.00085,
    private val betaEnd: Double = 0.012,
    private val sigmaData: Double = 0.5,
    /** `timestep_scaling` in the reference: LCM-LoRA's boundary conditions are in `t * 10`. */
    private val timestepScaling: Double = 10.0
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

    /**
     * Evenly spaced timesteps, descending, ending above zero.
     *
     * `k = 1000 / steps`, `t_i = 999 - i * k` — the schedule the LCM distillation was trained on,
     * so four steps land on 999/749/499/249.
     */
    fun timesteps(steps: Int): IntArray {
        // Clamped rather than trusted: a zero here used to divide by zero inside the schedule.
        val count = steps.coerceAtLeast(1)
        val stepRatio = numTrainTimesteps / count
        return IntArray(count) { i -> numTrainTimesteps - 1 - i * stepRatio }
    }

    fun sigmaFor(timestep: Int): Double {
        val alpha = alphasCumprod[timestep.coerceIn(0, numTrainTimesteps - 1)]
        return sqrt((1 - alpha) / alpha)
    }

    fun initialNoiseScale(timestep: Int): Double {
        val sigma = sigmaFor(timestep)
        return sqrt(sigma * sigma + sigmaData * sigmaData)
    }

    /**
     * `c_skip` of the LCM boundary conditions, evaluated at the scaled timestep.
     *
     * The scaling is not cosmetic. Evaluated at the DDPM noise level instead (`sigma` reaches
     * 14.6 at t=999) these coefficients came out as `c_skip ~ 0`, `c_out ~ 0.5`, so the denoised
     * estimate handed to the update was *half* the model's prediction — a systematic
     * under-estimate of `x0` at every step. At the reference's `t * 10` scaling they are
     * `c_out ~ 1`, `c_skip ~ 0`, which is what LCM-LoRA is distilled against.
     */
    fun cSkip(timestep: Int): Double {
        val scaled = scaledTimestep(timestep)
        return (sigmaData * sigmaData) / (scaled * scaled + sigmaData * sigmaData)
    }

    /** `c_out` of the LCM boundary conditions, evaluated at the scaled timestep. */
    fun cOut(timestep: Int): Double {
        val scaled = scaledTimestep(timestep)
        return scaled / sqrt(scaled * scaled + sigmaData * sigmaData)
    }

    fun sqrtAlpha(timestep: Int): Double = sqrt(alphaAt(timestep))

    fun sqrtOneMinusAlpha(timestep: Int): Double = sqrt(1 - alphaAt(timestep))

    fun sqrtAlphaPrev(prevTimestep: Int): Double = sqrt(alphaPrev(prevTimestep))

    fun sqrtOneMinusAlphaPrev(prevTimestep: Int): Double = sqrt(1 - alphaPrev(prevTimestep))

    /** The scalars one step needs. Exposed so the native path can be fed the same numbers. */
    data class StepScalars(
        val cSkip: Double,
        val cOut: Double,
        val sqrtAlpha: Double,
        val sqrtOneMinusAlpha: Double,
        val sqrtAlphaPrev: Double,
        val sqrtOneMinusAlphaPrev: Double
    )

    fun scalarsFor(timestep: Int, prevTimestep: Int): StepScalars = StepScalars(
        cSkip = cSkip(timestep),
        cOut = cOut(timestep),
        sqrtAlpha = sqrtAlpha(timestep),
        sqrtOneMinusAlpha = sqrtOneMinusAlpha(timestep),
        sqrtAlphaPrev = sqrtAlphaPrev(prevTimestep),
        sqrtOneMinusAlphaPrev = sqrtOneMinusAlphaPrev(prevTimestep)
    )

    /**
     * One consistency step. [latent] and [modelOutput] are flat CHW float arrays; the result is
     * written back into [latent].
     *
     * [modelOutput] is the UNet's epsilon prediction — SD-1.5 is an epsilon model, and LCM-LoRA
     * does not change the parameterisation. The reference order is: recover `x0` from epsilon,
     * apply the boundary conditions, then take the deterministic DDIM step towards
     * `alpha_prev`. For an epsilon model the noise estimate that step needs *is* the model
     * output, so no second estimate is computed.
     *
     * What this used to do: feed the boundary-condition blend (a denoised `x0`, not a noise
     * prediction) into the `x_prev = (x - sqrt(1-a_prev) * eps) / sqrt(a_prev)` form, which
     * expects a noise estimate. At the last step, where `alpha_prev == 1`, that returns the
     * *input latent* — the noisy one — instead of the denoised image, so the final step of every
     * generation was a no-op.
     */
    fun step(latent: FloatArray, modelOutput: FloatArray, timestep: Int, prevTimestep: Int) {
        require(modelOutput.size >= latent.size) {
            "model output has ${modelOutput.size} values for ${latent.size} latents"
        }
        val s = scalarsFor(timestep, prevTimestep)
        val finalStep = prevTimestep < 0
        for (i in latent.indices) {
            val x = latent[i]
            val eps = modelOutput[i]
            val x0 = ((x - s.sqrtOneMinusAlpha * eps) / s.sqrtAlpha)
            val denoised = s.cOut * x0 + s.cSkip * x
            val next = if (finalStep) {
                denoised
            } else {
                s.sqrtAlphaPrev * denoised + s.sqrtOneMinusAlphaPrev * eps
            }
            latent[i] = next.toFloat()
        }
    }

    private fun scaledTimestep(timestep: Int): Double = timestep * timestepScaling

    private fun alphaAt(timestep: Int): Double =
        alphasCumprod[timestep.coerceIn(0, numTrainTimesteps - 1)]

    private fun alphaPrev(prevTimestep: Int): Double =
        if (prevTimestep < 0) 1.0 else alphasCumprod[prevTimestep.coerceIn(0, numTrainTimesteps - 1)]

    companion object {
        /** SD-1.5 VAE scaling factor: latents live in this scaled space. */
        const val LATENT_SCALE = 0.18215f
    }
}
