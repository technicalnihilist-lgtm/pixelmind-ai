package ai.pixelmind.engine

import android.util.Log

/**
 * Optional NEON-accelerated scheduler step, compiled from app/src/main/cpp.
 *
 * The native library is an *optimisation*, never a requirement: when the .so is absent
 * (a build without the NDK) the identical Kotlin implementation runs. Both paths are fed the
 * same [LcmScheduler.StepScalars], so the results agree bit-for-bit and a device that falls
 * back does not produce different art. No `.so` is built or shipped today — `System.loadLibrary`
 * fails and every step takes the Kotlin path — so the native step is compile-checked only.
 */
object NativeScheduler {

    private const val TAG = "NativeScheduler"

    private val loaded: Boolean by lazy {
        runCatching {
            System.loadLibrary("pixelmind")
            true
        }.getOrElse {
            Log.i(TAG, "Native scheduler unavailable, using Kotlin path")
            false
        }
    }

    val isAvailable: Boolean get() = loaded

    private external fun lcmStepNative(
        latent: FloatArray,
        modelOutput: FloatArray,
        cSkip: Float,
        cOut: Float,
        sqrtAlpha: Float,
        sqrtOneMinusAlpha: Float,
        sqrtOneMinusAlphaPrev: Float,
        sqrtAlphaPrev: Float
    )

    fun step(
        latent: FloatArray,
        modelOutput: FloatArray,
        timestep: Int,
        prevTimestep: Int,
        scheduler: LcmScheduler
    ) {
        if (loaded) {
            // One scalar bundle, so the native and Kotlin paths cannot drift apart: both read the
            // same six numbers from the same place.
            val scalars = scheduler.scalarsFor(timestep, prevTimestep)
            runCatching {
                lcmStepNative(
                    latent,
                    modelOutput,
                    scalars.cSkip.toFloat(),
                    scalars.cOut.toFloat(),
                    scalars.sqrtAlpha.toFloat(),
                    scalars.sqrtOneMinusAlpha.toFloat(),
                    scalars.sqrtOneMinusAlphaPrev.toFloat(),
                    scalars.sqrtAlphaPrev.toFloat()
                )
            }.onSuccess { return }.onFailure { Log.w(TAG, "Native step failed, falling back", it) }
        }
        scheduler.step(latent, modelOutput, timestep, prevTimestep)
    }
}
