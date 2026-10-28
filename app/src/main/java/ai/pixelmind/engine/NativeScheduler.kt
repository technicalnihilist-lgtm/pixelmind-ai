package ai.pixelmind.engine

import android.util.Log

/**
 * Optional NEON-accelerated scheduler step, compiled from app/src/main/cpp.
 *
 * The native library is an *optimisation*, never a requirement: when the .so is absent
 * (a build without the NDK) the identical Kotlin implementation runs. Both paths are fed the
 * same scalars from [LcmScheduler], so the results agree bit-for-bit and a device that falls
 * back does not produce different art.
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
            runCatching {
                lcmStepNative(
                    latent,
                    modelOutput,
                    scheduler.cSkip(timestep).toFloat(),
                    scheduler.cOut(timestep).toFloat(),
                    scheduler.sqrtOneMinusAlphaPrev(prevTimestep).toFloat(),
                    scheduler.sqrtAlphaPrev(prevTimestep).toFloat()
                )
            }.onSuccess { return }.onFailure { Log.w(TAG, "Native step failed, falling back", it) }
        }
        scheduler.step(latent, modelOutput, timestep, prevTimestep)
    }
}
