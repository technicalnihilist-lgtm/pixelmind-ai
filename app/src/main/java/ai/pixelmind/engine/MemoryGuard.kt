package ai.pixelmind.engine

import android.app.ActivityManager
import android.content.Context
import kotlin.math.max
import kotlin.math.min

/**
 * Decides what the device can safely attempt *before* inference starts, so we never
 * discover an out-of-memory condition halfway through a diffusion loop.
 *
 * The estimate is deliberately conservative: UNet activations dominate, and int8 weights
 * still expand into float32 activations at runtime.
 */
class MemoryGuard(context: Context) {

    private val activityManager =
        context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    /** Total memory the process may reasonably use, in MB. */
    val memoryClassMb: Int = activityManager.memoryClass

    val isLowRamDevice: Boolean = activityManager.isLowRamDevice

    fun estimatePeakMb(width: Int, height: Int, steps: Int): Int {
        val pixels = width.toLong() * height.toLong()
        // Weights resident in RAM (int8 UNet ~ 470 MB, text encoder ~ 60 MB, VAE ~ 100 MB).
        val weights = 640
        // Latents + attention workspace + VAE decode buffer, scaled by output size.
        val activations = (pixels / 262_144.0) * 180.0
        val decode = (pixels / 262_144.0) * 120.0
        val overhead = 60 + steps * 2
        return (weights + activations + decode + overhead).toInt()
    }

    /**
     * Returns the request adjusted down until it fits. Returns null when even the floor
     * configuration cannot run on this device.
     */
    fun fitToDevice(request: GenerationRequest): GenerationRequest? {
        val budgetMb = max(384, (memoryClassMb * 0.75).toInt())
        var width = request.width
        var height = request.height
        var steps = request.steps

        repeat(6) {
            val peak = estimatePeakMb(width, height, steps)
            if (peak <= budgetMb) {
                return request.copy(width = width, height = height, steps = steps)
            }
            when {
                steps > MIN_STEPS -> steps = max(MIN_STEPS, steps - 2)
                width > MIN_EDGE || height > MIN_EDGE -> {
                    width = snapTo8(max(MIN_EDGE, (width * 0.85f).toInt()))
                    height = snapTo8(max(MIN_EDGE, (height * 0.85f).toInt()))
                }
                else -> return null
            }
        }
        val peak = estimatePeakMb(width, height, steps)
        return if (peak <= budgetMb) request.copy(width = width, height = height, steps = steps) else null
    }

    fun describe(): String {
        val tier = when {
            memoryClassMb >= 512 -> "High"
            memoryClassMb >= 256 -> "Standard"
            else -> "Limited"
        }
        return "$tier · ${memoryClassMb} MB heap · ${if (isLowRamDevice) "low-RAM device" else "full-RAM device"}"
    }

    private fun snapTo8(value: Int): Int = (value / 8) * 8

    companion object {
        const val MIN_EDGE = 256
        const val MIN_STEPS = 4
        fun clampResolution(value: Int): Int = min(768, max(MIN_EDGE, (value / 8) * 8))
    }
}
