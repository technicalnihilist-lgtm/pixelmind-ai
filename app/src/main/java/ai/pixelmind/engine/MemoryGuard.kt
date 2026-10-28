package ai.pixelmind.engine

import android.app.ActivityManager
import android.content.Context
import kotlin.math.max
import kotlin.math.min

/**
 * Decides what the device can safely attempt *before* inference starts, so we never discover an
 * out-of-memory condition halfway through a diffusion loop.
 *
 * Two things this deliberately does not do:
 *
 *  * it does not use [ActivityManager.getMemoryClass] as the budget. That number is the *Java
 *    heap* cap, and ONNX Runtime allocates its tensors natively, outside it. A phone with a
 *    512 MB heap class can still run a 1.2 GB model if it has the RAM. The budget comes from
 *    the device's actual free memory instead.
 *  * it does not assume a fixed weight size. An 8 MB verification model and a 1.2 GB SD export
 *    need very different working sets, so the caller passes the real figure from the manifest.
 */
class MemoryGuard(context: Context) {

    private val activityManager =
        context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

    private val info = ActivityManager.MemoryInfo()

    /** Java heap class, shown in Settings for information only — not used as a budget. */
    val memoryClassMb: Int = activityManager.memoryClass

    val isLowRamDevice: Boolean = activityManager.isLowRamDevice

    val totalRamMb: Int
        get() = (readInfo().totalMem / MB).toInt()

    val availableRamMb: Int
        get() = (readInfo().availMem / MB).toInt()

    private fun readInfo(): ActivityManager.MemoryInfo {
        activityManager.getMemoryInfo(info)
        return info
    }

    /**
     * Estimated peak resident memory for one generation, in MB.
     *
     * [weightsMb] is the model's on-disk size, which for these graphs is also roughly what it
     * costs in RAM once loaded. The workspace terms scale with both output resolution and model
     * complexity: a toy graph has tiny activations, a 600 MB UNet has very large ones.
     */
    fun estimatePeakMb(width: Int, height: Int, steps: Int, weightsMb: Double): Int {
        val megapixels = (width.toLong() * height.toLong()) / 262_144.0 // 512x512 == 1.0
        val complexity = (weightsMb / 640.0).coerceIn(0.15, 1.0)
        val activations = megapixels * 180.0 * complexity
        val decode = megapixels * 120.0 * complexity
        val overhead = 60.0 + steps * 2
        return (weightsMb + activations + decode + overhead).toInt()
    }

    /**
     * Returns the request adjusted down until it fits, or null when even the floor configuration
     * cannot run here.
     */
    fun fitToDevice(request: GenerationRequest, weightsMb: Double): GenerationRequest? {
        val free = availableRamMb
        val total = totalRamMb
        // Leave the OS and the rest of the process room to breathe: never plan to use more than
        // 60% of what is free, and never more than half of physical RAM.
        val budgetMb = min((free * 0.60).toInt(), (total * 0.50).toInt())
            .coerceAtLeast(MIN_BUDGET_MB)

        var width = request.width
        var height = request.height
        var steps = request.steps

        repeat(8) {
            val peak = estimatePeakMb(width, height, steps, weightsMb)
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
        val peak = estimatePeakMb(width, height, steps, weightsMb)
        return if (peak <= budgetMb) {
            request.copy(width = width, height = height, steps = steps)
        } else {
            null
        }
    }

    fun describe(): String {
        val tier = when {
            totalRamMb >= 8_000 -> "High"
            totalRamMb >= 4_000 -> "Standard"
            else -> "Limited"
        }
        return "$tier · ${totalRamMb} MB RAM · ${availableRamMb} MB free · " +
            "${if (isLowRamDevice) "low-RAM device" else "full-RAM device"}"
    }

    private fun snapTo8(value: Int): Int = (value / 8) * 8

    companion object {
        private const val MB = 1_048_576.0
        const val MIN_EDGE = 256
        const val MIN_STEPS = 4
        private const val MIN_BUDGET_MB = 256
        fun clampResolution(value: Int): Int = min(768, max(MIN_EDGE, (value / 8) * 8))
    }
}
