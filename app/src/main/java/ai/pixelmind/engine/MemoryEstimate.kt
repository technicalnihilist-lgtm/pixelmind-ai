package ai.pixelmind.engine

/**
 * Peak resident-memory estimate for one generation, as a pure function.
 *
 * Kept separate from [MemoryGuard] so the numbers can be asserted in a JVM test and quoted in
 * `docs/MEASUREMENTS.md` without an Android device: the guard needs an `ActivityManager` to
 * *decide* anything, but the arithmetic it decides with is just this.
 *
 * It is an estimate, not a measurement. Nothing here was measured on a device — see
 * `docs/MEASUREMENTS.md`, which says so in the same breath as the numbers.
 */
object MemoryEstimate {

    /** One 512x512 output is the reference "megapixel" of the workspace terms. */
    private const val PIXELS_PER_UNIT = 262_144.0

    /** Weights at or above this size are assumed to activate at full width. */
    private const val REFERENCE_WEIGHTS_MB = 640.0

    private const val ACTIVATION_MB_PER_UNIT = 180.0
    private const val DECODE_MB_PER_UNIT = 120.0
    private const val FIXED_OVERHEAD_MB = 60.0
    private const val PER_STEP_MB = 2.0

    /**
     * Estimated peak resident memory for one generation, in MB.
     *
     * [weightsMb] is the model's on-disk size, which for these graphs is also roughly what it
     * costs in RAM once loaded. The workspace terms scale with output resolution and with model
     * complexity: a toy graph has tiny activations, a 600 MB UNet has very large ones.
     */
    fun peakMb(width: Int, height: Int, steps: Int, weightsMb: Double): Int {
        val megapixels = (width.toLong() * height.toLong()) / PIXELS_PER_UNIT // 512x512 == 1.0
        val complexity = (weightsMb / REFERENCE_WEIGHTS_MB).coerceIn(0.15, 1.0)
        val activations = megapixels * ACTIVATION_MB_PER_UNIT * complexity
        val decode = megapixels * DECODE_MB_PER_UNIT * complexity
        val overhead = FIXED_OVERHEAD_MB + steps * PER_STEP_MB
        return (weightsMb + activations + decode + overhead).toInt()
    }
}
