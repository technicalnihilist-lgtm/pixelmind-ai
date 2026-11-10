package ai.pixelmind.util

import kotlin.math.min

/**
 * Sizing and keying rules for [ImageStore]'s decode cache, kept as pure functions so they can be
 * tested and reasoned about without an Android device.
 *
 * Why the budget is not simply "a fraction of the Java heap": from API 26 (this app's `minSdk`)
 * bitmap pixels live in *native* memory, so `Runtime.maxMemory()` no longer describes the memory
 * a cache of decoded bitmaps occupies. A cache sized from the heap class therefore either
 * over-reserves on a device whose native budget is tight, or under-reserves on one with a large
 * heap class and little free RAM. The budget is derived from both figures instead, capped
 * absolutely, and halved on a device that has declared itself low-RAM.
 */
object PreviewCachePolicy {

    /** Never cache less than this: below it the gallery re-decodes on every scroll. */
    const val MIN_KB = 4 * 1024

    /** Never cache more than this, whatever the device reports. */
    const val MAX_KB = 24 * 1024

    /**
     * Cache budget in KB.
     *
     * [memoryClassMb] is `ActivityManager.memoryClass` (the Java heap class), [totalRamMb] the
     * device's physical RAM, [isLowRam] `ActivityManager.isLowRamDevice`.
     */
    fun budgetKb(memoryClassMb: Int, totalRamMb: Int, isLowRam: Boolean): Int {
        val byHeapClass = (memoryClassMb.coerceAtLeast(0) * 1024) / 8
        val byRam = (totalRamMb.coerceAtLeast(0) * 1024) / 96
        val share = min(byHeapClass, byRam)
        val scaled = if (isLowRam) share / 2 else share
        return scaled.coerceIn(MIN_KB, MAX_KB)
    }

    /**
     * The sample size `BitmapFactory` should use so that neither edge exceeds [maxEdge].
     *
     * Identical to the loop this replaced, kept in one place so a test can pin it: sample sizes
     * are powers of two, which is also why the cache key has to record the sample rather than the
     * requested edge.
     */
    fun sampleSize(outWidth: Int, outHeight: Int, maxEdge: Int): Int {
        if (outWidth <= 0 || outHeight <= 0 || maxEdge <= 0) return 1
        var sample = 1
        while (outWidth / sample > maxEdge || outHeight / sample > maxEdge) sample *= 2
        return sample
    }

    /** Below this the thumbnail is too soft to be worth the memory it saves. */
    const val MIN_EDGE = 96

    /**
     * Decode edge to actually use, given what the caller asked for.
     *
     * A 480 px grid thumbnail is 0.9 MB of ARGB_8888 pixels and a 1080 px detail view is 4.5 MB.
     * On a device that has declared itself low-RAM — 1 GB or less, which is where the gallery is
     * most likely to be evicted mid-scroll — those decodes were competing with the model's working
     * set for the same memory, so the requested edge is halved. The images are the same images at
     * the same size on screen; only the number of pixels decoded behind them changes.
     */
    fun thumbnailEdge(requested: Int, isLowRam: Boolean): Int {
        if (requested <= 0) return MIN_EDGE
        if (!isLowRam) return requested
        return (requested / 2).coerceAtLeast(MIN_EDGE)
    }

    /**
     * Cache key for one decoded size of one file.
     *
     * The path alone was the key, so the first decode of a file won for every later request: a
     * 320 px thumbnail in the gallery satisfied a 1080 px request from the detail view, which
     * then drew a visibly soft image, and vice versa. The sample size is what the bitmap actually
     * is, so it belongs in the key.
     */
    fun key(path: String, sample: Int): String = "$path@$sample"

    /** Prefix shared by every cached size of [path]; [ImageStore.delete] removes them all. */
    fun prefix(path: String): String = "$path@"
}
