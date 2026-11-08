package ai.pixelmind.engine

import java.util.concurrent.atomic.AtomicInteger

/**
 * The float working set for one diffusion pass, allocated once and reused by every step.
 *
 * A pass needs five arrays of `latentChannels * latentH * latentW` floats: the latent being
 * denoised, the VAE-scaled copy the UNet reads, one output buffer for each of the conditional
 * and unconditional UNet passes, and the un-scaled copy the decoder reads. Allocating them per
 * step meant four fresh arrays per step (two of them full UNet outputs) — at 512x512 that is
 * 64 KB a time, several hundred kilobytes of garbage per generation on the same thread that is
 * about to run the next one.
 *
 * [release] drops every reference. The engine calls it on every exit path from a pass, including
 * the cancellation path, so a cancelled generation does not leave a latent-sized set of arrays
 * for the collector to guess about — which is the point of [liveCount] in the cancellation test.
 */
class DiffusionBuffers(val latentSize: Int) {

    init {
        require(latentSize > 0) { "latent size must be positive, was $latentSize" }
        live.incrementAndGet()
    }

    private var latentStore: FloatArray? = FloatArray(latentSize)
    private var scaledStore: FloatArray? = FloatArray(latentSize)
    private var conditionalStore: FloatArray? = FloatArray(latentSize)
    private var unconditionalStore: FloatArray? = FloatArray(latentSize)
    private var decodeStore: FloatArray? = FloatArray(latentSize)

    @Volatile
    private var released = false

    /** The latent being denoised: seeded with noise, then updated in place by every step. */
    val latent: FloatArray get() = alive(latentStore, "latent")

    /** The VAE-scaled copy handed to the UNet; refilled from [latent] before each UNet call. */
    val scaled: FloatArray get() = alive(scaledStore, "scaled")

    /** UNet output for the conditional (prompt) pass. */
    val conditionalEps: FloatArray get() = alive(conditionalStore, "conditionalEps")

    /** UNet output for the unconditional (negative prompt) pass. */
    val unconditionalEps: FloatArray get() = alive(unconditionalStore, "unconditionalEps")

    /** The latent scaled back into the decoder's space — the last large buffer of the pass. */
    val decodeScratch: FloatArray get() = alive(decodeStore, "decodeScratch")

    val isReleased: Boolean get() = released

    /** Bytes held while this working set is alive. */
    val bytes: Int get() = latentSize * Float.SIZE_BYTES * ARRAYS

    /**
     * Drops every buffer. Idempotent, so a `finally` and a cancellation handler can both call it
     * without double-counting; reading any buffer afterwards throws instead of returning a stale
     * array that a late writer could scribble on.
     */
    fun release() {
        if (released) return
        released = true
        latentStore = null
        scaledStore = null
        conditionalStore = null
        unconditionalStore = null
        decodeStore = null
        live.decrementAndGet()
    }

    private fun alive(store: FloatArray?, name: String): FloatArray =
        store ?: throw IllegalStateException("$name was read after the pass released its buffers")

    companion object {
        /** Buffers per pass: latent, scaled, conditional eps, unconditional eps, decode scratch. */
        const val ARRAYS = 5

        private val live = AtomicInteger(0)

        /** Working sets alive right now. Used by the cancellation test; not read in production. */
        fun liveCount(): Int = live.get()
    }
}
