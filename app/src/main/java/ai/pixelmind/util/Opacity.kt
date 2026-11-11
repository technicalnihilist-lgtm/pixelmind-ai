package ai.pixelmind.util

/**
 * Alpha inspection for generated images.
 *
 * The diffusion decoder writes every pixel with an alpha byte of `0xFF`, so a generated image is
 * opaque by construction — but the bitmap still carries an alpha channel, and a PNG encoded from
 * it therefore carries one too. Checking the pixels rather than assuming is what makes it safe to
 * drop the channel at encode time: a bitmap that really does have transparency keeps it.
 */
object Opacity {

    /**
     * True when the first [count] pixels of [argb] are all fully opaque.
     *
     * `argb` is a packed ARGB row as returned by `Bitmap.getPixels`.
     */
    fun isOpaque(argb: IntArray, count: Int = argb.size): Boolean {
        val limit = count.coerceIn(0, argb.size)
        for (i in 0 until limit) {
            if ((argb[i] ushr 24) != 0xFF) return false
        }
        return true
    }
}
