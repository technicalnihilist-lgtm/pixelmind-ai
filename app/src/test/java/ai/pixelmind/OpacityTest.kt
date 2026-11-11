package ai.pixelmind

import ai.pixelmind.util.Opacity
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The alpha test behind the PNG encoder change.
 *
 * A generated image is opaque in every pixel — the decoder writes `0xFF` into the alpha byte of
 * every one — but the bitmap still carries an alpha channel, so the encoder wrote 32-bit RGBA and
 * the file was a quarter larger than it needed to be. Declaring the bitmap opaque is only safe if
 * the pixels really are opaque, which is what this decides; a bitmap with any transparency keeps
 * its channel.
 */
class OpacityTest {

    @Test
    fun `a row of fully opaque pixels is opaque`() {
        assertTrue(Opacity.isOpaque(IntArray(16) { 0xFF123456.toInt() }))
    }

    @Test
    fun `a single transparent pixel makes the row transparent`() {
        val row = IntArray(16) { 0xFF123456.toInt() }
        row[9] = 0x00123456
        assertFalse(Opacity.isOpaque(row))
    }

    @Test
    fun `a nearly opaque pixel still counts as transparent`() {
        val row = IntArray(4) { 0xFF000000.toInt() }
        row[0] = 0xFE000000.toInt()
        assertFalse(Opacity.isOpaque(row))
    }

    @Test
    fun `only the requested prefix of the row is inspected`() {
        val row = IntArray(8) { 0xFF000000.toInt() }
        row[7] = 0x00000000
        assertTrue(Opacity.isOpaque(row, count = 7))
        assertFalse(Opacity.isOpaque(row, count = 8))
    }

    @Test
    fun `an empty row is opaque and an over-long count is clamped`() {
        assertTrue(Opacity.isOpaque(IntArray(0)))
        assertTrue(Opacity.isOpaque(IntArray(2) { 0xFF000000.toInt() }, count = 99))
    }
}
