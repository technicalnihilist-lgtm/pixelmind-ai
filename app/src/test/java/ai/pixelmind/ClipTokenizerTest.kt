package ai.pixelmind

import ai.pixelmind.engine.ClipTokenizer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ClipTokenizerTest {

    // A miniature vocabulary: enough to exercise BPE merging without shipping the 49k-entry file.
    private fun tokenizer(): ClipTokenizer {
        val encoder = mutableMapOf(
            "<|startoftext|>" to 0,
            "<|endoftext|>" to 1
        )
        // Map single characters and a couple of merged pairs.
        val chars = "abcdefghijklmnopqrstuvwxyz0123456789!?,.</w>"
        chars.forEachIndexed { index, c -> encoder[c.toString()] = 100 + index }
        encoder["he</w>"] = 400
        encoder["llo</w>"] = 401
        encoder["hello</w>"] = 402
        val ranks = mapOf(
            "h\u0000e</w>" to 0,
            "he</w>\u0000l" to 1,
            "he</w>\u0000ll" to 2,
            "he</w>\u0000llo</w>" to 3
        )
        return ClipTokenizer(encoder, ranks, maxLength = 8)
    }

    @Test
    fun `encoding is padded to a fixed length and framed by bos and eos`() {
        val ids = tokenizer().encode("hello")
        assertEquals(8, ids.size)
        assertEquals(0L, ids[0])
        assertEquals(1L, ids[ids.size - 1])
    }

    @Test
    fun `empty prompt still produces a valid framed sequence`() {
        val ids = tokenizer().encode("")
        assertEquals(8, ids.size)
        assertEquals(0L, ids[0])
        assertTrue(ids.drop(1).all { it == 1L })
    }

    @Test
    fun `word splitting follows the clip pre-tokenizer`() {
        val tokens = tokenizer().tokenize("hello, world")
        assertTrue(tokens.isNotEmpty())
        // The comma must be its own token, not glued to the word.
        assertTrue(tokens.any { it.startsWith(",") })
    }

    @Test
    fun `lowercasing is applied before tokenisation`() {
        val upper = tokenizer().tokenize("HELLO")
        val lower = tokenizer().tokenize("hello")
        assertEquals(lower, upper)
    }
}
