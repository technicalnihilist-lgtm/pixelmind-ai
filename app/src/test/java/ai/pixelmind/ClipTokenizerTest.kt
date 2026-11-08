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
    fun `a prompt longer than the context window is truncated, not overflowed`() {
        val ids = tokenizer().encode("hello hello hello hello hello hello hello hello")
        assertEquals(8, ids.size)
        assertEquals(0L, ids[0])
        assertEquals(1L, ids[ids.size - 1])
    }

    @Test
    fun `repeated encodes return identical ids`() {
        val tokenizer = tokenizer()
        assertTrue(tokenizer.encode("hello, world").contentEquals(tokenizer.encode("hello, world")))
    }

    @Test
    fun `lowercasing is applied before tokenisation`() {
        val upper = tokenizer().tokenize("HELLO")
        val lower = tokenizer().tokenize("hello")
        assertEquals(lower, upper)
    }

    @Test
    fun `the BPE cache is bounded however many distinct words are tokenised`() {
        // Item 26: the cache grows with prompt vocabulary, so it must have a hard ceiling. 2048
        // is the limit in ClipTokenizer; three thousand distinct words must not push past it.
        val tokenizer = tokenizer()
        fun word(i: Int): String {
            var n = i
            val sb = StringBuilder()
            do {
                sb.append('a' + n % 26)
                n /= 26
            } while (n > 0)
            return sb.toString()
        }
        val words = (0 until 3000).map { word(it) }
        assertEquals("the fixture must produce distinct words", 3000, words.toSet().size)

        tokenizer.tokenize(words.joinToString(" "))

        assertEquals(2048, tokenizer.cachedWordCount())
    }

    @Test(timeout = 10_000)
    fun `a ranked pair is actually merged`() {
        // The merge used to split the "left\u0000right" rank key by character rather than on the
        // separator, so no adjacent pair ever matched: `he` produced [h, e</w>] instead of the
        // merged token, and a word with any ranked pair in it looped for ever.
        val tokenizer = tokenizer()

        assertEquals(listOf("he</w>"), tokenizer.tokenize("he"))
    }

    @Test(timeout = 10_000)
    fun `a merge that makes no progress cannot spin for ever`() {
        // Defensive guard, kept because a hand-edited or corrupted merges.txt can ship in an
        // imported model: a merge that removes nothing must break out, not loop.
        val encoder = mutableMapOf("<|startoftext|>" to 0, "<|endoftext|>" to 1)
        "helo".forEachIndexed { index, c -> encoder[c.toString()] = 100 + index }
        // A rank whose right half is empty: the pair key is "h\u0000", which can be looked up but
        // cannot describe a merge.
        val ranks = mapOf("h\u0000" to 0, "he</w>\u0000l" to 1)
        val tokenizer = ClipTokenizer(encoder, ranks, maxLength = 8)

        assertTrue(tokenizer.tokenize("hel").isNotEmpty())
    }
}
