package ai.pixelmind

import ai.pixelmind.engine.TokenizerLoader
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * The vocabulary is parsed by the first generation, not by opening the app.
 *
 * `buildSessions` used to parse `vocab.json` (49k entries) and `merges.txt` (48k ranks) every time
 * the engine started — at launch and again on every return from background, because the sessions
 * are released on stop. Nothing on the Home or Gallery screens needs the tokenizer, so a user who
 * opened the app to look at an image paid for it and never used it.
 */
class TokenizerLoaderTest {

    private val vocabJson =
        """{"<|startoftext|>":0,"<|endoftext|>":1,"a":2,"b":3,"ab":4,"a</w>":5,"b</w>":6,"ab</w>":7}"""

    private fun vocab(dir: File): File = File(dir, "vocab.json").apply { writeText(vocabJson) }

    private fun merges(dir: File): File = File(dir, "merges.txt").apply {
        writeText("#version: 0.2\na b\n")
    }

    @Test
    fun `nothing is parsed until the first generation asks for the tokenizer`() {
        val dir = createTempDir("tokenizer")
        val loader = TokenizerLoader(vocab(dir), merges(dir), maxLength = 8)

        assertFalse("construction must not parse the table", loader.isLoaded)

        // If the files had already been read, deleting them would change nothing. They have not
        // been, so the first use is the first read and it fails on the missing file.
        dir.listFiles()?.forEach { it.delete() }
        try {
            loader.get()
            fail("the loader read the vocabulary before it was asked to")
        } catch (expected: java.io.FileNotFoundException) {
            // Expected: this is the proof that the parse was deferred.
        }
    }

    @Test
    fun `the second generation reuses the parsed table instead of parsing again`() {
        val dir = createTempDir("tokenizer")
        val loader = TokenizerLoader(vocab(dir), merges(dir), maxLength = 8)

        val first = loader.get()
        assertTrue(loader.isLoaded)
        assertTrue("the vocabulary should encode", first.encode("ab").isNotEmpty())

        // A second parse would fail here; a cached one cannot.
        dir.listFiles()?.forEach { it.delete() }
        assertSame("the same parsed tokenizer must be handed out", first, loader.get())
    }

    @Test
    fun `concurrent first uses parse once and agree on the result`() {
        val dir = createTempDir("tokenizer")
        val loader = TokenizerLoader(vocab(dir), merges(dir), maxLength = 8)

        val threads = 8
        val ready = CountDownLatch(threads)
        val go = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(threads)
        val results = Array(threads) { _ -> null as ai.pixelmind.engine.ClipTokenizer? }
        try {
            repeat(threads) { index ->
                pool.submit {
                    ready.countDown()
                    go.await()
                    results[index] = loader.get()
                }
            }
            ready.await(10, TimeUnit.SECONDS)
            go.countDown()
            pool.shutdown()
            assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS))
        } finally {
            pool.shutdownNow()
        }

        val first = results.first()
        results.forEach { assertSame(first, it) }
        assertEquals(threads, results.size)
    }
}
