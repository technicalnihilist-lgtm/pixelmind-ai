package ai.pixelmind.engine

import android.util.Log
import java.io.File

/**
 * Parses the CLIP vocabulary the first time a generation needs it, not when the sessions open.
 *
 * `vocab.json` is ~49,000 entries and `merges.txt` ~48,000 ranks, so [ClipTokenizer.fromFiles] is
 * a JSON parse plus two large hash-map builds. It used to run inside `buildSessions`, which
 * happens at app start and again on every return from background (the sessions are released on
 * stop), so a user who opened the app only to browse the gallery paid that parse every time for a
 * tokenizer no generation ever touched.
 *
 * The parse is now paid once, by the first generation, and the result lives as long as the loaded
 * engine: `release()` drops it together with the sessions. Parsing is idempotent and serialised,
 * because a retry that lands while the first generation is still starting would otherwise parse
 * the table twice and keep whichever finished last.
 */
internal class TokenizerLoader(
    private val vocab: File,
    private val merges: File,
    private val maxLength: Int
) {

    @Volatile
    private var parsed: ClipTokenizer? = null

    private val lock = Any()

    /** False until the first generation has parsed the table. */
    val isLoaded: Boolean get() = parsed != null

    fun get(): ClipTokenizer {
        parsed?.let { return it }
        synchronized(lock) {
            parsed?.let { return it }
            val started = System.currentTimeMillis()
            val tokenizer = ClipTokenizer.fromFiles(vocab, merges, maxLength)
            parsed = tokenizer
            Log.i(TAG, "tokenizer parsed on first use in ${System.currentTimeMillis() - started} ms")
            return tokenizer
        }
    }

    private companion object {
        const val TAG = "TokenizerLoader"
    }
}
