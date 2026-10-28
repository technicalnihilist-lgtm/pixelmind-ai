package ai.pixelmind.engine

import org.json.JSONObject
import java.io.File
import kotlin.math.min

/**
 * A faithful port of the CLIP byte-pair tokenizer that Stable Diffusion text encoders expect.
 *
 * Prompt text has to be tokenised exactly the way the checkpoint was trained, or the
 * conditioning degrades into noise. Two details that are easy to get wrong and are handled
 * here: bytes are mapped through CLIP's unicode table before BPE, and the final character of
 * each word carries the `</w>` end-of-word marker.
 */
class ClipTokenizer(
    private val encoder: Map<String, Int>,
    private val bpeRanks: Map<String, Int>,
    private val maxLength: Int = 77,
    bosToken: String = "<|startoftext|>",
    eosToken: String = "<|endoftext|>",
    padToken: String = "<|endoftext|>"
) {
    private val cache = HashMap<String, List<String>>()
    private val byteEncoder: Map<Byte, String> = buildByteEncoder()

    private val bosId = encoder[bosToken] ?: 49406
    private val eosId = encoder[eosToken] ?: 49407
    private val padId = encoder[padToken] ?: 49407

    /** Fixed-length LongArray of token ids: BOS + content + EOS + PAD. */
    fun encode(text: String): LongArray {
        val ids = ArrayList<Int>(maxLength)
        ids += bosId
        outer@ for (token in tokenize(text)) {
            val id = encoder[token] ?: continue
            if (ids.size >= maxLength - 1) break@outer
            ids += id
        }
        ids += eosId
        val out = LongArray(maxLength) { padId.toLong() }
        for (i in 0 until min(ids.size, maxLength)) out[i] = ids[i].toLong()
        return out
    }

    /** Splits raw text into vocabulary tokens. Visible for testing. */
    fun tokenize(text: String): List<String> {
        // CLIP lowercases before BPE; doing it here means every caller gets the same tokens.
        val lowered = text.lowercase()
        val tokens = ArrayList<String>()
        for (match in PATTERN.toRegex().findAll(lowered)) {
            val mapped = buildString {
                for (b in match.value.toByteArray(Charsets.UTF_8)) append(byteEncoder[b] ?: "")
            }
            tokens += bpeWord(mapped)
        }
        return tokens
    }

    private fun bpeWord(token: String): List<String> {
        if (token.isEmpty()) return emptyList()
        cache[token]?.let { return it }
        if (token.length == 1) {
            val single = listOf(token + END_OF_WORD)
            cache[token] = single
            return single
        }

        var word: List<String> =
            token.dropLast(1).map { it.toString() } + (token.last() + END_OF_WORD)

        var pairs = getPairs(word)
        while (true) {
            var bestRank = Int.MAX_VALUE
            var best: String? = null
            for (pair in pairs) {
                val rank = bpeRanks[pair] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    best = pair
                }
            }
            val merged = best ?: break
            val first = merged.substring(0, merged.length - 1)
            val second = merged.substring(merged.length - 1)

            val newWord = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                val j = word.subList(i, word.size).indexOf(first)
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                newWord.addAll(word.subList(i, i + j))
                i += j
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    newWord += first + second
                    i += 2
                } else {
                    newWord += word[i]
                    i += 1
                }
            }
            word = newWord
            if (word.size == 1) break
            pairs = getPairs(word)
        }
        cache[token] = word
        return word
    }

    private fun getPairs(word: List<String>): Set<String> {
        val pairs = LinkedHashSet<String>(word.size)
        for (i in 0 until word.size - 1) {
            // Rank table keys are "left right"; keeping the same encoding avoids a tuple alloc
            // per candidate pair on every merge round.
            pairs += word[i] + PAIR_SEPARATOR + word[i + 1]
        }
        return pairs
    }

    companion object {
        private const val END_OF_WORD = "</w>"
        private const val PAIR_SEPARATOR = '\u0000'

        // The exact pattern OpenAI's CLIP uses to pre-split text before BPE.
        private const val PATTERN =
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"

        private fun bytesToUnicode(): Pair<List<Int>, List<String>> {
            val bs = mutableListOf<Int>()
            for (b in '!'.code..'~'.code) bs += b
            for (b in '\u00A1'.code..'\u00AC'.code) bs += b
            for (b in '\u00AE'.code..'\u00FF'.code) bs += b
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs += b
                    cs += 256 + n
                    n += 1
                }
            }
            return bs to cs.map { String(Character.toChars(it)) }
        }

        private fun buildByteEncoder(): Map<Byte, String> {
            val (bs, cs) = bytesToUnicode()
            return bs.indices.associate { bs[it].toByte() to cs[it] }
        }

        /** Loads the standard CLIP `vocab.json` + `merges.txt` pair from disk. */
        fun fromFiles(vocab: File, merges: File, maxLength: Int = 77): ClipTokenizer {
            val encoder = HashMap<String, Int>(49152)
            val vocabJson = JSONObject(vocab.readText())
            val keys = vocabJson.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                encoder[key] = vocabJson.getInt(key)
            }

            val ranks = HashMap<String, Int>(50000)
            var index = 0
            merges.readLines().forEach { line ->
                val trimmed = line.trim()
                if (trimmed.isEmpty() || trimmed.startsWith("#")) return@forEach
                val parts = trimmed.split(" ")
                if (parts.size == 2) {
                    ranks[parts[0] + PAIR_SEPARATOR + parts[1]] = index
                    index += 1
                }
            }
            return ClipTokenizer(encoder, ranks, maxLength)
        }
    }
}
