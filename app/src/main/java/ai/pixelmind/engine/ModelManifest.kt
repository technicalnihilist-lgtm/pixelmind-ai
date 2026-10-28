package ai.pixelmind.engine

import org.json.JSONObject

/**
 * Describes the ONNX graph files that make up the pipeline. Kept as data (not hard-coded paths)
 * so a re-export with different tensor names or a different precision does not need code changes.
 */
data class ModelManifest(
    val name: String,
    val version: String,
    val architecture: String,
    val precision: String,
    val license: String,
    val components: List<Component>,
    val textEncoder: GraphSpec,
    val unet: GraphSpec,
    val vaeDecoder: GraphSpec,
    val tokenizer: TokenizerSpec,
    val latentChannels: Int,
    val sampleSize: Int
) {
    data class Component(val file: String, val bytes: Long, val sha256: String)

    data class GraphSpec(
        val file: String,
        val inputNames: List<String>,
        val outputName: String
    )

    data class TokenizerSpec(
        val vocabFile: String,
        val mergesFile: String,
        val maxLength: Int,
        val bosToken: String,
        val eosToken: String,
        val padToken: String
    )

    val totalBytes: Long get() = components.sumOf { it.bytes }

    companion object {
        const val FILE_NAME = "model_manifest.json"

        fun parse(json: String): ModelManifest {
            val root = JSONObject(json)
            fun graph(key: String): GraphSpec {
                val g = root.getJSONObject(key)
                val inputs = g.getJSONArray("inputs").let { arr ->
                    List(arr.length()) { arr.getString(it) }
                }
                return GraphSpec(
                    file = g.getString("file"),
                    inputNames = inputs,
                    outputName = g.getString("output")
                )
            }
            val componentsJson = root.getJSONArray("components")
            val components = List(componentsJson.length()) { i ->
                val c = componentsJson.getJSONObject(i)
                Component(
                    file = c.getString("file"),
                    bytes = c.optLong("bytes", 0L),
                    sha256 = c.optString("sha256", "")
                )
            }
            val tok = root.getJSONObject("tokenizer")
            return ModelManifest(
                name = root.getString("name"),
                version = root.getString("version"),
                architecture = root.getString("architecture"),
                precision = root.optString("precision", "int8"),
                license = root.optString("license", "CreativeML OpenRAIL-M"),
                components = components,
                textEncoder = graph("text_encoder"),
                unet = graph("unet"),
                vaeDecoder = graph("vae_decoder"),
                tokenizer = TokenizerSpec(
                    vocabFile = tok.getString("vocab"),
                    mergesFile = tok.getString("merges"),
                    maxLength = tok.optInt("max_length", 77),
                    bosToken = tok.optString("bos", "<|startoftext|>"),
                    eosToken = tok.optString("eos", "<|endoftext|>"),
                    padToken = tok.optString("pad", "<|endoftext|>")
                ),
                latentChannels = root.optInt("latent_channels", 4),
                sampleSize = root.optInt("sample_size", 64)
            )
        }
    }
}
