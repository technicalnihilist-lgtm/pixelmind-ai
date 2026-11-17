package ai.pixelmind.engine

/**
 * The tensor contract an imported model has to satisfy before it is kept.
 *
 * Opening a session proves the file is a well-formed ONNX graph. It does not prove the graph is
 * *this* pipeline's graph: a UNet exported with a different channel count, or a text encoder with a
 * hidden size other than 768, opens cleanly and then fails — or, worse, produces noise — on the
 * first generation. By then the import has been reported as successful and `models.previous` has
 * been deleted, so the user is left with a model that cannot generate and nothing to roll back to.
 *
 * The check is deliberately narrow. It looks only at what the Kotlin pipeline hard-codes: the input
 * names it feeds by position, the output name it reads, the 768-wide text embedding, the
 * 4-channel latent and the 3-channel decoder output. Dynamic dimensions (-1) are left alone,
 * because the pipeline runs several resolutions and aspect ratios through the same graphs.
 *
 * Nothing here is measured on a device: it is a comparison between the manifest and what the
 * session reports about itself.
 */
internal object TensorContract {

    /** The hidden width the pipeline builds its conditioning tensor with. */
    const val HIDDEN_SIZE = 768L

    /** The decoder's output channels: the PNG writer reads R, G and B, in that order. */
    const val RGB_CHANNELS = 3L

    /** What one opened session reports: input and output names, and their shapes. */
    data class Opened(
        val graph: GraphName,
        val inputs: Map<String, LongArray>,
        val outputs: Map<String, LongArray>
    )

    data class Violation(val graph: GraphName, val reason: String)

    /** Null when every graph fits the pipeline, otherwise the first thing that does not. */
    fun check(manifest: ModelManifest, opened: List<Opened>): Violation? {
        val byGraph = opened.associateBy { it.graph }
        return checkTextEncoder(manifest, byGraph[GraphName.TEXT_ENCODER])
            ?: checkUnet(manifest, byGraph[GraphName.UNET])
            ?: checkVaeDecoder(manifest, byGraph[GraphName.VAE_DECODER])
    }

    private fun checkTextEncoder(m: ModelManifest, o: Opened?): Violation? {
        val graph = GraphName.TEXT_ENCODER
        if (o == null) return Violation(graph, "its session was not opened")
        missingInputs(o, m.textEncoder)?.let { return Violation(graph, it) }
        missingOutput(o, m.textEncoder)?.let { return Violation(graph, it) }

        val hidden = dim(o.outputs.getValue(m.textEncoder.outputName), 2)
        if (hidden != null && hidden != HIDDEN_SIZE) {
            return Violation(
                graph,
                "it produces $hidden-wide embeddings, but the pipeline conditions on " +
                    "$HIDDEN_SIZE-wide ones, so every prompt would be misread"
            )
        }
        return null
    }

    private fun checkUnet(m: ModelManifest, o: Opened?): Violation? {
        val graph = GraphName.UNET
        if (o == null) return Violation(graph, "its session was not opened")
        val spec = m.unet
        if (spec.inputNames.size < 3) {
            return Violation(
                graph,
                "its manifest lists ${spec.inputNames.size} inputs, but the pipeline feeds three " +
                    "(latent sample, timestep, encoder_hidden_states)"
            )
        }
        missingInputs(o, spec)?.let { return Violation(graph, it) }
        missingOutput(o, spec)?.let { return Violation(graph, it) }

        val channels = dim(o.inputs.getValue(spec.inputNames[0]), 1)
        if (channels != null && channels != m.latentChannels.toLong()) {
            return Violation(
                graph,
                "it takes $channels-channel latents, but the manifest declares " +
                    "${m.latentChannels}"
            )
        }
        val conditioning = dim(o.inputs.getValue(spec.inputNames[2]), 2)
        if (conditioning != null && conditioning != HIDDEN_SIZE) {
            return Violation(
                graph,
                "it expects $conditioning-wide conditioning, but the text encoder produces " +
                    "$HIDDEN_SIZE"
            )
        }
        val outChannels = dim(o.outputs.getValue(spec.outputName), 1)
        if (outChannels != null && outChannels != m.latentChannels.toLong()) {
            return Violation(
                graph,
                "it predicts $outChannels-channel latents, but the manifest declares " +
                    "${m.latentChannels}"
            )
        }
        return null
    }

    private fun checkVaeDecoder(m: ModelManifest, o: Opened?): Violation? {
        val graph = GraphName.VAE_DECODER
        if (o == null) return Violation(graph, "its session was not opened")
        val spec = m.vaeDecoder
        missingInputs(o, spec)?.let { return Violation(graph, it) }
        missingOutput(o, spec)?.let { return Violation(graph, it) }

        val channels = dim(o.inputs.getValue(spec.inputNames[0]), 1)
        if (channels != null && channels != m.latentChannels.toLong()) {
            return Violation(
                graph,
                "it takes $channels-channel latents, but the manifest declares " +
                    "${m.latentChannels}"
            )
        }
        val out = o.outputs.getValue(spec.outputName)
        if (out.isNotEmpty() && out.size != 4) {
            return Violation(graph, "it returns a rank-${out.size} tensor, but an image is rank 4")
        }
        val outChannels = dim(out, 1)
        if (outChannels != null && outChannels != RGB_CHANNELS) {
            return Violation(
                graph,
                "it produces $outChannels-channel images, but the pipeline reads RGB"
            )
        }
        return null
    }

    private fun missingInputs(o: Opened, spec: ModelManifest.GraphSpec): String? {
        val missing = spec.inputNames.filterNot { o.inputs.containsKey(it) }
        if (missing.isEmpty()) return null
        val present = o.inputs.keys.joinToString(", ") { "`$it`" }.ifEmpty { "none" }
        return "it has no input named ${missing.joinToString(", ") { "`$it`" }}, so the pipeline " +
            "cannot feed it. The graph's inputs are $present"
    }

    private fun missingOutput(o: Opened, spec: ModelManifest.GraphSpec): String? {
        if (o.outputs.containsKey(spec.outputName)) return null
        val present = o.outputs.keys.joinToString(", ") { "`$it`" }.ifEmpty { "none" }
        return "it has no output named `${spec.outputName}`, which is the one the pipeline reads. " +
            "The graph's outputs are $present"
    }

    /** A dimension when it is static and positive; null when it is dynamic or absent. */
    private fun dim(shape: LongArray?, index: Int): Long? =
        shape?.getOrNull(index)?.takeIf { it > 0 }
}
