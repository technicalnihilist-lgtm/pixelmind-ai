package ai.pixelmind

import ai.pixelmind.engine.GraphName
import ai.pixelmind.engine.ModelImporter
import ai.pixelmind.engine.ModelManifest
import ai.pixelmind.engine.ModelSwap
import ai.pixelmind.engine.TensorContract
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * The contract an imported model has to meet before it is kept.
 *
 * Session creation proves a file is a well-formed graph; it does not prove the graph belongs to
 * this pipeline. These cases are the mismatches that open cleanly and then fail — or quietly
 * produce noise — on the first generation, which used to happen *after* the import had been
 * reported as successful and the previous model deleted.
 */
class TensorContractTest {

    @get:Rule
    val temp = TemporaryFolder()

    private val files = listOf(
        "unet/model.onnx",
        "text_encoder/model.onnx",
        "vae_decoder/model.onnx",
        "clip/vocab.json",
        "clip/merges.txt"
    )

    private fun manifest(): ModelManifest {
        val root = temp.newFolder("staging")
        files.forEach { File(root, it).apply { parentFile?.mkdirs() }.writeText("weights:$it") }
        val detected = ModelImporter.detectComponents(files)!!
        return ModelManifest.parse(ModelImporter.buildManifestJson(detected, root)!!)
    }

    /** The graphs the bundled verification model reports, which is the shape the app ships with. */
    private fun shipped(
        encoderHidden: Long = 768,
        latentChannels: Long = 4,
        conditioning: Long = 768,
        rgb: Long = 3,
        unetInputs: Map<String, LongArray>? = null,
        vaeOutputs: Map<String, LongArray>? = null
    ) = listOf(
        TensorContract.Opened(
            GraphName.TEXT_ENCODER,
            inputs = mapOf("input_ids" to longArrayOf(1, 77)),
            outputs = mapOf("last_hidden_state" to longArrayOf(1, 77, encoderHidden))
        ),
        TensorContract.Opened(
            GraphName.UNET,
            inputs = unetInputs ?: mapOf(
                "sample" to longArrayOf(1, latentChannels, -1, -1),
                "timestep" to longArrayOf(1),
                "encoder_hidden_states" to longArrayOf(1, 77, conditioning)
            ),
            outputs = mapOf("out_sample" to longArrayOf(1, latentChannels, -1, -1))
        ),
        TensorContract.Opened(
            GraphName.VAE_DECODER,
            inputs = mapOf("latent_sample" to longArrayOf(1, latentChannels, -1, -1)),
            outputs = vaeOutputs ?: mapOf("sample" to longArrayOf(1, rgb, -1, -1))
        )
    )

    @Test
    fun `the shipped contract passes`() {
        assertNull(TensorContract.check(manifest(), shipped()))
    }

    @Test
    fun `dynamic spatial dimensions are not judged`() {
        // Every resolution the app runs goes through the same graphs, so H and W are -1 by design.
        assertNull(TensorContract.check(manifest(), shipped()))
    }

    @Test
    fun `a text encoder of the wrong width is rejected by name`() {
        val violation = TensorContract.check(manifest(), shipped(encoderHidden = 512))
        assertNotNull(violation)
        assertEquals(GraphName.TEXT_ENCODER, violation!!.graph)
        assertTrue(violation.reason, violation.reason.contains("512"))
        assertTrue(violation.reason, violation.reason.contains("768"))
    }

    @Test
    fun `a unet that takes the wrong latent channels is rejected`() {
        val violation = TensorContract.check(manifest(), shipped(latentChannels = 8))
        assertNotNull(violation)
        assertEquals(GraphName.UNET, violation!!.graph)
        assertTrue(violation.reason, violation.reason.contains("8-channel"))
    }

    @Test
    fun `a unet conditioned on the wrong width is rejected`() {
        val violation = TensorContract.check(manifest(), shipped(conditioning = 1024))
        assertNotNull(violation)
        assertEquals(GraphName.UNET, violation!!.graph)
        assertTrue(violation.reason, violation.reason.contains("1024"))
    }

    @Test
    fun `a decoder that returns the wrong channel count is rejected`() {
        val violation = TensorContract.check(manifest(), shipped(rgb = 4))
        assertNotNull(violation)
        assertEquals(GraphName.VAE_DECODER, violation!!.graph)
        assertTrue(violation.reason, violation.reason.contains("RGB"))
    }

    @Test
    fun `a missing input is reported with the name the pipeline needs`() {
        val graphs = shipped().map { opened ->
            if (opened.graph == GraphName.UNET) {
                opened.copy(inputs = opened.inputs - "encoder_hidden_states")
            } else {
                opened
            }
        }
        val violation = TensorContract.check(manifest(), graphs)
        assertNotNull(violation)
        assertEquals(GraphName.UNET, violation!!.graph)
        assertTrue(violation.reason, violation.reason.contains("encoder_hidden_states"))
    }

    @Test
    fun `a missing output is reported with the name the pipeline reads`() {
        val graphs = shipped().map { opened ->
            if (opened.graph == GraphName.VAE_DECODER) {
                opened.copy(outputs = mapOf("decoded" to longArrayOf(1, 3, -1, -1)))
            } else {
                opened
            }
        }
        val violation = TensorContract.check(manifest(), graphs)
        assertNotNull(violation)
        assertEquals(GraphName.VAE_DECODER, violation!!.graph)
        assertTrue(violation.reason, violation.reason.contains("sample"))
    }

    @Test
    fun `a missing session is a violation rather than a crash`() {
        val graphs = shipped().filterNot { it.graph == GraphName.UNET }
        val violation = TensorContract.check(manifest(), graphs)
        assertNotNull(violation)
        assertEquals(GraphName.UNET, violation!!.graph)
    }

    /**
     * The end-to-end reason this check exists: a model that opens but does not fit is rolled back
     * to the previous one instead of being kept and failing on every generation.
     */
    @Test
    fun `a model that loads with the wrong shapes is rolled back to the previous one`() {
        val root = File(temp.root, "models").apply {
            File(this, "unet").mkdirs()
            File(this, "unet/model.onnx").writeText("previous weights")
        }
        val staging = File(temp.root, "models.staging").apply {
            File(this, "unet").mkdirs()
            File(this, "unet/model.onnx").writeText("imported weights")
        }
        val previous = File(temp.root, "models.previous")

        // The import itself is accepted — the file opens — so the swap goes ahead.
        assertNull(ModelSwap.install(root, staging, previous))
        assertEquals("imported weights", File(root, "unet/model.onnx").readText())

        // Only then does the engine see that the graph does not fit the pipeline.
        val violation = TensorContract.check(manifest(), shipped(latentChannels = 16))
        assertNotNull(violation)

        assertTrue(ModelSwap.rollback(root, previous))
        assertEquals("previous weights", File(root, "unet/model.onnx").readText())
    }
}
