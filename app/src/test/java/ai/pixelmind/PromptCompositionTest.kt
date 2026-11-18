package ai.pixelmind

import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.PromptComposition
import ai.pixelmind.engine.StylePreset
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Which string reaches which text-encoder pass.
 *
 * Classifier-free guidance subtracts the unconditional pass from the conditional one, so the
 * negative prompt has to be in the unconditional pass *only*. In both, it partly cancels itself
 * out. The two strings used to be built inline inside the generation coroutine, where no test
 * could reach them.
 */
class PromptCompositionTest {

    @Test
    fun `the conditional text is the prompt plus the style's art direction`() {
        val request = GenerationRequest(prompt = "a lighthouse", style = StylePreset.CINEMATIC)
        val text = PromptComposition.conditional(request)
        assertTrue(text.startsWith("a lighthouse"))
        assertTrue(text.endsWith(StylePreset.CINEMATIC.promptSuffix))
    }

    @Test
    fun `the conditional text never contains the negative prompt`() {
        val request = GenerationRequest(
            prompt = "a lighthouse",
            negativePrompt = "blurry, watermark, text",
            style = StylePreset.PHOTOREAL
        )
        val text = PromptComposition.conditional(request)
        assertFalse("negative prompt leaked into the conditional pass", text.contains("blurry"))
        assertFalse(text.contains("watermark"))
    }

    @Test
    fun `the unconditional text is exactly the negative prompt`() {
        val request = GenerationRequest(
            prompt = "a lighthouse",
            negativePrompt = "  blurry, watermark, text  ",
            style = StylePreset.PHOTOREAL
        )
        assertEquals("blurry, watermark, text", PromptComposition.unconditional(request))
        assertFalse(
            "the style suffix belongs to the conditional pass",
            PromptComposition.unconditional(request).contains(StylePreset.PHOTOREAL.promptSuffix)
        )
    }

    @Test
    fun `an empty negative prompt produces an empty unconditional string`() {
        val request = GenerationRequest(prompt = "a lighthouse", negativePrompt = "")
        assertEquals("", PromptComposition.unconditional(request))
        assertTrue(PromptComposition.conditional(request).isNotBlank())
    }

    @Test
    fun `the prompt is trimmed before it is encoded`() {
        val request = GenerationRequest(prompt = "   a lighthouse   ", style = StylePreset.ABSTRACT)
        val text = PromptComposition.conditional(request)
        assertTrue("leading space reached the tokenizer", text.startsWith("a lighthouse"))
        assertFalse(text.contains("  "))
    }

    @Test
    fun `every shipped preset produces a non-empty conditional string`() {
        for (style in StylePreset.entries) {
            val text = PromptComposition.conditional(GenerationRequest(prompt = "x", style = style))
            assertTrue("${style.label} produced \"$text\"", text.isNotBlank())
        }
    }
}
