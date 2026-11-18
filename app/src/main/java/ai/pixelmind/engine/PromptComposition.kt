package ai.pixelmind.engine

/**
 * The two strings the text encoder is given, and which of them may contain the negative prompt.
 *
 * Classifier-free guidance subtracts the unconditional pass from the conditional one, so the
 * negative prompt belongs to the unconditional pass *only*. Putting it in the conditional text
 * as well does not merely weaken it: it cancels part of itself out and leaves the guidance
 * scale with nothing to steer on. Small enough to be a function, so it is one — the engine used
 * to build both strings inline, where no test could reach them.
 */
internal object PromptComposition {

    /**
     * What the conditional pass sees: the user's words plus the style's art direction.
     *
     * [GenerationRequest.negativePrompt] is deliberately absent. The style's own
     * `negativeSuffix` is not appended either — it would be a change in what the model is told,
     * and the shipped presets leave it blank.
     */
    fun conditional(request: GenerationRequest): String = buildString {
        append(request.prompt.trim())
        if (request.style.promptSuffix.isNotBlank()) append(", ").append(request.style.promptSuffix)
    }

    /** What the unconditional pass sees: the negative prompt, and nothing else. */
    fun unconditional(request: GenerationRequest): String = request.negativePrompt.trim()
}
