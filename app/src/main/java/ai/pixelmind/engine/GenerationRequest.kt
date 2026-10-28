package ai.pixelmind.engine

/** Visual presets. Each one maps to a prompt suffix plus sampler settings the model supports. */
enum class StylePreset(
    val label: String,
    val promptSuffix: String,
    val negativeSuffix: String = ""
) {
    PHOTOREAL(
        label = "Photoreal",
        promptSuffix = "photorealistic, ultra detailed, natural lighting, 50mm lens, sharp focus, high dynamic range"
    ),
    CINEMATIC(
        label = "Cinematic",
        promptSuffix = "cinematic still, anamorphic lighting, volumetric haze, dramatic contrast, film grain"
    ),
    ANIME(
        label = "Anime",
        promptSuffix = "anime illustration, clean line art, cel shading, expressive eyes, vibrant palette"
    ),
    DIGITAL_ART(
        label = "Digital Art",
        promptSuffix = "digital illustration, painterly brushwork, rich colour harmony, concept art quality"
    ),
    FANTASY(
        label = "Fantasy & Sci-Fi",
        promptSuffix = "epic fantasy concept art, otherworldly atmosphere, intricate detail, dramatic scale"
    ),
    LANDSCAPE(
        label = "Landscape",
        promptSuffix = "wide landscape vista, golden hour light, atmospheric depth, ultra sharp detail"
    ),
    PORTRAIT(
        label = "Portrait",
        promptSuffix = "studio portrait, soft key light, shallow depth of field, skin detail, 85mm lens"
    ),
    WALLPAPER(
        label = "Wallpaper",
        promptSuffix = "phone wallpaper composition, centred subject, clean negative space, high contrast, 9:16 friendly"
    ),
    PRODUCT(
        label = "Product",
        promptSuffix = "product photography, seamless studio backdrop, softbox lighting, crisp reflections, catalogue quality"
    ),
    ABSTRACT(
        label = "Abstract 3D",
        promptSuffix = "abstract 3d render, glossy materials, subsurface scattering, octane style, geometric elegance"
    );

    companion object {
        fun fromName(name: String?): StylePreset =
            entries.firstOrNull { it.name == name } ?: PHOTOREAL
    }
}

/**
 * Aspect ratios the bundled pipeline can produce without distorting the latent grid.
 * Every value is a multiple of 64 on both axes, which is what the VAE requires.
 */
enum class AspectRatio(val label: String, val width: Int, val height: Int) {
    SQUARE("1:1", 512, 512),
    PORTRAIT_4_5("4:5", 448, 576),
    PORTRAIT_9_16("9:16", 384, 640),
    LANDSCAPE_16_9("16:9", 640, 384),
    LANDSCAPE_3_2("3:2", 576, 384);

    val latentWidth: Int get() = width / 8
    val latentHeight: Int get() = height / 8
}

/** Speed/quality tiers, expressed purely in sampler settings the model genuinely supports. */
enum class QualityTier(
    val label: String,
    val steps: Int,
    val guidance: Float,
    /** Multiplier applied to the base resolution when memory allows. */
    val resolutionScale: Float
) {
    FAST("Fast", steps = 4, guidance = 1.0f, resolutionScale = 0.75f),
    BALANCED("Balanced", steps = 6, guidance = 1.4f, resolutionScale = 1.0f),
    QUALITY("Quality", steps = 8, guidance = 1.9f, resolutionScale = 1.0f)
}

data class GenerationRequest(
    val prompt: String,
    val negativePrompt: String = "",
    val style: StylePreset = StylePreset.PHOTOREAL,
    val aspect: AspectRatio = AspectRatio.SQUARE,
    val quality: QualityTier = QualityTier.BALANCED,
    val seed: Long = -1L,
    val width: Int = aspect.width,
    val height: Int = aspect.height,
    val steps: Int = quality.steps,
    val guidance: Float = quality.guidance
) {
    /** Deterministic seed: -1 means "derive one and remember it". */
    fun resolvedSeed(): Long = if (seed >= 0) seed else System.nanoTime() and 0x7FFFFFFFL
}
