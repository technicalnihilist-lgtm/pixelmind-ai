package ai.pixelmind.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import ai.pixelmind.engine.AspectRatio
import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.engine.GenerationProgress
import ai.pixelmind.engine.GenerationStage
import ai.pixelmind.engine.QualityTier
import ai.pixelmind.engine.StylePreset
import ai.pixelmind.manager.GenerationManager
import ai.pixelmind.ui.PixelMindViewModel
import ai.pixelmind.ui.components.GlassButton
import ai.pixelmind.ui.theme.GlassCard
import ai.pixelmind.ui.components.GlassDivider
import ai.pixelmind.ui.components.GradientButton
import ai.pixelmind.ui.components.LocalImage
import ai.pixelmind.ui.components.ProgressRing
import ai.pixelmind.ui.components.PromptField
import ai.pixelmind.ui.components.SectionHeader
import ai.pixelmind.ui.components.SelectablePill
import ai.pixelmind.ui.theme.Eyebrow
import ai.pixelmind.ui.theme.Palette
import ai.pixelmind.ui.theme.Radii

@Composable
fun StudioScreen(viewModel: PixelMindViewModel) {
    val isDark = MaterialTheme.colorScheme.background.luminance() < 0.5f
    val studio by viewModel.studio.collectAsState()
    val progress by viewModel.progress.collectAsState()
    val jobs by viewModel.jobs.collectAsState()
    val status by viewModel.engineStatus.collectAsState()
    val lastImage by viewModel.lastImage.collectAsState()

    val activeJob = jobs.lastOrNull {
        it.state == GenerationManager.JobState.RUNNING || it.state == GenerationManager.JobState.QUEUED
    }
    val failedJob = jobs.lastOrNull { it.state == GenerationManager.JobState.FAILED }
    val busy = activeJob != null

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
    ) {
        Spacer(Modifier.height(18.dp))
        Text("Studio", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(4.dp))
        Text(
            "Full control over the local pipeline.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(20.dp))

        // ---- prompt ------------------------------------------------------------------------
        GlassCard(isDark = isDark, padding = PaddingValues(16.dp)) {
            Eyebrow("Prompt")
            Spacer(Modifier.height(10.dp))
            PromptField(
                value = studio.prompt,
                onValueChange = viewModel::setPrompt,
                placeholder = "Describe subject, light, lens, mood…",
                isDark = isDark,
                minHeight = 110.dp
            )
            Spacer(Modifier.height(16.dp))
            Eyebrow("Negative prompt")
            Spacer(Modifier.height(10.dp))
            PromptField(
                value = studio.negativePrompt,
                onValueChange = viewModel::setNegative,
                placeholder = "blurry, extra fingers, watermark, text",
                isDark = isDark,
                minHeight = 64.dp,
                textStyle = MaterialTheme.typography.bodyMedium,
                imeAction = ImeAction.Done
            )
        }

        Spacer(Modifier.height(16.dp))

        // ---- style -------------------------------------------------------------------------
        GlassCard(isDark = isDark, padding = PaddingValues(16.dp)) {
            Eyebrow("Style")
            Spacer(Modifier.height(12.dp))
            FlowRowOfPills(
                items = StylePreset.entries.map { it.label },
                selectedIndex = StylePreset.entries.indexOf(studio.style),
                isDark = isDark,
                onSelect = { viewModel.setStyle(StylePreset.entries[it]) }
            )
        }

        Spacer(Modifier.height(16.dp))

        // ---- aspect ------------------------------------------------------------------------
        GlassCard(isDark = isDark, padding = PaddingValues(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Eyebrow("Aspect ratio")
                Text(
                    studio.estimatedSize,
                    style = MaterialTheme.typography.labelSmall,
                    color = Palette.Cyan
                )
            }
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AspectRatio.entries.forEach { ratio ->
                    SelectablePill(
                        text = ratio.label,
                        selected = studio.aspect == ratio,
                        onClick = { viewModel.setAspect(ratio) },
                        isDark = isDark,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        // ---- quality -----------------------------------------------------------------------
        GlassCard(isDark = isDark, padding = PaddingValues(16.dp)) {
            Eyebrow("Speed / quality")
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                QualityTier.entries.forEach { tier ->
                    SelectablePill(
                        text = tier.label,
                        selected = studio.quality == tier,
                        onClick = { viewModel.setQuality(tier) },
                        isDark = isDark,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
            Spacer(Modifier.height(12.dp))
            GlassDivider(isDark = isDark)
            Spacer(Modifier.height(12.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    "${studio.quality.steps} steps · guidance ${"%.1f".format(studio.quality.guidance)}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    "Seed ${studio.seed.ifBlank { "random" }}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Spacer(Modifier.height(12.dp))
            PromptField(
                value = studio.seed,
                onValueChange = viewModel::setSeed,
                placeholder = "Optional fixed seed for reproducible results",
                isDark = isDark,
                minHeight = 52.dp,
                singleLine = true,
                textStyle = MaterialTheme.typography.bodyMedium,
                imeAction = ImeAction.Done
            )
        }

        Spacer(Modifier.height(18.dp))

        // ---- action ------------------------------------------------------------------------
        if (!busy) {
            GradientButton(
                text = "Generate on device",
                onClick = viewModel::generate,
                modifier = Modifier.fillMaxWidth(),
                enabled = studio.canGenerate && status is EngineStatus.Ready
            )
            if (status !is EngineStatus.Ready) {
                Spacer(Modifier.height(10.dp))
                Text(
                    when (val s = status) {
                        is EngineStatus.WeightsMissing -> s.detail
                        is EngineStatus.Failed -> "Engine error: ${s.detail}"
                        EngineStatus.Loading -> "Warming up the local engine…"
                        else -> "Preparing the local engine…"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = Palette.Amber
                )
            }
        }

        // ---- progress ----------------------------------------------------------------------
        AnimatedVisibility(visible = busy, enter = fadeIn(), exit = fadeOut()) {
            Column {
                Spacer(Modifier.height(16.dp))
                GlassCard(isDark = isDark) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        ProgressRing(
                            fraction = progress?.fraction ?: 0f,
                            isDark = isDark,
                            size = 108.dp,
                            strokeWidth = 8.dp
                        )
                        Spacer(Modifier.width(20.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                stageLabel(progress),
                                style = MaterialTheme.typography.titleMedium
                            )
                            Spacer(Modifier.height(4.dp))
                            Text(
                                progress?.let {
                                    "Step ${it.step} of ${it.totalSteps} · ${it.elapsedMs / 1000}s"
                                } ?: "Queued",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(Modifier.height(6.dp))
                            Text(
                                "Running locally · no network",
                                style = MaterialTheme.typography.labelSmall,
                                color = Palette.Mint
                            )
                        }
                    }
                    Spacer(Modifier.height(16.dp))
                    GlassButton(
                        text = "Cancel",
                        onClick = viewModel::cancel,
                        modifier = Modifier.fillMaxWidth(),
                        isDark = isDark,
                        icon = Icons.Filled.Close
                    )
                }
            }
        }

        // ---- failure -----------------------------------------------------------------------
        AnimatedVisibility(visible = !busy && failedJob != null, enter = fadeIn(), exit = fadeOut()) {
            Column {
                Spacer(Modifier.height(16.dp))
                GlassCard(isDark = isDark, tint = Palette.Danger, padding = PaddingValues(16.dp)) {
                    Text("Generation failed", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(6.dp))
                    Text(
                        failedJob?.message ?: "Unknown engine error",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(Modifier.height(14.dp))
                    GlassButton(
                        text = "Retry",
                        onClick = viewModel::retryLast,
                        modifier = Modifier.fillMaxWidth(),
                        isDark = isDark,
                        icon = Icons.Filled.Refresh
                    )
                }
            }
        }

        // ---- result ------------------------------------------------------------------------
        AnimatedVisibility(visible = !busy && lastImage != null, enter = fadeIn(), exit = fadeOut()) {
            Column {
                Spacer(Modifier.height(22.dp))
                SectionHeader(title = "Result")
                Spacer(Modifier.height(12.dp))
                LocalImage(
                    path = lastImage.orEmpty(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(studio.aspect.width.toFloat() / studio.aspect.height.toFloat()),
                    cornerRadius = Radii.card,
                    contentDescription = studio.prompt
                )
            }
        }

        Spacer(Modifier.height(110.dp))
    }
}

private fun stageLabel(progress: GenerationProgress?): String = when (progress?.stage) {
    GenerationStage.PREPARING -> "Preparing latents"
    GenerationStage.ENCODING_TEXT -> "Reading your prompt"
    GenerationStage.DIFFUSING -> "Diffusing"
    GenerationStage.DECODING -> "Decoding image"
    GenerationStage.SAVING -> "Saving"
    GenerationStage.DONE -> "Done"
    null -> "Queued"
}

/**
 * A wrapping row of pills. Compose has no built-in flow layout in this version of the
 * foundation library, so two fixed rows are used — the style list is known and static.
 */
@Composable
private fun FlowRowOfPills(
    items: List<String>,
    selectedIndex: Int,
    isDark: Boolean,
    onSelect: (Int) -> Unit
) {
    val rows = items.chunked(3)
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        rows.forEach { row ->
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                row.forEach { label ->
                    val index = items.indexOf(label)
                    SelectablePill(
                        text = label,
                        selected = index == selectedIndex,
                        onClick = { onSelect(index) },
                        isDark = isDark,
                        modifier = Modifier.weight(1f)
                    )
                }
                repeat(3 - row.size) { Spacer(Modifier.weight(1f)) }
            }
        }
    }
}
