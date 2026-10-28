package ai.pixelmind.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import ai.pixelmind.data.GenerationRecord
import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.engine.StylePreset
import ai.pixelmind.ui.PixelMindViewModel
import ai.pixelmind.ui.components.EmptyState
import ai.pixelmind.ui.theme.GlassCard
import ai.pixelmind.ui.components.GradientButton
import ai.pixelmind.ui.components.LocalImage
import ai.pixelmind.ui.components.PromptField
import ai.pixelmind.ui.components.SectionHeader
import ai.pixelmind.ui.components.SelectablePill
import ai.pixelmind.ui.components.StatRow
import ai.pixelmind.ui.components.StatusPill
import ai.pixelmind.ui.theme.Eyebrow
import ai.pixelmind.ui.components.GlassDivider
import ai.pixelmind.ui.theme.Palette
import ai.pixelmind.ui.theme.Radii

@Composable
fun HomeScreen(
    viewModel: PixelMindViewModel,
    onOpenStudio: () -> Unit,
    onOpenGallery: () -> Unit
) {
    val isDark = MaterialTheme.colorScheme.background.luminance() < 0.5f
    val studio by viewModel.studio.collectAsState()
    val status by viewModel.engineStatus.collectAsState()
    val generations by viewModel.generations.collectAsState()
    val progress by viewModel.progress.collectAsState()
    val modelInfo by viewModel.modelInfo.collectAsState()

    val busy = progress != null

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
    ) {
        Spacer(Modifier.height(18.dp))

        // ---- brand row ------------------------------------------------------------------
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(34.dp)
                        .clip(RoundedCornerShape(11.dp))
                        .background(Brush.linearGradient(Palette.AccentGradient)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.Filled.Star,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                }
                Spacer(Modifier.width(11.dp))
                Text("PixelMind", style = MaterialTheme.typography.titleLarge)
            }
            EngineStatusPill(status)
        }

        Spacer(Modifier.height(30.dp))

        Text(
            "Imagine it.\nSee it in seconds.",
            style = MaterialTheme.typography.displayLarge,
            color = MaterialTheme.colorScheme.onSurface
        )
        Spacer(Modifier.height(10.dp))
        Text(
            "Every pixel is generated on this device. No account, no server, no upload.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(Modifier.height(24.dp))

        // ---- prompt + generate -----------------------------------------------------------
        GlassCard(isDark = isDark, padding = PaddingValues(16.dp)) {
            PromptField(
                value = studio.prompt,
                onValueChange = viewModel::setPrompt,
                placeholder = "A snow leopard on a neon-lit rooftop at dusk, cinematic…",
                isDark = isDark,
                minHeight = 104.dp
            )
            Spacer(Modifier.height(14.dp))
            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(end = 4.dp)
            ) {
                items(StylePreset.entries.toList()) { preset ->
                    SelectablePill(
                        text = preset.label,
                        selected = studio.style == preset,
                        onClick = { viewModel.setStyle(preset) },
                        isDark = isDark
                    )
                }
            }
            Spacer(Modifier.height(16.dp))
            GradientButton(
                text = if (busy) "Generating on device…" else "Generate",
                onClick = {
                    if (busy) onOpenStudio() else viewModel.generate()
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = studio.canGenerate || busy,
                icon = Icons.Filled.PlayArrow
            )
        }

        // ---- honest model state ----------------------------------------------------------
        AnimatedVisibility(visible = status is EngineStatus.WeightsMissing, enter = fadeIn(), exit = fadeOut()) {
            Column {
                Spacer(Modifier.height(14.dp))
                GlassCard(isDark = isDark, tint = Palette.Amber, padding = PaddingValues(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            Icons.Filled.Info,
                            contentDescription = null,
                            tint = Palette.Amber,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(Modifier.width(10.dp))
                        Text("Model weights not installed", style = MaterialTheme.typography.titleMedium)
                    }
                    Spacer(Modifier.height(8.dp))
                    Text(
                        (status as? EngineStatus.WeightsMissing)?.detail
                            ?: "The bundled weights are missing from this build.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "PixelMind will not substitute stock images. Confirm this build contains " +
                            "app/src/main/assets/models — see docs/GETTING_IT_RUNNING.md.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        Spacer(Modifier.height(26.dp))

        // ---- recent -----------------------------------------------------------------------
        SectionHeader(
            title = "Recent",
            trailing = {
                Text(
                    "Gallery",
                    style = MaterialTheme.typography.labelLarge,
                    color = Palette.Cyan,
                    modifier = Modifier.clickable(onClick = onOpenGallery)
                )
            }
        )
        Spacer(Modifier.height(12.dp))

        if (generations.isEmpty()) {
            GlassCard(isDark = isDark, padding = PaddingValues(0.dp)) {
                EmptyState(
                    title = "Nothing generated yet",
                    body = "Your first image will appear here the moment the local engine finishes.",
                    icon = Icons.Filled.Star
                )
            }
        } else {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                items(generations.take(8)) { record ->
                    RecentTile(record, isDark)
                }
            }
        }

        Spacer(Modifier.height(26.dp))

        // ---- local AI status ---------------------------------------------------------------
        GlassCard(isDark = isDark) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Eyebrow("Local AI")
                EngineStatusPill(status)
            }
            Spacer(Modifier.height(12.dp))
            GlassDivider(isDark = isDark)
            Spacer(Modifier.height(6.dp))
            StatRow("Model", modelInfo?.name ?: "Not loaded")
            StatRow("Precision", modelInfo?.precision?.uppercase() ?: "—")
            StatRow("Runtime", (status as? EngineStatus.Ready)?.accelerator ?: "CPU")
            StatRow(
                "Weights on disk",
                modelInfo?.let { "%.1f MB".format(it.bytesOnDisk / 1_048_576.0) } ?: "—"
            )
        }

        Spacer(Modifier.height(110.dp))
    }
}

@Composable
fun EngineStatusPill(status: EngineStatus) {
    val (label, color) = when (status) {
        is EngineStatus.Ready -> "Local AI · ${status.accelerator}" to Palette.Mint
        EngineStatus.Loading -> "Loading engine" to Palette.Amber
        is EngineStatus.WeightsMissing -> "Weights missing" to Palette.Amber
        is EngineStatus.Failed -> "Engine error" to Palette.Danger
        EngineStatus.Uninitialised -> "Standby" to Palette.Cyan
    }
    StatusPill(label = label, dotColor = color)
}

@Composable
private fun RecentTile(record: GenerationRecord, isDark: Boolean) {
    Column(modifier = Modifier.width(132.dp)) {
        LocalImage(
            path = record.filePath,
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f),
            cornerRadius = Radii.panel,
            maxEdge = 360,
            contentDescription = record.prompt
        )
        Spacer(Modifier.height(8.dp))
        Text(
            record.prompt,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}
