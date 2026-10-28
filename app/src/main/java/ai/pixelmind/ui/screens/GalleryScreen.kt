package ai.pixelmind.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import ai.pixelmind.data.GenerationRecord
import ai.pixelmind.ui.PixelMindViewModel
import ai.pixelmind.ui.components.EmptyState
import ai.pixelmind.ui.components.GlassButton
import ai.pixelmind.ui.theme.GlassCard
import ai.pixelmind.ui.components.GlassIconButton
import ai.pixelmind.ui.components.LocalImage
import ai.pixelmind.ui.components.SelectablePill
import ai.pixelmind.ui.theme.Eyebrow
import ai.pixelmind.ui.theme.Palette
import ai.pixelmind.ui.theme.Radii
import ai.pixelmind.util.ExportUtil
import ai.pixelmind.util.ShareUtil
import java.io.File

private enum class GalleryFilter { ALL, FAVORITES }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GalleryScreen(viewModel: PixelMindViewModel) {
    val isDark = MaterialTheme.colorScheme.background.luminance() < 0.5f
    val context = LocalContext.current
    val generations by viewModel.generations.collectAsState()
    val history by viewModel.promptHistory.collectAsState()

    var filter by remember { mutableStateOf(GalleryFilter.ALL) }
    var selected by remember { mutableStateOf<GenerationRecord?>(null) }

    val visible = when (filter) {
        GalleryFilter.ALL -> generations
        GalleryFilter.FAVORITES -> generations.filter { it.favorite }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.padding(horizontal = 20.dp)) {
            Spacer(Modifier.height(18.dp))
            Text("Gallery", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(4.dp))
            Text(
                "${generations.size} images on this device",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(16.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SelectablePill(
                    text = "All",
                    selected = filter == GalleryFilter.ALL,
                    onClick = { filter = GalleryFilter.ALL },
                    isDark = isDark
                )
                SelectablePill(
                    text = "Favorites",
                    selected = filter == GalleryFilter.FAVORITES,
                    onClick = { filter = GalleryFilter.FAVORITES },
                    isDark = isDark
                )
            }
            if (history.isNotEmpty()) {
                Spacer(Modifier.height(18.dp))
                Eyebrow("Prompt history")
                Spacer(Modifier.height(10.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(history) { prompt ->
                        SelectablePill(
                            text = prompt.take(34),
                            selected = false,
                            onClick = { viewModel.applyPrompt(prompt) },
                            isDark = isDark
                        )
                    }
                }
            }
            Spacer(Modifier.height(18.dp))
        }

        if (visible.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
                GlassCard(
                    modifier = Modifier.padding(horizontal = 20.dp),
                    isDark = isDark,
                    padding = PaddingValues(0.dp)
                ) {
                    EmptyState(
                        title = if (filter == GalleryFilter.FAVORITES) "No favorites yet" else "No images yet",
                        body = "Everything you generate is stored privately in this app's sandbox.",
                        icon = Icons.Filled.Star
                    )
                }
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                contentPadding = PaddingValues(start = 20.dp, end = 20.dp, bottom = 110.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(visible, key = { it.id }) { record ->
                    GalleryTile(
                        record = record,
                        isDark = isDark,
                        onClick = { selected = record },
                        onFavorite = { viewModel.toggleFavorite(record) }
                    )
                }
            }
        }
    }

    selected?.let { record ->
        ModalBottomSheet(
            onDismissRequest = { selected = null },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = MaterialTheme.colorScheme.surface,
            dragHandle = null
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Eyebrow("Generated offline")
                    GlassIconButton(
                        icon = Icons.Filled.Close,
                        onClick = { selected = null },
                        isDark = isDark,
                        size = 36.dp
                    )
                }
                Spacer(Modifier.height(14.dp))
                LocalImage(
                    path = record.filePath,
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(record.width.toFloat() / record.height.toFloat()),
                    cornerRadius = Radii.card
                )
                Spacer(Modifier.height(16.dp))
                Text(record.prompt, style = MaterialTheme.typography.bodyLarge)
                Spacer(Modifier.height(10.dp))
                Text(
                    "${record.width}×${record.height} · ${record.steps} steps · seed ${record.seed} · " +
                        "${record.durationMs / 1000}s",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(18.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    GlassButton(
                        text = if (ExportUtil.canExportToGallery) "Save" else "Share",
                        onClick = {
                            if (ExportUtil.canExportToGallery) {
                                ExportUtil.exportToPictures(context, File(record.filePath))
                            } else {
                                ShareUtil.shareImage(context, File(record.filePath))
                            }
                        },
                        isDark = isDark,
                        modifier = Modifier.weight(1f)
                    )
                    GlassButton(
                        text = "Share",
                        onClick = { ShareUtil.shareImage(context, File(record.filePath)) },
                        isDark = isDark,
                        icon = Icons.Filled.Share,
                        modifier = Modifier.weight(1f)
                    )
                    GlassIconButton(
                        icon = Icons.Filled.Delete,
                        onClick = {
                            viewModel.delete(record)
                            selected = null
                        },
                        isDark = isDark,
                        tint = Palette.Danger
                    )
                }
                Spacer(Modifier.height(10.dp))
                GlassButton(
                    text = "Reuse prompt",
                    onClick = {
                        viewModel.reusePrompt(record)
                        selected = null
                    },
                    isDark = isDark,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(24.dp))
            }
        }
    }
}

@Composable
private fun GalleryTile(
    record: GenerationRecord,
    isDark: Boolean,
    onClick: () -> Unit,
    onFavorite: () -> Unit
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(Radii.panel))
            .clickable(onClick = onClick)
    ) {
        Column {
            LocalImage(
                path = record.filePath,
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f),
                cornerRadius = Radii.panel,
                maxEdge = 480
            )
            Spacer(Modifier.height(7.dp))
            Text(
                record.prompt,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(8.dp)
                .size(30.dp)
                .clip(CircleShape)
                .background(Color.Black.copy(alpha = 0.38f))
                .clickable(onClick = onFavorite),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (record.favorite) Icons.Filled.Favorite else Icons.Filled.FavoriteBorder,
                contentDescription = null,
                tint = if (record.favorite) Palette.Rose else Color.White.copy(alpha = 0.85f),
                modifier = Modifier.size(15.dp)
            )
        }
    }
}
