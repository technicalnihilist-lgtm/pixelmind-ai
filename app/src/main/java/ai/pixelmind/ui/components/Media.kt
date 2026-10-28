package ai.pixelmind.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.Image
import ai.pixelmind.PixelMindApp
import ai.pixelmind.ui.theme.Palette
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Draws a generated PNG from local storage. Deliberately not an image-loading library:
 * the files are already on disk at a known size, and the store keeps a decode cache.
 */
@Composable
fun LocalImage(
    path: String,
    modifier: Modifier = Modifier,
    contentDescription: String? = null,
    cornerRadius: Dp = 20.dp,
    maxEdge: Int = 1080,
    contentScale: ContentScale = ContentScale.Crop
) {
    val context = LocalContext.current
    val store = remember(context) { (context.applicationContext as PixelMindApp).container.imageStore }
    var bitmap by remember(path) { mutableStateOf<android.graphics.Bitmap?>(null) }

    LaunchedEffect(path, maxEdge) {
        bitmap = withContext(Dispatchers.IO) { store.load(path, maxEdge) }
    }

    Box(
        modifier = modifier
            .clip(RoundedCornerShape(cornerRadius))
            .background(
                Brush.linearGradient(
                    listOf(
                        Palette.Indigo.copy(alpha = 0.18f),
                        Palette.Cyan.copy(alpha = 0.10f),
                        Palette.Rose.copy(alpha = 0.12f)
                    )
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        bitmap?.let { bmp ->
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = contentDescription,
                modifier = Modifier.fillMaxSize(),
                contentScale = contentScale
            )
        }
    }
}

/**
 * Generation progress as a ring. The sweep is animated so it glides between steps instead of
 * snapping, which matters when a single step takes several seconds on a mid-range CPU.
 */
@Composable
fun ProgressRing(
    fraction: Float,
    modifier: Modifier = Modifier,
    size: Dp = 132.dp,
    strokeWidth: Dp = 9.dp,
    label: String? = null,
    isDark: Boolean = true
) {
    val animated by animateFloatAsState(
        targetValue = fraction.coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 600),
        label = "ring"
    )
    val trackColor = Color.White.copy(alpha = if (isDark) 0.10f else 0.55f)

    Box(modifier = modifier.size(size), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val stroke = strokeWidth.toPx()
            val inset = stroke / 2f
            val arcSize = Size(this.size.width - stroke, this.size.height - stroke)
            drawArc(
                color = trackColor,
                startAngle = -90f,
                sweepAngle = 360f,
                useCenter = false,
                topLeft = Offset(inset, inset),
                size = arcSize,
                style = Stroke(width = stroke, cap = StrokeCap.Round)
            )
            if (animated > 0f) {
                drawArc(
                    brush = Brush.sweepGradient(
                        listOf(Palette.Indigo, Palette.Violet, Palette.Cyan, Palette.Indigo)
                    ),
                    startAngle = -90f,
                    sweepAngle = 360f * animated,
                    useCenter = false,
                    topLeft = Offset(inset, inset),
                    size = arcSize,
                    style = Stroke(width = stroke, cap = StrokeCap.Round)
                )
            }
        }
        Box(contentAlignment = Alignment.Center) {
            Text(
                text = label ?: "${(animated * 100).toInt()}%",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }
}

/** Skeleton shimmer used while the first decode of a thumbnail is in flight. */
@Composable
fun GlassSkeleton(modifier: Modifier = Modifier, cornerRadius: Dp = 20.dp) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(cornerRadius))
            .background(
                Brush.linearGradient(
                    listOf(
                        Color.White.copy(alpha = 0.06f),
                        Color.White.copy(alpha = 0.12f),
                        Color.White.copy(alpha = 0.06f)
                    )
                )
            )
    )
}
