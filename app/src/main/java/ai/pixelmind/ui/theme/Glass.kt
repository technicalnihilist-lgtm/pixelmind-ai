package ai.pixelmind.ui.theme

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * The ambient layer every screen floats on: three slowly drifting light blooms over a deep
 * base. It animates on a single transition and draws with gradients only — no bitmaps, no
 * shaders — so it costs almost nothing on a low-end GPU.
 */
@Composable
fun LiquidBackground(
    modifier: Modifier = Modifier,
    isDark: Boolean = true,
    content: @Composable BoxScope.() -> Unit
) {
    val transition = rememberInfiniteTransition(label = "ambient")
    val drift by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 26_000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "drift"
    )

    val base = if (isDark) Palette.Ink else Palette.Paper
    val bloomA = if (isDark) Palette.Indigo.copy(alpha = 0.34f) else Palette.Indigo.copy(alpha = 0.16f)
    val bloomB = if (isDark) Palette.Cyan.copy(alpha = 0.20f) else Palette.Cyan.copy(alpha = 0.13f)
    val bloomC = if (isDark) Palette.Rose.copy(alpha = 0.18f) else Palette.Rose.copy(alpha = 0.10f)

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(base)
            .drawBehind {
                val w = size.width
                val h = size.height
                val radius = maxOf(w, h) * 0.85f

                drawRect(
                    brush = Brush.radialGradient(
                        colors = listOf(bloomA, Color.Transparent),
                        center = Offset(w * (0.12f + 0.10f * drift), h * (0.06f + 0.05f * drift)),
                        radius = radius * 0.75f
                    ),
                    size = Size(w, h)
                )
                drawRect(
                    brush = Brush.radialGradient(
                        colors = listOf(bloomB, Color.Transparent),
                        center = Offset(w * (0.92f - 0.12f * drift), h * (0.42f + 0.06f * drift)),
                        radius = radius * 0.62f
                    ),
                    size = Size(w, h)
                )
                drawRect(
                    brush = Brush.radialGradient(
                        colors = listOf(bloomC, Color.Transparent),
                        center = Offset(w * (0.30f + 0.14f * drift), h * (0.94f - 0.05f * drift)),
                        radius = radius * 0.70f
                    ),
                    size = Size(w, h)
                )
            }
    ) {
        content()
    }
}

/**
 * The glass recipe.
 *
 * A translucent vertical gradient (brighter at the top, the way light catches a bevel), a
 * hairline border that fades from a bright top edge to almost nothing at the bottom, and a
 * soft drop shadow that separates the panel from the bloom behind it. This reads as depth
 * without sampling the backdrop, which keeps it cheap and consistent on every API level.
 */
fun Modifier.glassSurface(
    shape: Shape = RoundedCornerShape(Radii.card),
    isDark: Boolean = true,
    tint: Color? = null,
    elevation: Dp = 18.dp,
    borderStrength: Float = 1f
): Modifier {
    val fillTop = (tint ?: Color.White).copy(alpha = if (isDark) 0.085f else 0.72f)
    val fillBottom = (tint ?: Color.White).copy(alpha = if (isDark) 0.035f else 0.42f)
    val borderTop = Color.White.copy(alpha = (if (isDark) 0.34f else 0.95f) * borderStrength)
    val borderBottom = Color.White.copy(alpha = (if (isDark) 0.05f else 0.30f) * borderStrength)
    val shadowTint = if (isDark) Color.Black.copy(alpha = 0.42f) else Palette.Indigo.copy(alpha = 0.12f)

    return this
        .shadow(elevation, shape, ambientColor = shadowTint, spotColor = shadowTint)
        .clip(shape)
        .background(
            Brush.verticalGradient(
                colors = listOf(fillTop, fillBottom)
            )
        )
        .border(
            width = 1.dp,
            brush = Brush.verticalGradient(listOf(borderTop, borderBottom)),
            shape = shape
        )
}

/** A glass panel with the standard inner padding and a specular top edge. */
@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    isDark: Boolean = true,
    corner: Dp = Radii.card,
    tint: Color? = null,
    padding: PaddingValues = PaddingValues(20.dp),
    content: @Composable ColumnScope.() -> Unit
) {
    val shape = remember(corner) { RoundedCornerShape(corner) }
    Box(
        modifier = modifier
            .glassSurface(shape = shape, isDark = isDark, tint = tint)
            .drawBehind {
                // Specular highlight: a short bright run along the top-left edge.
                val inset = 26.dp.toPx()
                drawLine(
                    brush = Brush.horizontalGradient(
                        listOf(
                            Color.White.copy(alpha = if (isDark) 0.42f else 0.9f),
                            Color.Transparent
                        )
                    ),
                    start = Offset(inset, 1.2f),
                    end = Offset(size.width - inset, 1.2f),
                    strokeWidth = 1.4f
                )
            }
    ) {
        Column(modifier = Modifier.padding(padding), content = content)
    }
}

/** Small uppercase eyebrow used above section content. */
@Composable
fun Eyebrow(text: String, modifier: Modifier = Modifier, color: Color? = null) {
    androidx.compose.material3.Text(
        text = text.uppercase(),
        modifier = modifier,
        style = MaterialTheme.typography.labelSmall.copy(letterSpacing = 1.6.sp),
        color = color ?: MaterialTheme.colorScheme.onSurfaceVariant
    )
}
