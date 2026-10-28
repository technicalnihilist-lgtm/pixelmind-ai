package ai.pixelmind.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.dp
import ai.pixelmind.ui.theme.Palette
import ai.pixelmind.ui.theme.Radii

/** The primary call to action: a lit gradient pill that physically presses in. */
@Composable
fun GradientButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    icon: ImageVector? = null,
    gradient: List<Color> = Palette.AccentGradient
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) 0.97f else 1f,
        animationSpec = spring(dampingRatio = 0.6f, stiffness = 900f),
        label = "press"
    )
    val alpha = if (enabled) 1f else 0.38f

    Box(
        modifier = modifier
            .scale(scale)
            .height(56.dp)
            .clip(RoundedCornerShape(Radii.control))
            .background(
                Brush.horizontalGradient(
                    gradient.map { it.copy(alpha = if (enabled) 0.95f else 0.4f) }
                )
            )
            .border(
                BorderStroke(1.dp, Color.White.copy(alpha = 0.24f * alpha)),
                RoundedCornerShape(Radii.control)
            )
            .clickable(
                interactionSource = interaction,
                indication = null,
                enabled = enabled,
                onClick = onClick
            ),
        contentAlignment = Alignment.Center
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (icon != null) {
                Icon(icon, contentDescription = null, tint = Color.White, modifier = Modifier.size(19.dp))
                Spacer(Modifier.width(10.dp))
            }
            Text(
                text = text,
                color = Color.White,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                maxLines = 1
            )
        }
    }
}

/** Frosted secondary action. */
@Composable
fun GlassButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    isDark: Boolean = true,
    icon: ImageVector? = null,
    tint: Color? = null,
    enabled: Boolean = true
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) 0.97f else 1f,
        animationSpec = spring(dampingRatio = 0.6f, stiffness = 900f),
        label = "press"
    )
    val content = tint ?: MaterialTheme.colorScheme.onSurface

    Row(
        modifier = modifier
            .scale(scale)
            .height(48.dp)
            .clip(RoundedCornerShape(Radii.control))
            .background(Color.White.copy(alpha = if (isDark) 0.07f else 0.55f))
            .border(
                1.dp,
                Color.White.copy(alpha = if (isDark) 0.16f else 0.9f),
                RoundedCornerShape(Radii.control)
            )
            .clickable(
                interactionSource = interaction,
                indication = null,
                enabled = enabled,
                onClick = onClick
            )
            .padding(horizontal = 18.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, tint = content.copy(alpha = 0.9f), modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(8.dp))
        }
        Text(text, color = content.copy(alpha = 0.92f), style = MaterialTheme.typography.labelLarge)
    }
}

/** Selectable pill used for styles, ratios and quality tiers. */
@Composable
fun SelectablePill(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    isDark: Boolean = true
) {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) 0.96f else 1f,
        animationSpec = spring(dampingRatio = 0.55f, stiffness = 800f),
        label = "pill"
    )
    val shape = RoundedCornerShape(Radii.pill)
    Box(
        modifier = modifier
            .scale(scale)
            .clip(shape)
            .background(
                if (selected) {
                    Brush.horizontalGradient(
                        listOf(Palette.Indigo.copy(alpha = 0.85f), Palette.Violet.copy(alpha = 0.8f))
                    )
                } else {
                    Brush.horizontalGradient(
                        listOf(
                            Color.White.copy(alpha = if (isDark) 0.07f else 0.6f),
                            Color.White.copy(alpha = if (isDark) 0.04f else 0.45f)
                        )
                    )
                }
            )
            .border(
                1.dp,
                if (selected) Color.White.copy(alpha = 0.32f) else Color.White.copy(alpha = if (isDark) 0.12f else 0.8f),
                shape
            )
            .clickable(interactionSource = interaction, indication = null, onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 9.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelLarge,
            color = if (selected) Color.White else MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

/** Circular icon button on glass. */
@Composable
fun GlassIconButton(
    icon: ImageVector,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    isDark: Boolean = true,
    tint: Color? = null,
    size: Dp = 44.dp
) {
    Box(
        modifier = modifier
            .size(size)
            .clip(CircleShape)
            .background(Color.White.copy(alpha = if (isDark) 0.08f else 0.6f))
            .border(1.dp, Color.White.copy(alpha = if (isDark) 0.18f else 0.9f), CircleShape)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            icon,
            contentDescription = null,
            tint = tint ?: MaterialTheme.colorScheme.onSurface.copy(alpha = 0.88f),
            modifier = Modifier.size(size * 0.44f)
        )
    }
}

/** Status dot + label, used for "Local AI" state everywhere. */
@Composable
fun StatusPill(
    label: String,
    dotColor: Color,
    modifier: Modifier = Modifier,
    isDark: Boolean = true,
    pulsing: Boolean = false
) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(Radii.pill))
            .background(Color.White.copy(alpha = if (isDark) 0.07f else 0.6f))
            .border(1.dp, Color.White.copy(alpha = if (isDark) 0.14f else 0.8f), RoundedCornerShape(Radii.pill))
            .padding(horizontal = 12.dp, vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(7.dp)
                .clip(CircleShape)
                .background(dotColor)
        )
        Spacer(Modifier.width(8.dp))
        Text(
            label,
            style = MaterialTheme.typography.labelSmall.copy(letterSpacing = 0.5.sp),
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f)
        )
    }
}

/** Label/value row for the settings and model panels. */
@Composable
fun StatRow(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    valueColor: Color? = null
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 7.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Medium),
            color = valueColor ?: MaterialTheme.colorScheme.onSurface,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.padding(start = 16.dp)
        )
    }
}

/** Thin hairline separator that reads as a fold in the glass. */
@Composable
fun GlassDivider(modifier: Modifier = Modifier, isDark: Boolean = true) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(
                Brush.horizontalGradient(
                    listOf(
                        Color.Transparent,
                        Color.White.copy(alpha = if (isDark) 0.14f else 0.7f),
                        Color.Transparent
                    )
                )
            )
    )
}

@Composable
fun SectionHeader(
    title: String,
    modifier: Modifier = Modifier,
    trailing: (@Composable () -> Unit)? = null
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(title, style = MaterialTheme.typography.titleLarge)
        trailing?.invoke()
    }
}

@Composable
fun EmptyState(
    title: String,
    body: String,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null
) {
    Column(
        modifier = modifier.fillMaxWidth().padding(28.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (icon != null) {
            Icon(
                icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                modifier = Modifier.size(38.dp)
            )
            Spacer(Modifier.height(14.dp))
        }
        Text(title, style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(6.dp))
        Text(
            body,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center
        )
    }
}
