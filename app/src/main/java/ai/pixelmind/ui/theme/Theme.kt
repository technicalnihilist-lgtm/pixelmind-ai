package ai.pixelmind.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat

/**
 * PixelMind's palette: a near-black canvas with three light sources — indigo, cyan and a warm
 * rose — so glass panels have something worth refracting.
 */
object Palette {
    val Indigo = Color(0xFF5B6CFF)
    val Violet = Color(0xFF8B5CF6)
    val Cyan = Color(0xFF22D3EE)
    val Mint = Color(0xFF34D399)
    val Rose = Color(0xFFF472B6)
    val Amber = Color(0xFFFBBF24)

    val Ink = Color(0xFF07070C)
    val InkSoft = Color(0xFF0D0E17)
    val Paper = Color(0xFFF6F7FB)
    val PaperSoft = Color(0xFFEDEFF7)

    val Danger = Color(0xFFFF6B81)

    val AccentGradient = listOf(Indigo, Violet, Cyan)
    val WarmGradient = listOf(Rose, Amber)
}

private val DarkScheme = darkColorScheme(
    primary = Palette.Indigo,
    onPrimary = Color.White,
    secondary = Palette.Cyan,
    onSecondary = Palette.Ink,
    tertiary = Palette.Rose,
    background = Palette.Ink,
    onBackground = Color(0xFFF2F3F9),
    surface = Palette.InkSoft,
    onSurface = Color(0xFFF2F3F9),
    surfaceVariant = Color(0xFF171826),
    onSurfaceVariant = Color(0xFFA7AABE),
    outline = Color(0xFF2A2C3D),
    error = Palette.Danger
)

private val LightScheme = lightColorScheme(
    primary = Color(0xFF4353E0),
    onPrimary = Color.White,
    secondary = Color(0xFF0E7490),
    onSecondary = Color.White,
    tertiary = Color(0xFFBE185D),
    background = Palette.Paper,
    onBackground = Color(0xFF12131B),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF12131B),
    surfaceVariant = Color(0xFFE8EAF3),
    onSurfaceVariant = Color(0xFF545A72),
    outline = Color(0xFFD3D7E6),
    error = Color(0xFFD92D4A)
)

private val PixelTypography = Typography(
    displayLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Light,
        fontSize = 44.sp,
        lineHeight = 50.sp,
        letterSpacing = (-1.2).sp
    ),
    headlineMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 26.sp,
        lineHeight = 32.sp,
        letterSpacing = (-0.6).sp
    ),
    titleLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 19.sp,
        lineHeight = 26.sp,
        letterSpacing = (-0.2).sp
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 15.sp,
        lineHeight = 21.sp
    ),
    bodyLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 15.sp,
        lineHeight = 23.sp
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Normal,
        fontSize = 13.5.sp,
        lineHeight = 20.sp
    ),
    labelLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 13.sp,
        letterSpacing = 0.3.sp
    ),
    labelSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 11.sp,
        letterSpacing = 0.8.sp
    )
)

@Composable
fun PixelMindTheme(
    themeMode: ThemeMode = ThemeMode.SYSTEM,
    content: @Composable () -> Unit
) {
    val dark = when (themeMode) {
        ThemeMode.DARK -> true
        ThemeMode.LIGHT -> false
        ThemeMode.SYSTEM -> isSystemInDarkTheme()
    }
    val scheme = if (dark) DarkScheme else LightScheme
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as? Activity)?.window ?: return@SideEffect
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !dark
            WindowCompat.getInsetsController(window, view).isAppearanceLightNavigationBars = !dark
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                window.isNavigationBarContrastEnforced = false
            }
        }
    }
    MaterialTheme(colorScheme = scheme, typography = PixelTypography, content = content)
}

/** Radius scale, so every rounded surface in the app agrees with every other one. */
object Radii {
    val card = 28.dp
    val panel = 22.dp
    val control = 18.dp
    val pill = 100.dp
}
