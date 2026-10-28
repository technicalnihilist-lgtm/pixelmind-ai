package ai.pixelmind.ui.nav

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import ai.pixelmind.ui.PixelMindViewModel
import ai.pixelmind.ui.screens.GalleryScreen
import ai.pixelmind.ui.screens.HomeScreen
import ai.pixelmind.ui.screens.SettingsScreen
import ai.pixelmind.ui.screens.StudioScreen
import ai.pixelmind.ui.theme.Palette
import ai.pixelmind.ui.theme.Radii
import ai.pixelmind.ui.theme.glassSurface

private enum class Dest(val route: String, val label: String, val icon: ImageVector) {
    HOME("home", "Home", Icons.Filled.Home),
    STUDIO("studio", "Studio", Icons.Filled.PlayArrow),
    GALLERY("gallery", "Gallery", Icons.Filled.Favorite),
    SETTINGS("settings", "Settings", Icons.Filled.Settings)
}

@Composable
fun PixelMindNav(viewModel: PixelMindViewModel, navController: NavHostController = rememberNavController()) {
    val backStack by navController.currentBackStackEntryAsState()
    val currentRoute = backStack?.destination?.route ?: Dest.HOME.route

    Column(modifier = Modifier.fillMaxWidth()) {
        NavHost(
            navController = navController,
            startDestination = Dest.HOME.route,
            modifier = Modifier.weight(1f),
            enterTransition = { fadeIn() + slideInHorizontally { it / 14 } },
            exitTransition = { fadeOut() + slideOutHorizontally { -it / 14 } }
        ) {
            composable(Dest.HOME.route) {
                HomeScreen(
                    viewModel = viewModel,
                    onOpenStudio = { navController.navigate(Dest.STUDIO.route) },
                    onOpenGallery = { navController.navigate(Dest.GALLERY.route) }
                )
            }
            composable(Dest.STUDIO.route) { StudioScreen(viewModel) }
            composable(Dest.GALLERY.route) { GalleryScreen(viewModel) }
            composable(Dest.SETTINGS.route) { SettingsScreen(viewModel) }
        }

        GlassNavBar(
            current = Dest.entries.first { it.route == currentRoute },
            onSelect = { dest ->
                if (dest.route != currentRoute) {
                    navController.navigate(dest.route) {
                        popUpTo(Dest.HOME.route) { saveState = true }
                        launchSingleTop = true
                        restoreState = true
                    }
                }
            }
        )
    }
}

@Composable
private fun GlassNavBar(current: Dest, onSelect: (Dest) -> Unit) {
    val isDark = MaterialTheme.colorScheme.background.luminance() < 0.5f
    val shape = RoundedCornerShape(Radii.card)

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .padding(horizontal = 18.dp, vertical = 10.dp)
            .glassSurface(shape = shape, isDark = isDark, elevation = 24.dp)
            .padding(horizontal = 8.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Dest.entries.forEach { dest ->
            NavItem(
                dest = dest,
                selected = dest == current,
                isDark = isDark,
                onClick = { onSelect(dest) }
            )
        }
    }
}

@Composable
private fun NavItem(dest: Dest, selected: Boolean, isDark: Boolean, onClick: () -> Unit) {
    val scale by animateFloatAsState(
        targetValue = if (selected) 1f else 0.94f,
        animationSpec = spring(dampingRatio = 0.6f, stiffness = 700f),
        label = "nav"
    )
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clip(RoundedCornerShape(Radii.control))
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 6.dp)
    ) {
        Box(
            modifier = Modifier
                .scale(scale)
                .size(width = 44.dp, height = 30.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(
                    if (selected) {
                        Brush.horizontalGradient(
                            listOf(
                                Palette.Indigo.copy(alpha = 0.85f),
                                Palette.Violet.copy(alpha = 0.75f)
                            )
                        )
                    } else {
                        Brush.horizontalGradient(listOf(Color.Transparent, Color.Transparent))
                    }
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = dest.icon,
                contentDescription = dest.label,
                tint = if (selected) Color.White else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(19.dp)
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(
            dest.label,
            style = MaterialTheme.typography.labelSmall,
            color = if (selected) MaterialTheme.colorScheme.onSurface
            else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f)
        )
    }
}
