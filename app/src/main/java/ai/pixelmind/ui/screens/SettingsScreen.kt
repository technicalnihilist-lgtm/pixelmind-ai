package ai.pixelmind.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.unit.dp
import ai.pixelmind.BuildConfig
import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.ui.PixelMindViewModel
import ai.pixelmind.ui.theme.GlassCard
import ai.pixelmind.ui.components.GlassDivider
import ai.pixelmind.ui.components.SelectablePill
import ai.pixelmind.ui.components.StatRow
import ai.pixelmind.ui.theme.Eyebrow
import ai.pixelmind.ui.theme.Palette
import ai.pixelmind.ui.theme.ThemeMode

@Composable
fun SettingsScreen(viewModel: PixelMindViewModel) {
    val isDark = MaterialTheme.colorScheme.background.luminance() < 0.5f
    val theme by viewModel.themeMode.collectAsState()
    val status by viewModel.engineStatus.collectAsState()
    val modelInfo by viewModel.modelInfo.collectAsState()
    val hardware by viewModel.hardware.collectAsState()
    val storage by viewModel.storageBytes.collectAsState()
    val modelBytes by viewModel.modelBytes.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
    ) {
        Spacer(Modifier.height(18.dp))
        Text("Settings", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(4.dp))
        Text(
            "Everything about this install, on the record.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(Modifier.height(20.dp))

        // ---- model --------------------------------------------------------------------------
        GlassCard(isDark = isDark) {
            Eyebrow("Local model")
            Spacer(Modifier.height(12.dp))
            StatRow("Name", modelInfo?.name ?: "Not loaded")
            StatRow("Version", modelInfo?.version ?: "—")
            StatRow("Architecture", modelInfo?.architecture ?: "—")
            StatRow("Precision", modelInfo?.precision?.uppercase() ?: "—")
            StatRow("Recommended steps", modelInfo?.stepsRecommended?.toString() ?: "—")
            StatRow("Max resolution", modelInfo?.let { "${it.maxResolution}px" } ?: "—")
            StatRow(
                "Weights on disk",
                if (modelBytes > 0) formatBytes(modelBytes) else "not provisioned"
            )
            Spacer(Modifier.height(10.dp))
            GlassDivider(isDark = isDark)
            Spacer(Modifier.height(10.dp))
            Text(
                "License: ${modelInfo?.license ?: "CreativeML OpenRAIL-M"}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(6.dp))
            Text(
                "The model ships inside the installation as an install-time asset pack. Nothing is " +
                    "downloaded at launch or on first use.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(Modifier.height(16.dp))

        // ---- hardware ------------------------------------------------------------------------
        GlassCard(isDark = isDark) {
            Eyebrow("Hardware")
            Spacer(Modifier.height(12.dp))
            StatRow("Device", hardware.device)
            StatRow("Runtime", (status as? EngineStatus.Ready)?.accelerator ?: "CPU")
            StatRow("Heap class", "${hardware.memoryClassMb} MB")
            StatRow("Low-RAM mode", if (hardware.isLowRam) "yes" else "no")
            StatRow("CPU cores", hardware.cores.toString())
            StatRow("ABIs", hardware.abis)
            Spacer(Modifier.height(8.dp))
            Text(
                when {
                    status is EngineStatus.Ready -> "This device can run the local pipeline."
                    hardware.memoryClassMb < 256 ->
                        "This device has a small heap class. PixelMind will reduce resolution and steps automatically."
                    else -> "Engine not started yet."
                },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(Modifier.height(16.dp))

        // ---- storage -------------------------------------------------------------------------
        GlassCard(isDark = isDark) {
            Eyebrow("Storage")
            Spacer(Modifier.height(12.dp))
            StatRow("Generated images", formatBytes(storage))
            StatRow("Model weights", if (modelBytes > 0) formatBytes(modelBytes) else "—")
            StatRow("Total app data", formatBytes(storage + modelBytes))
            Spacer(Modifier.height(8.dp))
            Text(
                "Images live in app-private storage and are removed when the app is uninstalled. " +
                    "Use Save in the gallery to copy one into Pictures.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(Modifier.height(16.dp))

        // ---- theme ----------------------------------------------------------------------------
        GlassCard(isDark = isDark) {
            Eyebrow("Appearance")
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ThemeMode.entries.forEach { mode ->
                    SelectablePill(
                        text = mode.name.lowercase().replaceFirstChar { it.uppercase() },
                        selected = theme == mode,
                        onClick = { viewModel.setTheme(mode) },
                        isDark = isDark,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        // ---- privacy ---------------------------------------------------------------------------
        GlassCard(isDark = isDark, tint = Palette.Mint, padding = PaddingValues(20.dp)) {
            Eyebrow("Privacy", color = Palette.Mint)
            Spacer(Modifier.height(12.dp))
            Text("No internet permission", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(6.dp))
            Text(
                "No platform permissions are requested — not storage, not location, and above all " +
                    "not android.permission.INTERNET. The only entry in the merged manifest is an " +
                    "app-scoped signature permission AndroidX adds for receiver registration. " +
                    "Without INTERNET the process cannot open a socket, so prompts and images " +
                    "cannot leave the device — enforced by the OS, not just by intent.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(12.dp))
            GlassDivider(isDark = isDark)
            Spacer(Modifier.height(12.dp))
            StatRow("Analytics", "none")
            StatRow("Accounts", "none")
            StatRow("Cloud calls", "none")
            StatRow("Backups", "disabled")
        }

        Spacer(Modifier.height(16.dp))

        GlassCard(isDark = isDark) {
            Eyebrow("About")
            Spacer(Modifier.height(12.dp))
            StatRow("PixelMind AI", BuildConfig.VERSION_NAME)
            StatRow("Build", BuildConfig.BUILD_TYPE)
            Spacer(Modifier.height(8.dp))
            Text(
                "Generated images are produced by a distilled diffusion model running entirely on " +
                    "this device. There is no fallback to stock imagery.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(Modifier.height(110.dp))
    }
}

private fun formatBytes(bytes: Long): String = when {
    bytes >= 1_073_741_824 -> "%.2f GB".format(bytes / 1_073_741_824.0)
    bytes >= 1_048_576 -> "%.1f MB".format(bytes / 1_048_576.0)
    bytes >= 1024 -> "%.0f KB".format(bytes / 1024.0)
    else -> "$bytes B"
}
