package ai.pixelmind

import ai.pixelmind.ui.PixelMindViewModel
import ai.pixelmind.ui.nav.PixelMindNav
import ai.pixelmind.ui.theme.LiquidBackground
import ai.pixelmind.ui.theme.PixelMindTheme
import android.os.Bundle
import android.os.Process
import android.util.Log
import android.view.Choreographer
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.luminance
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val vm: PixelMindViewModel = viewModel(factory = PixelMindViewModel.Factory)
            val themeMode by vm.themeMode.collectAsState()

            PixelMindTheme(themeMode = themeMode) {
                val isDark = MaterialTheme.colorScheme.background.luminance() < 0.5f
                LiquidBackground(isDark = isDark) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        Box(modifier = Modifier.statusBarsPadding()) {
                            PixelMindNav(viewModel = vm)
                        }
                        TransientMessages(vm, Modifier.align(Alignment.BottomCenter))
                    }
                }
            }
        }
        // Logged, never shown: the one number in this app that cannot be measured without a
        // device. See docs/MEASUREMENTS.md — the hook exists, the figure does not.
        logTimeToFirstFrame()
    }

    /**
     * Reports how long the first frame took, from process start rather than from `onCreate`.
     *
     * [Process.getStartUptimeMillis] is when the zygote forked this process, so the number covers
     * class loading, `Application.onCreate` (which is where the engine used to start loading the
     * model), activity creation and the first composition. It is written to logcat under the
     * `ColdStart` tag; nothing on screen reads it.
     */
    private fun logTimeToFirstFrame() {
        val processStart = Process.getStartUptimeMillis()
        Choreographer.getInstance().postFrameCallback { frameTimeNanos ->
            val firstFrameMs = frameTimeNanos / 1_000_000L - processStart
            Log.i("ColdStart", "first frame ${firstFrameMs}ms after process start")
        }
    }
}

@Composable
private fun TransientMessages(vm: PixelMindViewModel, modifier: Modifier = Modifier) {
    val message by vm.message.collectAsState()
    val hostState = remember { SnackbarHostState() }
    LaunchedEffect(message) {
        val text = message ?: return@LaunchedEffect
        hostState.showSnackbar(text)
        vm.consumeMessage()
    }
    SnackbarHost(hostState = hostState, modifier = modifier) { data ->
        Snackbar(
            snackbarData = data,
            containerColor = MaterialTheme.colorScheme.surface,
            contentColor = MaterialTheme.colorScheme.onSurface
        )
    }
}
