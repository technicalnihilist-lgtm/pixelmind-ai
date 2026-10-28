package ai.pixelmind.data

import android.content.Context
import ai.pixelmind.ui.theme.ThemeMode
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Tiny preferences surface: theme choice only, kept out of SharedPreferences boilerplate. */
class SettingsStore(context: Context) {

    private val prefs = context.getSharedPreferences("pixelmind.settings", Context.MODE_PRIVATE)

    private val _theme = MutableStateFlow(
        runCatching { ThemeMode.valueOf(prefs.getString(KEY_THEME, ThemeMode.SYSTEM.name)!!) }
            .getOrDefault(ThemeMode.SYSTEM)
    )
    val theme: StateFlow<ThemeMode> = _theme.asStateFlow()

    fun setTheme(mode: ThemeMode) {
        prefs.edit().putString(KEY_THEME, mode.name).apply()
        _theme.value = mode
    }

    private companion object {
        const val KEY_THEME = "theme_mode"
    }
}
