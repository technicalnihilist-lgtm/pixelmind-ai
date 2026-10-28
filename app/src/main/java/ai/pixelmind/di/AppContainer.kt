package ai.pixelmind.di

import android.content.Context
import ai.pixelmind.data.PixelMindDb
import ai.pixelmind.data.PixelMindRepository
import ai.pixelmind.data.SettingsStore
import ai.pixelmind.engine.LocalAIEngine
import ai.pixelmind.engine.MemoryGuard
import ai.pixelmind.engine.ModelLocator
import ai.pixelmind.engine.OnnxDiffusionEngine
import ai.pixelmind.manager.GenerationManager
import ai.pixelmind.util.ImageStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Hand-rolled service locator. A DI framework would add a code-generation step and a few
 * hundred kilobytes for an app that has nine objects.
 */
class AppContainer(context: Context) {

    private val appContext = context.applicationContext

    val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    val imageStore = ImageStore(appContext)
    val memoryGuard = MemoryGuard(appContext)
    val db = PixelMindDb(appContext)
    val repository = PixelMindRepository(db, imageStore)
    val modelLocator = ModelLocator(appContext)
    val settingsStore = SettingsStore(appContext)

    val engine: LocalAIEngine = OnnxDiffusionEngine(
        context = appContext,
        locator = modelLocator,
        memoryGuard = memoryGuard,
        imageStore = imageStore
    )

    val generationManager = GenerationManager(engine, repository, applicationScope)

    fun bootstrap() {
        generationManager.start()
        applicationScope.launch {
            repository.refresh()
            generationManager.ensureReady()
        }
    }
}
