package ai.pixelmind.ui

import ai.pixelmind.data.GenerationRecord
import ai.pixelmind.PixelMindApp
import ai.pixelmind.di.AppContainer
import ai.pixelmind.engine.AspectRatio
import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.engine.GenerationProgress
import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.GenerationResult
import ai.pixelmind.engine.ModelImporter
import ai.pixelmind.engine.ModelInfo
import ai.pixelmind.engine.QualityTier
import ai.pixelmind.engine.RunTimings
import ai.pixelmind.engine.StylePreset
import ai.pixelmind.manager.GenerationManager
import ai.pixelmind.ui.theme.ThemeMode
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class StudioUiState(
    val prompt: String = "",
    val negativePrompt: String = "",
    val style: StylePreset = StylePreset.PHOTOREAL,
    val aspect: AspectRatio = AspectRatio.SQUARE,
    val quality: QualityTier = QualityTier.BALANCED,
    val seed: String = ""
) {
    val canGenerate: Boolean get() = prompt.isNotBlank()
    val estimatedSize: String get() = "${aspect.width} × ${aspect.height}"
}

data class HardwareUiState(
    val memoryClassMb: Int = 0,
    val isLowRam: Boolean = false,
    val accelerator: String = "CPU",
    val cores: Int = 0,
    val abis: String = "",
    val device: String = ""
)

/** Progress and outcome of a model import, surfaced in Settings. */
sealed interface ImportUiState {
    data object Idle : ImportUiState
    data class Working(val message: String) : ImportUiState
    data class Done(val message: String) : ImportUiState
    data class Error(val message: String) : ImportUiState
}

class PixelMindViewModel(private val container: AppContainer) : ViewModel() {

    private val manager: GenerationManager = container.generationManager
    private val repository = container.repository
    private val settings = container.settingsStore

    private val _studio = MutableStateFlow(StudioUiState())
    val studio: StateFlow<StudioUiState> = _studio.asStateFlow()

    val engineStatus: StateFlow<EngineStatus> = manager.engineStatus
    val progress: StateFlow<GenerationProgress?> = manager.progress
    val jobs: StateFlow<List<GenerationManager.Job>> = manager.jobs
    val modelInfo: StateFlow<ModelInfo?> = manager.modelInfo
    val generations: StateFlow<List<GenerationRecord>> = repository.generations
    val promptHistory: StateFlow<List<String>> = repository.promptHistory
    val themeMode: StateFlow<ThemeMode> = settings.theme

    private val _lastImage = MutableStateFlow<String?>(null)
    val lastImage: StateFlow<String?> = _lastImage.asStateFlow()

    private val _importState = MutableStateFlow<ImportUiState>(ImportUiState.Idle)
    val importState: StateFlow<ImportUiState> = _importState.asStateFlow()

    private val _lastTimings = MutableStateFlow<RunTimings?>(null)
    val lastTimings: StateFlow<RunTimings?> = _lastTimings.asStateFlow()

    private val _message = MutableStateFlow<String?>(null)
    val message: StateFlow<String?> = _message.asStateFlow()

    private val _hardware = MutableStateFlow(HardwareUiState())
    val hardware: StateFlow<HardwareUiState> = _hardware.asStateFlow()

    private val _storageBytes = MutableStateFlow(0L)
    val storageBytes: StateFlow<Long> = _storageBytes.asStateFlow()

    private val _modelBytes = MutableStateFlow(0L)
    val modelBytes: StateFlow<Long> = _modelBytes.asStateFlow()

    init {
        refreshStorage()
        loadHardware()
        viewModelScope.launch {
            manager.resultEvents.collect { result ->
                when (result) {
                    is GenerationResult.Success -> {
                        _lastImage.value = result.image.file.absolutePath
                        _lastTimings.value = result.timings
                    }
                    is GenerationResult.Failed -> _message.value = result.reason
                    GenerationResult.Cancelled -> _message.value = "Generation cancelled."
                }
                refreshStorage()
            }
        }
        viewModelScope.launch { repository.refresh() }
    }

    // ---- studio editing -------------------------------------------------------------------

    fun setPrompt(value: String) { _studio.value = _studio.value.copy(prompt = value) }
    fun setNegative(value: String) { _studio.value = _studio.value.copy(negativePrompt = value) }
    fun setStyle(value: StylePreset) { _studio.value = _studio.value.copy(style = value) }
    fun setAspect(value: AspectRatio) { _studio.value = _studio.value.copy(aspect = value) }
    fun setQuality(value: QualityTier) { _studio.value = _studio.value.copy(quality = value) }
    fun setSeed(value: String) {
        _studio.value = _studio.value.copy(seed = value.filter { it.isDigit() }.take(10))
    }

    fun applyPrompt(prompt: String) {
        _studio.value = _studio.value.copy(prompt = prompt)
    }

    fun setTheme(mode: ThemeMode) = settings.setTheme(mode)

    fun consumeMessage() { _message.value = null }

    fun clearLastImage() { _lastImage.value = null }

    // ---- generation -----------------------------------------------------------------------

    fun generate() {
        val state = _studio.value
        if (!state.canGenerate) {
            _message.value = "Describe the image you want first."
            return
        }
        val seed = state.seed.toLongOrNull() ?: -1L
        val request = GenerationRequest(
            prompt = state.prompt.trim(),
            negativePrompt = state.negativePrompt.trim(),
            style = state.style,
            aspect = state.aspect,
            quality = state.quality,
            seed = seed,
            width = state.aspect.width,
            height = state.aspect.height,
            steps = state.quality.steps,
            guidance = state.quality.guidance
        )
        viewModelScope.launch {
            val status = manager.ensureReady()
            if (status !is EngineStatus.Ready) {
                _message.value = when (status) {
                    is EngineStatus.WeightsMissing -> status.detail
                    is EngineStatus.Failed -> "Local engine error: ${status.detail}"
                    else -> "The local engine is still starting."
                }
                return@launch
            }
            manager.enqueue(request)
        }
    }

    fun cancel() = manager.cancelActive()

    fun retryLast() {
        val last = manager.jobs.value.lastOrNull { it.state == GenerationManager.JobState.FAILED }
        if (last != null) manager.retry(last.id) else generate()
    }

    fun retryJob(id: Long) = manager.retry(id)

    // ---- gallery --------------------------------------------------------------------------

    fun toggleFavorite(record: GenerationRecord) {
        viewModelScope.launch { repository.toggleFavorite(record) }
    }

    fun delete(record: GenerationRecord) {
        viewModelScope.launch {
            repository.delete(record)
            refreshStorage()
            _message.value = "Deleted from this device."
        }
    }

    fun reusePrompt(record: GenerationRecord) {
        _studio.value = _studio.value.copy(
            prompt = record.prompt,
            negativePrompt = record.negativePrompt,
            style = StylePreset.fromName(record.style)
        )
    }

    fun shareImage(record: GenerationRecord, context: android.content.Context) {
        ai.pixelmind.util.ShareUtil.shareImage(context, File(record.filePath))
    }

    // ---- model import ---------------------------------------------------------------------

    fun importModelZip(uri: android.net.Uri) {
        runImport("Unpacking archive…") { container.modelImporter.importZip(uri) }
    }

    fun importModelFiles(uris: List<android.net.Uri>) {
        if (uris.isEmpty()) return
        runImport("Reading ${uris.size} file(s)…") { container.modelImporter.importFiles(uris) }
    }

    fun restoreBundledModel() {
        viewModelScope.launch {
            _importState.value = ImportUiState.Working("Removing the imported model…")
            val result = container.modelImporter.restoreBundled()
            if (result is ModelImporter.Result.Failed) {
                _importState.value = ImportUiState.Error(result.reason)
                return@launch
            }
            when (val status = manager.reloadEngine()) {
                is EngineStatus.Ready -> {
                    _importState.value = ImportUiState.Done("Bundled model restored.")
                    refreshStorage()
                }
                else -> _importState.value = ImportUiState.Error(
                    "Bundled model did not load: ${describe(status)}"
                )
            }
        }
    }

    private fun runImport(initialMessage: String, action: suspend () -> ModelImporter.Result) {
        viewModelScope.launch {
            _importState.value = ImportUiState.Working(initialMessage)
            val result = action()
            if (result is ModelImporter.Result.Failed) {
                _importState.value = ImportUiState.Error(result.reason)
                return@launch
            }
            val imported = result as ModelImporter.Result.Imported
            _importState.value = ImportUiState.Working("Loading ${imported.name}…")

            // The only honest test of imported weights is whether they load. If they do not, put
            // the previous model back rather than leaving the app unable to generate.
            when (val status = manager.reloadEngine()) {
                is EngineStatus.Ready -> {
                    container.modelImporter.discardPrevious()
                    _importState.value = ImportUiState.Done(
                        "${imported.name} ${imported.version} loaded " +
                            "(${imported.bytes / 1_048_576} MB, ${status.accelerator})."
                    )
                    refreshStorage()
                }
                else -> {
                    val rolledBack = container.modelImporter.rollback()
                    manager.reloadEngine()
                    _importState.value = ImportUiState.Error(
                        "That model did not load: ${describe(status)}" +
                            if (rolledBack) " The previous model was restored." else ""
                    )
                }
            }
        }
    }

    private fun describe(status: EngineStatus): String = when (status) {
        is EngineStatus.Failed -> status.detail
        is EngineStatus.WeightsMissing -> status.detail
        EngineStatus.Loading -> "still loading"
        is EngineStatus.Ready -> "ready"
        EngineStatus.Uninitialised -> "not initialised"
    }

    // ---- settings -------------------------------------------------------------------------

    fun refreshStorage() {
        viewModelScope.launch {
            _storageBytes.value = withContext(Dispatchers.IO) { repository.storageBytes() }
            _modelBytes.value = withContext(Dispatchers.IO) { container.modelLocator.storageBytes() }
        }
    }

    private fun loadHardware() {
        val guard = container.memoryGuard
        _hardware.value = HardwareUiState(
            memoryClassMb = guard.memoryClassMb,
            isLowRam = guard.isLowRamDevice,
            accelerator = (engineStatus.value as? EngineStatus.Ready)?.accelerator ?: "CPU",
            cores = Runtime.getRuntime().availableProcessors(),
            abis = android.os.Build.SUPPORTED_ABIS.joinToString(", "),
            device = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} · API ${android.os.Build.VERSION.SDK_INT}"
        )
    }

    companion object {
        val Factory: ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val application = this[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY]
                    ?: error("PixelMind requires an Application instance")
                PixelMindViewModel((application as PixelMindApp).container)
            }
        }
    }
}
