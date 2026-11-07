package ai.pixelmind.manager

import ai.pixelmind.engine.EngineStatus

/**
 * Keeps [GenerationManager.engineStatus] honest about what it claims.
 *
 * The engine is loaded for as long as its sessions exist, so a transition into
 * [EngineStatus.Loading] while the manager already reports [EngineStatus.Ready] is not a load
 * starting — it is a second caller that has not looked at the current state yet. Publishing it
 * flipped the status pill back to "Loading" for a frame in the middle of a running job.
 *
 * Only that one transition is suppressed. Everything else is accepted as-is, including
 * `Ready -> Uninitialised`: releasing the sessions really does make the engine unready, and
 * pretending otherwise would leave the UI claiming a model it no longer has.
 */
internal object EngineStatusReducer {

    fun next(current: EngineStatus, incoming: EngineStatus): EngineStatus = when {
        current is EngineStatus.Ready && incoming == EngineStatus.Loading -> current
        else -> incoming
    }
}
