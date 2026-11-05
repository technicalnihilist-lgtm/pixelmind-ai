package ai.pixelmind

import ai.pixelmind.di.AppContainer
import android.app.Application
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ProcessLifecycleOwner

class PixelMindApp : Application() {

    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this)
        container.bootstrap()
        observeLifecycle()
    }

    /**
     * The weights are the single biggest consumer of RAM in this process. Holding them while
     * the app sits in the background invites the low-memory killer; dropping the sessions when
     * the last activity stops keeps the footprint honest.
     *
     * This is [ProcessLifecycleOwner], not the activity lifecycle, so a rotation or any other
     * configuration change does not fire it — the in-flight job keeps running and the retained
     * ViewModel keeps watching it. `onStop` can still be delivered more than once in a process
     * (a second background transition, a lifecycle re-dispatch after the observer is re-added),
     * so [ai.pixelmind.manager.GenerationManager.releaseEngine] is the thing that makes the
     * observer idempotent: a release with nothing loaded is a no-op rather than another
     * `engine.release()` queued behind the engine lock.
     */
    private fun observeLifecycle() {
        ProcessLifecycleOwner.get().lifecycle.addObserver(object : DefaultLifecycleObserver {
            override fun onStop(owner: LifecycleOwner) {
                container.generationManager.releaseEngine()
            }
        })
    }
}
