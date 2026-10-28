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
     */
    private fun observeLifecycle() {
        ProcessLifecycleOwner.get().lifecycle.addObserver(object : DefaultLifecycleObserver {
            override fun onStop(owner: LifecycleOwner) {
                container.generationManager.releaseEngine()
            }
        })
    }
}
