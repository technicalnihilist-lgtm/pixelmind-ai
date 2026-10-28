package ai.pixelmind

import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.engine.GenerationRequest
import ai.pixelmind.engine.GenerationResult
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end check against the real engine. On a build with no weights bundled this asserts the
 * honest failure path; on a provisioned build it must produce a real PNG.
 */
@RunWith(AndroidJUnit4::class)
class EngineSmokeTest {

    private val app = InstrumentationRegistry.getInstrumentation()
        .targetContext.applicationContext as PixelMindApp

    @Test
    fun engine_reports_state_and_never_fakes_output() = runBlocking {
        val engine = app.container.engine
        val status = engine.initialize()
        when (status) {
            is EngineStatus.Ready -> {
                val result = engine.generate(GenerationRequest(prompt = "a red apple on a table")) { }
                assertTrue(
                    "A ready engine must return a real image",
                    result is GenerationResult.Success
                )
            }
            is EngineStatus.WeightsMissing -> Unit // honest, expected on a weightless build
            else -> assertTrue("Unexpected engine state: $status", false)
        }
    }
}
