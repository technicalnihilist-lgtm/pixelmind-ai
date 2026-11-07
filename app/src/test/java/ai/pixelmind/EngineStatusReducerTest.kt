package ai.pixelmind

import ai.pixelmind.engine.EngineStatus
import ai.pixelmind.engine.ModelInfo
import ai.pixelmind.manager.EngineStatusReducer
import org.junit.Assert.assertEquals
import org.junit.Test

/** The status pill must never go back to "Loading" over a model that is already in memory. */
class EngineStatusReducerTest {

    private val info = ModelInfo(
        name = "test-model",
        version = "1.0",
        architecture = "lcm",
        precision = "int8",
        bytesOnDisk = 0L,
        license = "test",
        stepsRecommended = 4,
        maxResolution = 512
    )
    private val ready = EngineStatus.Ready(info, "test")

    @Test
    fun `loading never replaces ready`() {
        assertEquals(ready, EngineStatusReducer.next(ready, EngineStatus.Loading))
    }

    @Test
    fun `loading is accepted from an unloaded engine`() {
        assertEquals(
            EngineStatus.Loading,
            EngineStatusReducer.next(EngineStatus.Uninitialised, EngineStatus.Loading)
        )
    }

    @Test
    fun `ready is accepted after loading`() {
        assertEquals(ready, EngineStatusReducer.next(EngineStatus.Loading, ready))
    }

    @Test
    fun `a release is still reported`() {
        assertEquals(
            EngineStatus.Uninitialised,
            EngineStatusReducer.next(ready, EngineStatus.Uninitialised)
        )
    }

    @Test
    fun `a failure is still reported over a ready engine`() {
        val failed = EngineStatus.Failed("the VAE graph rejected the input shape")
        assertEquals(failed, EngineStatusReducer.next(ready, failed))
    }

    @Test
    fun `missing weights are still reported`() {
        val missing = EngineStatus.WeightsMissing("unet.onnx is absent")
        assertEquals(missing, EngineStatusReducer.next(EngineStatus.Uninitialised, missing))
    }
}
